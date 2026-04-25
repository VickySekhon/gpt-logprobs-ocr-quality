"""
Runs the entire pipeline from start to finish, generating a single CSV file.
"""

import threading, os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loader import load_text_pair
from logprobs_client import transcribe_with_logprobs
from entropy import token_entropies_from_logprobs, surprisal_from_logprobs
from metrics import cer, levenshtein_distance
from normalization import normalize_text
from utils import (
    get_page_id_from_image,
    write_anomalies,
    compute_pearson,
    compute_spearman,
    flatten_array,
    get_thread_start_and_end,
)
from utils import TOP_K, MAX_PAGES, THREADS, NORMALIZATION_TYPE, OUTPUT_DIRECTORY

anomaly_lock = threading.Lock()


def orchestrate_threads(top_k, max_pages, output, available_threads):
    image_folder = os.path.join(os.getcwd(), "data/images")
    page_ids = np.array(
        [get_page_id_from_image(image) for image in os.listdir(image_folder)]
    )[:max_pages]

    # Get random sample
    # np.random.shuffle(page_ids)

    threads, thread_results = [], [None] * available_threads
    n = len(page_ids)

    for rank in range(available_threads):
        start_index, end_index = get_thread_start_and_end(n, available_threads, rank)
        thread = threading.Thread(
            target=predict_subset,
            args=(
                top_k,
                page_ids[start_index:end_index],
                thread_results,
                rank,
            ),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    data = flatten_array(thread_results)
    df = pd.DataFrame(data)
    df.to_csv(f"{output}/csv/results_k_{top_k}.csv")
    return df


def predict_subset(top_k, page_ids, thread_results, rank):
    thread_id = threading.current_thread().name
    data = []
    for page_id in page_ids:
        image_path, ground_truth_text = load_text_pair(page_id)

        generated_transcript_text, token_logprobs = transcribe_with_logprobs(
            image_path, top_k
        )

        token_surprisals = surprisal_from_logprobs(token_logprobs)
        avg_surprisal_per_token = sum(token_surprisals) / len(token_surprisals)

        token_entropies = token_entropies_from_logprobs(token_logprobs)
        total_bits = sum(token_entropies)
        n_tokens = len(token_entropies)
        avg_bits_per_token = total_bits / n_tokens

        generated_transcript_text_norm, ground_truth_text_norm = normalize_text(
            generated_transcript_text, ground_truth_text, NORMALIZATION_TYPE
        )

        calculated_cer = cer(generated_transcript_text_norm, ground_truth_text_norm)

        if calculated_cer > 1:
            print(
                f"*Found an anomaly* OCR for {page_id} has a CER of {calculated_cer}. Skipping it and going directly to the next page"
            )
            with anomaly_lock:
                write_anomalies(
                    page_id, generated_transcript_text_norm, ground_truth_text_norm
                )
            continue

        calculated_levenshtein = levenshtein_distance(
            generated_transcript_text_norm, ground_truth_text_norm
        )

        row = {
            "page_id": page_id,
            "avg_bits_per_token": avg_bits_per_token,
            "avg_surprisal_per_token": avg_surprisal_per_token,
            "total_bits": total_bits,
            "n_tokens": n_tokens,
            "cer": calculated_cer,
            "levenshtein": calculated_levenshtein,
            "gt_length": len(ground_truth_text_norm),
            "normalization_profile": NORMALIZATION_TYPE,
        }
        data.append(row)
        print(
            f"{thread_id} | Processed excerpt with page id: {page_id} | Total excerpts to process: {len(page_ids)}"
        )

    thread_results[rank] = data


def visualize_correlation_coefficient(x, y, coefficient, top_k):
    plt.figure(figsize=(10, 6))
    ax = sns.regplot(x=x, y=y, ci=None, line_kws={"color": "red"})
    params = {
        "xlabel": "Token",
        "ylabel": f"{coefficient}",
        "title": f"Computed {coefficient} Correlation Coefficient Across Iterations",
    }
    ax.set(**params)
    plt.savefig(f"figures/{coefficient.lower()}_k_{top_k}")


def compute_bootstrap_confidence_interval(
    df: pd.DataFrame, resample_count, sample_size, top_k, indicator
):
    total_r, total_p = [], []
    for _ in range(resample_count):
        sample = df.sample(sample_size, replace=True)
        x, y = sample[f"{indicator}"], sample["cer"]
        r = compute_pearson(x, y)
        p = compute_spearman(x, y)
        total_r.append(r)
        total_p.append(p)
    r_ci_lower_bound = np.percentile(total_r, 2.5)
    r_ci_upper_bound = np.percentile(total_r, 97.5)

    p_ci_lower_bound = np.percentile(total_p, 2.5)
    p_ci_upper_bound = np.percentile(total_p, 97.5)

    # Plot correlation values across bootstrap iterations.
    x = [i for i in range(1, resample_count + 1)]
    visualize_correlation_coefficient(x, total_r, "Pearson", top_k)
    visualize_correlation_coefficient(x, total_p, "Spearman", top_k)

    return r_ci_lower_bound, r_ci_upper_bound, p_ci_lower_bound, p_ci_upper_bound


def main():
    parser = argparse.ArgumentParser(
        description=("Run prediction pipeline on entire BLN600 dataset")
    )

    parser.add_argument(
        "--top-k",
        type=int,
        help="How many top logprobs to consider when computing entropy",
    )
    parser.add_argument(
        "--max-pages", type=int, help="Maximum number of pages to process"
    )
    parser.add_argument("--output", type=str, help="Path (folder) to store output")
    parser.add_argument(
        "--threads",
        type=int,
        help="Number of threads to spawn when processing the dataset",
    )

    args = parser.parse_args()
    top_k = args.top_k or TOP_K
    max_pages = args.max_pages or MAX_PAGES
    output = args.output or OUTPUT_DIRECTORY
    available_threads = args.threads or THREADS

    try:
        orchestrate_threads(top_k, max_pages, output, available_threads)
        print("Execution successful")
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
