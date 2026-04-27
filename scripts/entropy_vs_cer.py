"""
Script to visualize the relationship between entropy and CER using plots from CSV data.
"""

import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import save_figures

from src.utils import TOP_K, OUTPUT_DIRECTORY


def visualize_cer(df, output, indicator="avg_bits_per_token"):
    x, y = df[f"{indicator}"], df["cer"]

    if indicator == "avg_surprisal_per_token":
        indicator = "Average Token Surprisal (Bits/Token)"
    else:
        indicator = "Average Token Entropy (Bits/Token)"

    plt.figure(figsize=(10, 6))
    ax = sns.regplot(x=x, y=y, ci=None, line_kws={"color": "red"})

    params = {
        "xlabel": f"{indicator}",
        "ylabel": "Character Error Rate (CER)",
        # Remove '(Bits/Token)' from crowding the title
        "title": f"Relationship Between {' '.join(indicator.split()[:3])} and Character Error Rate",
    }
    ax.set(**params)
    indicator_name = indicator.split()[2].lower()
    save_figures(plt.gcf(), f"{output}/figures/figure_01_{indicator_name}_vs_cer")


def visualize_entropy_distribution(df, output):
    data = df["avg_bits_per_token"]

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=len(df))

    plt.xlabel("Average Token Entropy (Bits/Token)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Average Token Entropy Across Data")

    save_figures(plt.gcf(), f"{output}/figures/figure_02_entropy_distribution")


def main():
    parser = argparse.ArgumentParser(
        description=("Generate a plot of the relationship between CER and Entropy")
    )

    parser.add_argument(
        "--top-k",
        type=int,
        help="How many top logprobs were considered when computing entropy",
    )
    parser.add_argument("--output", type=str, help="Path (folder) to store output")

    args = parser.parse_args()
    top_k = args.top_k or TOP_K
    output = args.output or OUTPUT_DIRECTORY

    path_to_csv = Path(f"{output}/csv/results_k_{top_k}.csv")

    assert (
        path_to_csv.exists()
    ), f"{path_to_csv} does not exist, please run `make run-all` to generate the results csv file before generating figures."

    try:
        df = pd.read_csv(path_to_csv)
        visualize_cer(df, output)
        visualize_entropy_distribution(df, output)
        print("Execution successful")
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
