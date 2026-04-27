"""
Performs stratified analysis on text data, computing correlations and generating visualizations
to explore the relationship between entropy, surprisal, and CER across different length quartiles.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils import compute_pearson, compute_spearman
from src.regression import get_misclassified_triage_decisions

from src.utils import YOUDEN_J


def stratify_df(df: pd.DataFrame, quartiles=4):
    df["length_quartile"] = pd.qcut(
        df["gt_length"], q=quartiles, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    return df


def plot_stratified_correlations(stratified_df, top_k, output):
    quartiles, pearsons, spearmans, group_sizes = compute_stratified_correlations(
        stratified_df
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(quartiles))
    width = 0.35

    ax.bar(x - width / 2, pearsons, width, label="Pearson")
    ax.bar(x + width / 2, spearmans, width, label="Spearman")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{q}\n(n={n})" for q, n in zip(quartiles, group_sizes)])
    ax.set_xlabel("Ground Truth Length Quartile")
    ax.set_ylabel("Correlation with CER")
    ax.set_title(f"Pearson vs Spearman by Length Quartile, k={top_k}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"{output}/figures/stratified_correlations_k_{top_k}.png", dpi=200)

    plt.close(fig)


def compute_stratified_correlations(
    stratified_df: pd.DataFrame, indicator="avg_bits_per_token"
):
    quartiles, pearsons, spearmans, group_sizes = [], [], [], []

    for quartile, group in stratified_df.groupby("length_quartile", sort=True):
        x, y = group[f"{indicator}"], group["cer"]
        r, p = compute_pearson(x, y), compute_spearman(x, y)
        quartiles.append(str(quartile))
        pearsons.append(r)
        spearmans.append(p)
        group_sizes.append(len(group))

    return quartiles, pearsons, spearmans, group_sizes


def visualize_entropy_and_cer_across_page_lengths(
    stratified_df: pd.DataFrame, top_k, output
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    for ax, (quartile, group) in zip(
        axes.flatten(), stratified_df.groupby("length_quartile")
    ):
        ax.scatter(group["avg_bits_per_token"], group["cer"])
        ax.set_title(f"{quartile} n={len(group)}")
        ax.set_xlabel("Entropy (Average Bits Per Token)")
        ax.set_ylabel("CER")

    fig.tight_layout(pad=2.0)

    plt.suptitle("The Relationship Between CER and Entropy Across Different Lengths")
    plt.savefig(f"{output}/figures/entropy_vs_cer_stratified_k_{top_k}.png", dpi=200)


def visualize_entropy_vs_surprisal_as_predictor(
    df: pd.DataFrame, top_k, output, threshold_type, use_primary=False
):
    correct, val_indices = get_misclassified_triage_decisions(
        top_k, output, use_primary, threshold_type
    )

    # Validation set is 20% of entire dataframe
    val_df = df.loc[val_indices]
    plt.figure(figsize=(10, 6))
    
    plt.scatter(
        val_df[correct]["avg_surprisal_per_token"],
        val_df[correct]["avg_bits_per_token"],
        color="green",
        label="Correct",
        alpha=0.6,
    )
    
    plt.scatter(
        val_df[~correct]["avg_surprisal_per_token"],
        val_df[~correct]["avg_bits_per_token"],
        color="red",
        label="Misclassified",
        alpha=0.6,
    )
    
    plt.legend()
    plt.xlabel("Surprisal (Average)")
    plt.ylabel("Entropy (Average Bits Per Token)")
    plt.title("The Correlation Between Entropy and Surprisal")
    plt.savefig(f"{output}/figures/surprisal_vs_entropy_k_{top_k}.png", dpi=200)


def main():
    parser = argparse.ArgumentParser(
        description=("Generate stratified scatter plot and surprisal vs entropy plot")
    )

    parser.add_argument(
        "--top-k",
        type=int,
        help="How many top logprobs were considered when computing entropy",
    )
    parser.add_argument("--output", type=str, help="Path (folder) to store output")

    args = parser.parse_args()
    top_k = args.top_k or 10
    output = args.output or "results"

    path_to_csv = Path(f"{output}/csv/results_k_{top_k}.csv")

    assert (
        path_to_csv.exists()
    ), f"{path_to_csv} does not exist, please run `make run-all` to generate the results csv file before generating figures."

    try:
        df = pd.read_csv(path_to_csv)
        stratified_df = stratify_df(df)
        plot_stratified_correlations(stratified_df, top_k, output)
        visualize_entropy_and_cer_across_page_lengths(stratified_df, top_k, output)
        visualize_entropy_vs_surprisal_as_predictor(df, top_k, output, YOUDEN_J)
        print("Execution successful")
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
