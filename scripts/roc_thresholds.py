"""
Generates a plot of ROC curves for regression results based on a specified top-k value for logprobs and reads data from a CSV file and uses the regression module to create the visualization.
"""

import argparse
import pandas as pd
from pathlib import Path

from src.regression import main

from src.utils import TOP_K, OUTPUT_DIRECTORY


def _main():
    parser = argparse.ArgumentParser(description=("Generate a plot of ROC"))

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
        main(df, output)
        print("Execution successful")
    except Exception as e:
        raise e


if __name__ == "__main__":
    _main()
