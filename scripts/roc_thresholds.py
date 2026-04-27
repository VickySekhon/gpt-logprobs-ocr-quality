"""
Generates a plot of ROC curves for regression results based on a specified top-k value for logprobs and reads data from a CSV file and uses the regression module to create the visualization.
"""

import argparse
from pathlib import Path

from src.regression import main


def _main():
    parser = argparse.ArgumentParser(description=("Generate a plot of ROC"))

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
        main(top_k, output)
        print("Execution successful")
    except Exception as e:
        raise e


if __name__ == "__main__":
    _main()
