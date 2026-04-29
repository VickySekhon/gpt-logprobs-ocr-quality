# BLN600: GPT Log-Probabilities as Predictors of OCR Quality

## Contents

- [About](#about)
- [Dataset](#dataset)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [Results](#results)
- [Figure Index](#figure-index)

## About

Modern OCR systems sometimes return poor text, especially for older print. When a language model reads a page, it not only predicts tokens but also provides probabilities for alternatives. By converting those probabilities into a single entropy value per token (higher values indicate more uncertainty) and averaging across the page, it is possible to compute a simple uncertainty score. This project tests whether that score can flag pages that are likely to have high OCR error, so teams can accept clean pages automatically and review risky pages.

This project provides a pipeline to measure the quality of OCR performed by GPT-4o on nineteenth-century English newspapers. The pipeline performs the following steps:

1. Loads images for OCR and their corresponding ground-truth texts.
2. Prompts GPT-4o to perform OCR and return the text alongside a logprobs object (representing the probability distribution for each token).
3. Calculates the model's token-level uncertainty (entropy) from the logprobs object and averages it to produce a page-level metric.
4. Normalizes the ground-truth and OCR-generated texts, then computes the character error rate (CER) using Levenshtein distance.
5. Writes the resulting metrics (`page_id`, `avg_bits_per_token`, `cer`, `levenshtein`, etc.) to a Pandas DataFrame.
6. Generates visualizations to explore the relationship between OCR quality (CER) and model uncertainty (average token entropy), including:
   - The overall relationship between entropy and CER.
   - The distribution of entropy across page excerpts.
   - Average surprisal versus average entropy as uncertainty indicators.
   - The effect of page length on the entropy-CER relationship.
   - Pearson and Spearman correlations across different page lengths.
   - An ROC curve determining the optimal entropy threshold for classifying pages as "good" (CER <= 2%).
   - A comparison of Youden J and Min Error statistics for selecting a binary classification threshold.

## Dataset

This project uses the [BLN600 dataset](https://aclanthology.org/2024.lrec-main.219.pdf) for OCR evaluation. Below is its description from the BLN600 whitepaper:

"The BLN600 dataset is a corpus of nineteenth-century newspaper text focused on crime in London, derived from the Gale British Library Newspapers corpus parts 1 and 2. The corpus comprises 600 newspaper excerpts and for each excerpt contains the original source image, the machine transcription of that image as found in the BLN and a gold standard manual transcription that we have created. The corpus was intended for the training and development of OCR and post-OCR correction methodologies for historical newspaper machine transcription—for which there is currently a dearth of publicly available resources."

## Setup

This project uses a `makefile` as the main entry point. It provides commands to:

1) Install dependencies from `requirements.txt`
2) Run the entire pipeline to generate a single CSV file
3) Generate figures and tables from the CSV file
4) Clean figures and tables to make room for new outputs

### Windows

Windows does not include `make` by default. Install it first.
One option is Chocolatey, a package manager for Windows. Install Chocolatey, then run `choco install make`. Restart your terminal session afterward.

Before beginning, create a virtual environment named `venv` using the command:

```bash
python -m venv venv
```

Then run the following command to install the required dependencies: 

```bash
make install
```

## Running the Project

First, download the dataset [here](https://orda.shef.ac.uk/articles/dataset/BLN600_A_Parallel_Corpus_of_Machine_Human_Transcribed_Nineteenth_Century_Newspaper_Texts/25439023). Read `data/README.md` for guidance on placing and naming it correctly within the project directory. 

After placing the dataset, preprocess it for OCR analysis:

```bash
make preprocess
```

**Note**: This command fails if the dataset is missing or misnamed. Read the error message to troubleshoot.

Next, run the full pipeline to parse OCR results and compute metrics in a unified CSV file:

```bash
make run-all
```

Finally, generate the figures and tables from the output CSV:

```bash
make figs
```

**Important**: To regenerate charts, run `make clean` first to clear previously produced outputs.

## Results

All generated results are located inside `/results`. The directory structure is as follows:

```
project/
   results/
      csv/
      figures/
      tables/
```

| Folder | Produced by | Description |
|---|---|---|
| `results/csv/` | `make run-all` | Per-page metrics exported as CSV |
| `results/figures/` | `make figs` | Generated figures (PNG/SVG) |
| `results/tables/` | `make figs` | Generated tables (PNG) |

## Figure Index

The following are each of the figures generated by this project along with the paths to the scripts that generated each and documentation for the figure:

| Figure (inside `results/figures/`) | Script (inside `scripts/`) | Documentation (inside `docs/figures/`) |
| -------- | -------- | -------- |
| figure_01_entropy_vs_cer.png | entropy_vs_cer.py | figure_01.md |
| figure_02_entropy_distribution.png | entropy_vs_cer.py | figure_02.md |
| figure_03_surprisal_vs_entropy.png | stratified_analysis.py | figure_03.md |
| figure_04_entropy_vs_cer_stratified.png | stratified_analysis.py | figure_04.md |
| figure_05_stratified_correlations.png | stratified_analysis.py | figure_05.md |
| figure_06_roc_entropy.png | roc_thresholds.py | figure_06.md |

## References

- Booth, Callum; Thomas, Alan; Gaizauskas, Robert (2024). BLN600: A Parallel Corpus of Machine/Human Transcribed Nineteenth Century Newspaper Texts. The University of Sheffield. Dataset. https://doi.org/10.15131/shef.data.25439023.v2