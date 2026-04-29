## BLN600: GPT Log-Probabilities as Predictors of OCR Quality

### About

Modern OCR systems sometimes return poor text, especially for older print. When a language model reads a page, it not only predicts tokens but also provides probabilities for alternatives. By converting those probabilities into a single entropy value per token (higher values indicate more uncertainty) and averaging across the page, it is possible to compute a simple uncertainty score. This project tests whether that score can flag pages that are likely to have high OCR error, so teams can accept clean pages automatically and review risky pages.

This project provides a complete pipeline to measure the quality of OCR performed by GPT-4o on nineteenth-century English newspapers. The pipeline works as follows:

1. Loads images for OCR along with corresponding ground truth texts from the dataset containing 19th century English newspapers.
2. Prompts GPT-4o to perform OCR on the image and return the text along with a logprobs object that represents the distribution of probabilities for each token.
3. Calculates the model's uncertainty (entropy) per token from the logprobs object, and averages the uncertainty to arrive at a page-level metric.
4. Normalizes both the ground truth and the OCR-generated texts and then computes a character error rate (CER) statistic that measures their differences using the Levenshtein distance.
5. Writes `page_id`, `avg_bits_per_token`, `avg_surprisal_per_token`, `total_bits`, `n_tokens`, `cer`, `levenshtein`, `gt_length`, and `normalization_profile` to a Pandas DataFrame for visualization.
6. Creates visualizations to help identify the relationship between OCR quality (measured by CER) and model uncertainty (measured by average token entropy in bits/token) by plotting:
   - The relationship between entropy and CER.
   - The distribution of entropy across page excerpts.
   - Average surprisal per token versus average entropy per token as an indicator of model uncertainty.
   - A scatter plot showcasing the effect of the length of page excerpts on the relationship between entropy and CER.
   - The computed correlations between entropy and CER across different page lengths using two correlation coefficients (Pearson and Spearman).
   - An ROC curve that determines the entropy threshold that maximizes specificity and sensitivity for a logistic regression model that classifies pages as good if they have a CER <= 2%.
   - A table comparing the performance of the Youden J and Min Error statistics to make a guided decision about which statistic to use when selecting a binary classification threshold.

### Dataset

The [BLN600 dataset](https://aclanthology.org/2024.lrec-main.219.pdf) was used to perform OCR for the purposes of this research project. The BLN600 dataset is a corpus of nineteenth-century newspaper text focused on crime in London, derived from the Gale British Library Newspapers corpus parts 1 and 2. The corpus comprises 600 newspaper excerpts and for each excerpt contains the original source image, the machine transcription of that image as found in the BLN and a gold standard manual transcription that we have created. The corpus was intended for the training and development of OCR and post-OCR correction methodologies for historical newspaper machine transcription—for which there is currently a dearth of publicly available resources.

### Setup

This project uses a `makefile` as the main entry point. It provides commands to:

1) Install dependencies from `requirements.txt`
2) Run the entire pipeline to generate a single CSV file
3) Generate figures and tables from the CSV file
4) Clean figures and tables to make room for new outputs

#### Windows

Windows does not natively include `make`. Install it explicitly.
Chocolatey is a package manager that can be used to install `make`. Follow the Chocolatey installation steps and then run `choco install make`. Restart your terminal session afterward.

Before beginning, create a virtual environment named `venv` using the command:

`python -m venv venv`

Then run the following command to install the required dependencies: 

`make install`

### Running the Project

After following the instructions in **Setup**, download the dataset [here](https://orda.shef.ac.uk/articles/dataset/BLN600_A_Parallel_Corpus_of_Machine_Human_Transcribed_Nineteenth_Century_Newspaper_Texts/25439023). Follow the instructions in `data/README.md` to ensure the dataset is downloaded to the correct location and named correctly. Once the dataset is downloaded, run the following command to preprocess it for OCR analysis:

`make preprocess`

**Important**: the command above will fail if the dataset has not been downloaded correctly. Use the error message to determine which step may have been missed.

Use the following command to run the pipeline to generate an all-encompassing CSV file:

`make run-all`

To generate figures and tables from your results CSV file run the following command:

`make figs`

**Note**: the command above will not work unless you run `make run-all` beforehand.

**Important**: if figures and tables have already been generated and need to be regenerated, run `make clean`. This removes previously generated figures and tables.

### Results

All generated results are located inside `/results`. The directory structure is as follows:

```
project/
   results/
      csv/
      figures/
      tables/
```

**csv/**: contains the output from `make run-all`

**figures/**: contains the output from `make figs`

**tables/**: contains the output from `make figs`

### Figure Index

The following are each of the figures generated by this project along with the paths to the scripts that generated each and documentation for the figure:

| Figure (inside `results/figures/`) | Script (inside `scripts/`) | Documentation (inside `docs/figures/`) |
| -------- | -------- | -------- |
| figure_01_entropy_vs_cer.png | entropy_vs_cer.py | figure_01.md |
| figure_02_entropy_distribution.png | entropy_vs_cer.py | figure_02.md |
| figure_03_surprisal_vs_entropy.png | stratified_analysis.py | figure_03.md |
| figure_04_entropy_vs_cer_stratified.png | stratified_analysis.py | figure_04.md |
| figure_05_stratified_correlations.png | stratified_analysis.py | figure_05.md |
| figure_06_roc_entropy.png | roc_thresholds.py | figure_06.md |

