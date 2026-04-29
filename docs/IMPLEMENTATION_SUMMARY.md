### What was Implemented

- [What is Complete](#what-is-complete)
- [What is Incomplete](#what-is-incomplete)
- [Code Structure and Coupling](#code-structure-and-coupling)
- [How to Read the Repository](#how-to-read-the-repository)

The entire OCR pipeline was implemented end to end. This includes loading the BLN600 dataset, performing OCR on images using GPT-4o to obtain a transcript and a logprobs object (cached for reuse), normalizing the OCR-generated and ground-truth transcripts to calculate the character error rate (CER) between the two texts, and computing the average token entropy (bits/token). These metrics, along with additional fields (see Figure 1), are stored in a Pandas DataFrame and exported as a CSV file in [results/csv](../results/csv).

Once the CSV file is generated, the metrics are consumed by figure-generation scripts. [entropy_vs_cer.py](../scripts/entropy_vs_cer.py) plots the relationship between entropy and CER, the distribution of entropy, and surprisal versus entropy as an indicator of uncertainty. [stratified_analysis.py](../scripts/stratified_analysis.py) plots entropy versus CER stratified into four quartiles based on ground-truth length and computes correlations stratified into quartiles using Pearson and Spearman coefficients. Finally, [roc_thresholds.py](../scripts/roc_thresholds.py) trains a logistic regression model on entropy values to determine an entropy threshold that separates “good” pages (CER <= 2%) from “bad” pages (CER > 2%) using an ROC curve.

**Figure 1. Per-page record schema stored in the results CSV**
| Field |
|---|
| `page_id` |
| `avg_bits_per_token` |
| `avg_surprisal_per_token` |
| `total_bits` |
| `n_tokens` |
| `cer` |
| `levenshtein` |
| `gt_length` |
| `normalization_profile` |

### What is Complete

- Entire OCR pipeline from start-to-finish
- Robust figure generation
- Centralized makefile execution point 
- Comprehensive documentation outlining figures and their purposes, project structure, and commands for setting up and running the pipeline 

### What is Incomplete

N/A

### Code Structure and Coupling

```
gpt-logprobs-ocr-quality/
     cache/
     data/
     notebooks/
     scripts/
     src/
     makefile
```

`makefile` is the single entry point into the project's code. Three configurable constants are defined (feel free to change these values if needed): TOP_K which determines how many alternative tokens will be returned by GPT-4o, MAX_PAGES which controls the number of excerpts to process, and THREADS which determines how many workers will run the pipeline in parallel. OUTPUT specifies where the results are stored, but should remain as 'result'.

| Constant | Purpose |
|---|---|
| `TOP_K` | Number of alternative tokens returned per token (top-k) |
| `MAX_PAGES` | Number of excerpts to process |
| `THREADS` | Number of parallel workers |
| `OUTPUT` | Output path/prefix for results |

`src` contains the following modules:

| File | Responsibility |
|---|---|
| `src/entropy.py` | Page-level entropy and surprisal calculations |
| `src/loader.py` | Dataset loading utilities (DataFrame plus image/ground-truth retrieval by page ID) |
| `src/logprobs_client.py` | OCR client logic and caching of transcripts and logprobs |
| `src/metrics.py` | Levenshtein distance and CER calculations |
| `src/normalization.py` | Normalization procedures applied to OCR and ground-truth text |
| `src/predict_quality.py` | Pipeline orchestration: OCR, metrics, and CSV export |
| `src/preprocess_dataset.py` | Dataset verification and preprocessing |
| `src/regression.py` | Logistic regression and threshold selection utilities |
| `src/scan2latex_entropy.py` | Scan-to-LaTeX conversion and sliding-window entropy analysis |
| `src/utils.py` | General helper utilities |

`cache` holds a file named 'cache.json' which contains an OCR-generated transcript and logprobs object which is written to by `logprobs_client.py`.

`data` holds the BLN600 dataset which is preprocessed by `preprocess_dataset.py` and loaded by `loader.py`.

`notebooks` holds miscellaneous python files used for thought experiments and is not used anywhere outside of this folder.

`scripts` contains scripts for figure generation that directly use functions from files within `src/`.

### How to Read the Repository

Developers should begin with the `makefile` to trace the execution of the pipeline. `predict_quality.py` should be the focal point when trying to understand the pipeline logic, since it is the top-level entry point and calls other modules for specific functionality. In addition, each `.py` file begins with a short synopsis describing its purpose; it is intended as a brief overview rather than a full description of implementation details.

To follow the figure-generation process, developers should read the `scripts/` directory. For details about each figure, `docs/figures/` contains documentation describing what the figure represents, the units used, and the input data.

Additional README files are included in `cache/` to describe the structure of the cache and what is cached versus omitted, in `results/` to describe each generated figure and how it was created, and in `data/` to describe how to obtain the dataset.

For project setup and running, refer to the README file at the root of the project.