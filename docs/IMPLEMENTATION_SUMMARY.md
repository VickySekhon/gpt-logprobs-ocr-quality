## Contents
- [What was Implemented](#what-was-implemented)
- [What is Complete](#what-is-complete)
- [What is Incomplete](#what-is-incomplete)
- [Code Structure and Coupling](#code-structure-and-coupling)
- [How to Read the Repository](#how-to-read-the-repository)

## What was Implemented

The OCR pipeline was implemented end to end. It loads the BLN600 dataset, performs OCR on images using GPT-4o to obtain a transcript plus a logprobs object (cached for reuse), normalizes the OCR-generated and ground-truth transcripts to compute character error rate (CER), and computes average token entropy (bits/token). These metrics, along with additional fields (see Figure 1), are stored in a Pandas DataFrame and exported to [results/csv](../results/csv).

Once the CSV is generated, figure scripts consume it. [entropy_vs_cer.py](../scripts/entropy_vs_cer.py) plots entropy versus CER, the entropy distribution, and surprisal versus entropy as an additional uncertainty indicator. [stratified_analysis.py](../scripts/stratified_analysis.py) stratifies pages into four quartiles by ground-truth length, plots entropy versus CER by quartile, and computes Pearson and Spearman correlations per quartile. Finally, [roc_thresholds.py](../scripts/roc_thresholds.py) trains a logistic regression model on entropy values and uses an ROC curve to choose an entropy threshold that separates “good” pages (CER <= 2%) from “bad” pages (CER > 2%).

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

## What is Complete

- Entire OCR pipeline from start-to-finish
- Robust figure generation
- Centralized `makefile` entry point
- Comprehensive documentation describing figures, project structure, and setup/run commands

## What is Incomplete

N/A

## Code Structure and Coupling

```
gpt-logprobs-ocr-quality/
     cache/
     data/
     notebooks/
     scripts/
     src/
     makefile
```

`makefile` is the single entry point into the project. It defines four configuration constants: `TOP_K` (how many alternative tokens GPT-4o returns per token), `MAX_PAGES` (how many excerpts to process), `THREADS` (parallel workers), and `OUTPUT` (output directory/prefix for generated artifacts).

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

`cache` contains `cache.json`, which stores OCR transcripts and logprobs objects written by `logprobs_client.py`.

`data` contains the BLN600 dataset. It is verified and preprocessed by `preprocess_dataset.py` and loaded by `loader.py`.

`notebooks` contains exploratory notebooks and is not used by the pipeline.

`scripts` contains figure-generation scripts that call into `src/`.

## How to Read the Repository

Developers should begin with the `makefile` to trace execution. `predict_quality.py` is the best starting point for pipeline logic because it is the top-level entry point and delegates to other modules. Each `.py` file also includes a short synopsis that is intended as an overview, not a full implementation description.

To follow figure generation, read `scripts/`. For figure-specific details (meaning, units, and inputs), see `docs/figures/`.

Additional README files are provided in `cache/` (cache structure and omissions), `results/` (generated outputs), and `data/` (dataset download and placement).

For project setup and running, refer to the README file at the root of the project.