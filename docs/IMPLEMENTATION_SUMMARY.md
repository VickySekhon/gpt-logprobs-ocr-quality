### What was Implemented

The entire OCR pipeline was successfully implemented, including the loading the BLN600 dataset, performing OCR on the images using GPT-4o to obtain a transcript, and a logprobs object that is cached for reuse, normalizing the OCR-generated and ground-truth transcripts to calculate the Character Error Rate (CER) between the two texts, calculating the Average Token Entropy (Bits/Token), and storing these metrics along with others (see figure 1.) inside a Pandas DataFrame exported as a csv file inside [results/csv](../results/csv).

Once the CSV file is generated, the metrics are consumed by [entropy_vs_cer.py](../scripts/entropy_vs_cer.py) to plot the relationship between entropy and CER, the distribution of entropy, and the use of surprisal vs entropy as an indicator of uncertainty. And [stratified_analysis.py](../scripts/stratified_analysis.py) to plot the relationship between entropy and cer stratified into four quartiles based on ground-truth length, and to plot the correlations between the entropy and cer stratified into four quartiles using pearson and spearman coefficients. And finally, [roc_thresholds.py](../scripts/roc_thresholds.py) to train a logistic regression model on the entropy levels in the data to determine what threshold of entropy constitutes a good page (which has a CER <= 2%) versus a bad page (which has a CER > 2%) with the ROC curve. 

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

`src` contains the following files listed alongside an overview of what they accomplish:
```
src/ 
     entropy.py # Contains the entropy and surprisal calculation logic at the page-level 
    
     loader.py # Contains functions that load the dataset into a DataFrame as well as an image and ground-truth pair from a page-id

     logprobs_client.py # Contains OCR logic that takes in the path to an image, prompts GPT-4o to perform OCR on the image, and saves the generated transcript and logprobs object to a local JSON cache

     metrics.py # Contains the levenshtein distance and CER calculation logic
     
     normalization.py # Contains interactive and fixed functions that remove whitespaces, quotes/dashes, and punctuation from a both OCR-generated and ground truth texts

     predict_quality.py # Contains logic that runs the entire pipeline from start-to-finish including loading the dataset, performing OCR, calculating metrics, and exporting the results to a CSV file

     preprocess_dataset.py # Contains logic that verifies and prepares the dataset for processing by predict_quality.py

     regression.py # Contains logic to train a logistic regression model to learn the relationship between entropy and a good page with low CER and find the binary classification threshold that maximizes sensitivity and specificity

     scan2latex_entropy.py # Contains logic to load a page excerpt, convert it into latex, and then runs sliding window analysis to determine areas with most entropy

     utils.py # Contains helper functions used just about everywhere in the files above
``` 

`cache` holds a file named 'cache.json' which contains an OCR-generated transcript and logprobs object which is written to by `logprobs_client.py`.

`data` holds the BLN600 dataset which is preprocessed by `preprocess_dataset.py` and loaded by `loader.py`.

`notebooks` holds miscellaneous python files used for thought experiments and is not used anywhere outside of this folder.

`scripts` contains scripts for figure generation that directly use functions from files within `src/`.

### How to Read the Repository

Developers should begin at the makefile to trace the execution of the pipeline. `predict_quality.py` should be the focal point when trying to understand the pipeline's logic, it is at the top-level of the pipeline and calls other scripts for specific functionality. Furthermore, at the top of every `.py` file there is a short synopsis that will help Developers understand what the file accomplishes, think of it as metaphorically similar to the summary of a book, as it is simply a top-level overview that leaves out specific details.

To follow the process of figure generation, Developers should read the `scripts/` directory. For details about each figure, the `docs/figures` contains specific information that elaborates on what the figure represents, the units used, the input data, etc.

README files are purposely created within `cache/` to describe the structure of the cache and what data is cached versus omitted, `results/` to describe each generated figure and how it was created, `data/` to describe the entire process of how to obtain the dataset.

For project setup and running, refer to the README file at the root of the project.