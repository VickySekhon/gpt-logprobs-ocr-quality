## BLN600: GPT Log-Probabilities as Predictors of OCR Quality

Explain clearly what the project does.

### About

Modern OCR systems sometimes return poor text, especially for older print. When a language model reads a page, it not only predicts tokens but also provides probabilities for alternatives. By converting those probabilities into a single entropy number per token (higher = more uncertainty) and averaging across the page, we get a simple score. We will test whether that score can flag pages likely to have high OCR error, so teams can accept clean pages automatically and review risky ones.

This project offers a full pipeline to measure the quality of OCR performed by GPT-4o on 19th century English newspapers that works as follows:

1. Loads images for OCR along with corresponding ground truth texts from the dataset containing 19th century English newspapers.
2. Prompts GPT-4o to perform OCR on the image and return the text along with a logprobs object that represents the distribution of probabilities for each token.
3. Calculates the model's uncertainty (entropy) per token from the logprobs object, and averages the uncertainty to arrive at a page-level metric.
4. Normalizes both the ground truth and the OCR-generated texts then computes a character error rate (CER) statistic that measures their differences using the levenshtein distance formula.
5. Writes `page_id`, `avg_bits_per_token`, `avg_surprisal_per_token`, `total_bits`,`n_tokens`, `cer`, `levenshtein`, `gt_length`, and `normalization_profile` to a Pandas Dataframe for easy visualization.
6. Creates visualizations to help better identify the relationship between the OCR quality (measured by the CER) and model uncertainty (measured by average bits of entropy per token) by plotting:
   - The relationship between entropy and CER.
   - The distribution of entropy across page excerpts.
   - Average surprisal per token versus average entropy per token as an indicator of model uncertainty.
   - A scatter plot showcasing the effect of the length of page excerpts on the relationship between entropy and CER.
   - The computed correlations between entropy and CER across different page lengths using two correlation coefficients, namely pearson and spearman.
   - An ROC curve that determines the entropy threshold that maximizes specificity and sensitivity for a logistic regression model that recognizes good pages that have a CER <= 2%.
   - A table comparing the performance of the Youden J and Min Error statistics to make the a guided decision about what statistic to follow when determining the binary classification threshold.

State the dataset used.

### Dataset

The [BLN600 dataset](https://aclanthology.org/2024.lrec-main.219.pdf) was used to perform OCR for the purposes of this research project. The BLN600 dataset is a corpus of nineteenth-century newspaper text focused on crime in London, derived from the Gale British Library Newspapers corpus parts 1 and 2. The corpus comprises 600 newspaper excerpts and for each excerpt contains the original source image, the machine transcription of that image as found in the BLN and a gold standard manual transcription that we have created. The corpus was intended for the training and development of OCR and post-OCR correction methodologies for historical newspaper machine transcription—for which there is currently a dearth of publicly available resources.

Give exact setup steps.

### Setup

To set this project up 
Show exactly how to run the code from a clean machine.
State where the final results, tables, and figures will appear.
Include a short Figure Index listing each important figure, the script/notebook that generated it, and the matching documentation file in docs/figures/
