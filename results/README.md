## Generated Results

| Output | Type | Description |
|---|---|---|
| [results_k_10.csv](csv/results_k_10.csv) | CSV | Per-page metrics (`page_id`, entropy/surprisal summaries, CER, Levenshtein distance, ground-truth length, normalization profile) exported from a Pandas DataFrame |
| [figure_01_entropy_vs_cer.png](figures/figure_01_entropy_vs_cer.png) | Figure | Entropy (bits/token) versus CER scatter plot with best-fit trend line |
| [figure_02_entropy_distribution.png](figures/figure_02_entropy_distribution.png) | Figure | Histogram of entropy (bits/token) |
| [figure_03_surprisal_vs_entropy.png](figures/figure_03_surprisal_vs_entropy.png) | Figure | Surprisal (bits/token) versus entropy (bits/token) scatter plot |
| [figure_04_entropy_vs_cer_stratified.png](figures/figure_04_entropy_vs_cer_stratified.png) | Figure | Entropy versus CER stratified by ground-truth length quartiles |
| [figure_05_stratified_correlations.png](figures/figure_05_stratified_correlations.png) | Figure | Pearson and Spearman correlations by quartile |
| [figure_06_roc_entropy.png](figures/figure_06_roc_entropy.png) | Figure | ROC curve for a 2% CER threshold (validation split) |
| [table_01_roc_table.png](tables/table_01_roc_table.png) | Table | Threshold comparison table (sensitivity/specificity trade-offs) |