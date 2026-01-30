## Requirements

## Steps

1. historical newspaper page image -> GPT-4o transcript + token-level top-k logprobs
2. find shannon entropy (top-k+tail) per token
3. compute the average entropy per page
4. compute CER (levenshtein distance from GPT-4o transcript to GT)
5. use entropy score as a triage to determine whether to accept or reject a GPT-4o-generated transcript

## Constraints

1. one GPT-4o decoding run per page
2. use native logprobs only
3. 'k' defaults to 5

# Code Structure

1. private github repo
2. README, LICENSE, src/, notebooks/, data/, cache/, figures/
3. virtual environment
     - pin dependencies:
          - pandas
          - numpy
          - rapidfuzz[levenshtein]
          - matplotlib
          - seaborn (optional)
          - scikit-learn
          - python-dotenv (or similar)
          - openai

