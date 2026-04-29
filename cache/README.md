### What is Cached

The OCR output from GPT-4o is cached in `cache/cache.json` along with an array of logprobs objects consisting of the generated token and `k-1` alternatives, along with their log probabilities for a given page:

```
{
     page-id_model_top-k_prompt-version: {
          transcript: "",
          token_logprobs: [
               {
                    token: "",
                    logprobs: [
                         ...logprobs for selected token and top-k alternatives
                    ]
                    alts: [
                         {
                              token: "",
                              logprob: 0
                         },
                         ...next k-1 alternatives
                    ]
               },
               ...next token 
          ]
     },
     ...next page
}
```

Each page has a unique key consisting of:

- page-id
- model used (`GPT-4o` in research experiments)
- top-k (`10` used in research experiments)
- prompt-version (`v1` used in research experiments)

### What is Omitted

No metrics are cached as they can be directly computed from the OCR-generated transcript and logprob objects.


### How to Regenerate

Once the pipeline is run once (without error), the cache will be fully populated and subsequent runs of the pipeline will result in cache-hits therefore preventing any novel results from being generated. To compute new results obtained by querying GPT-4o, the `cache/cache.json` file can be deleted.

**Important**: refrain from clearing the cache manually, it is designed to be deleted rather than emptied. 