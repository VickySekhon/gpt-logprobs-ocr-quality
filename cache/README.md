### What is Cached

The OCR output from GPT-4o is cached in `cache/cache.json` along with an array of logprobs objects. Each logprobs object contains the generated token and `k-1` alternatives, along with their log probabilities for a given page:

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

Once the pipeline is run once (without error), the cache will be fully populated. Subsequent pipeline runs will result in cache hits, thereby preventing new results from being generated. To compute new results obtained by querying GPT-4o, delete `cache/cache.json`.

**Important**: do not clear the cache manually. The cache is designed to be deleted rather than emptied.