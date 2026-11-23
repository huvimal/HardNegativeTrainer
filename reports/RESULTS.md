# HardNegativeTrainer — measured local run

Train shared 32-dimensional bi-encoder projection heads on frozen pretrained MiniLM features. Transformer backbone is not fine-tuned. Tiny authored fixture, four evaluation topics.

## Results

```json
{
  "pretrained_test_hit1": 1.0,
  "projection_test_hit1": 0.25,
  "train_hit1": 1.0,
  "hard_negative_pairs": [
    {
      "query": 0,
      "hard_negative": 3
    },
    {
      "query": 1,
      "hard_negative": 7
    },
    {
      "query": 2,
      "hard_negative": 5
    },
    {
      "query": 3,
      "hard_negative": 0
    },
    {
      "query": 4,
      "hard_negative": 7
    },
    {
      "query": 5,
      "hard_negative": 3
    },
    {
      "query": 6,
      "hard_negative": 3
    },
    {
      "query": 7,
      "hard_negative": 3
    }
  ],
  "checks": [
    "Training query IDs do not include evaluation IDs",
    "Actual contrastive objective optimized"
  ],
  "scope": "Train shared 32-dimensional bi-encoder projection heads on frozen pretrained MiniLM features. Transformer backbone is not fine-tuned. Tiny authored fixture, four evaluation topics."
}
```

Local experiment assertions passed. This does not establish production readiness or performance on real customer data. Source fingerprints and dependency versions are in results.json.
