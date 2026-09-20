# Stage B results

Timing is serial batch=1 end-to-end, not pure model latency.

| Run | Candidates (0=revision) | Accuracy | Calls/example | Seconds |
|---|---:|---:|---:|---:|
| ar_seed81 | 1 | 0.0000 | 6 | 0.063 |
| ar_seed81 | 2 | 0.0000 | 12 | 0.135 |
| ar_seed81 | 4 | 0.0000 | 24 | 0.303 |
| ar_more_training_seed81 | 1 | 0.0000 | 6 | 0.065 |
| ar_more_training_seed81 | 2 | 0.0000 | 12 | 0.141 |
| ar_more_training_seed81 | 4 | 0.0000 | 24 | 0.285 |
| hard_seed81 | 0 | 0.0000 | 12 | 0.128 |
| soft_seed81 | 0 | 0.0000 | 12 | 0.135 |
| posthoc_hard_seed81 | 0 | 0.0625 | 12 | 0.138 |
| posthoc_soft_seed81 | 0 | 0.0625 | 12 | 0.135 |
