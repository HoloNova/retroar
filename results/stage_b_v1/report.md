# Stage B results

Timing is serial batch=1 end-to-end, not pure model latency.

| Run | Candidates (0=revision) | Accuracy | Calls/example | Seconds |
|---|---:|---:|---:|---:|
| ar_seed10 | 1 | 0.8320 | 6 | 1.159 |
| ar_seed10 | 2 | 0.8555 | 12 | 2.274 |
| ar_seed10 | 4 | 0.8867 | 24 | 4.758 |
| ar_more_training_seed10 | 1 | 0.8750 | 6 | 1.028 |
| ar_more_training_seed10 | 2 | 0.8789 | 12 | 2.325 |
| ar_more_training_seed10 | 4 | 0.8906 | 24 | 4.847 |
| hard_seed10 | 0 | 0.8867 | 12 | 2.130 |
| soft_seed10 | 0 | 0.9062 | 12 | 1.876 |
| posthoc_hard_seed10 | 0 | 0.9023 | 12 | 2.059 |
| posthoc_soft_seed10 | 0 | 0.8906 | 12 | 2.066 |
| ar_seed11 | 1 | 0.7969 | 6 | 1.111 |
| ar_seed11 | 2 | 0.8242 | 12 | 2.276 |
| ar_seed11 | 4 | 0.8594 | 24 | 4.469 |
| ar_more_training_seed11 | 1 | 0.7969 | 6 | 1.034 |
| ar_more_training_seed11 | 2 | 0.8242 | 12 | 2.242 |
| ar_more_training_seed11 | 4 | 0.8594 | 24 | 4.920 |
| hard_seed11 | 0 | 0.8594 | 12 | 2.225 |
| soft_seed11 | 0 | 0.8398 | 12 | 1.904 |
| posthoc_hard_seed11 | 0 | 0.9180 | 12 | 2.173 |
| posthoc_soft_seed11 | 0 | 0.9609 | 12 | 2.046 |
| ar_seed12 | 1 | 0.7617 | 6 | 1.081 |
| ar_seed12 | 2 | 0.8047 | 12 | 2.230 |
| ar_seed12 | 4 | 0.8320 | 24 | 4.697 |
| ar_more_training_seed12 | 1 | 0.7109 | 6 | 1.091 |
| ar_more_training_seed12 | 2 | 0.7500 | 12 | 2.231 |
| ar_more_training_seed12 | 4 | 0.7773 | 24 | 4.722 |
| hard_seed12 | 0 | 0.8359 | 12 | 2.109 |
| soft_seed12 | 0 | 0.8594 | 12 | 2.013 |
| posthoc_hard_seed12 | 0 | 0.9102 | 12 | 2.659 |
| posthoc_soft_seed12 | 0 | 0.9375 | 12 | 1.872 |
| ar_seed13 | 1 | 0.7383 | 6 | 1.063 |
| ar_seed13 | 2 | 0.7891 | 12 | 2.237 |
| ar_seed13 | 4 | 0.8711 | 24 | 4.435 |
| ar_more_training_seed13 | 1 | 0.7500 | 6 | 1.052 |
| ar_more_training_seed13 | 2 | 0.7656 | 12 | 2.320 |
| ar_more_training_seed13 | 4 | 0.7969 | 24 | 4.562 |
| hard_seed13 | 0 | 0.8203 | 12 | 2.291 |
| soft_seed13 | 0 | 0.8164 | 12 | 1.930 |
| posthoc_hard_seed13 | 0 | 0.9219 | 12 | 2.151 |
| posthoc_soft_seed13 | 0 | 0.9492 | 12 | 2.107 |
| ar_seed14 | 1 | 0.7852 | 6 | 1.180 |
| ar_seed14 | 2 | 0.8281 | 12 | 2.411 |
| ar_seed14 | 4 | 0.8633 | 24 | 4.523 |
| ar_more_training_seed14 | 1 | 0.8242 | 6 | 1.029 |
| ar_more_training_seed14 | 2 | 0.8359 | 12 | 2.289 |
| ar_more_training_seed14 | 4 | 0.8477 | 24 | 4.978 |
| hard_seed14 | 0 | 0.8320 | 12 | 2.041 |
| soft_seed14 | 0 | 0.8398 | 12 | 1.915 |
| posthoc_hard_seed14 | 0 | 0.8906 | 12 | 1.995 |
| posthoc_soft_seed14 | 0 | 0.9297 | 12 | 2.105 |


## 分析

完整分析（配对检验、难度分层、每题结果转移、机制解释与限制）见项目根目录 `阶段B_分析.md`。

要点：同预算（12 次前向/题）下 posthoc_soft 93.36% > posthoc_hard 90.86% > soft 85.23% > hard 84.69% > ar(k=2) 82.03%；双倍训练无收益；soft 相对 hard 的优势在配对监督后未达显著（在线 +0.55 pp、post-hoc +2.50 pp），原因是训练后分布已塌缩为近似 one-hot。
