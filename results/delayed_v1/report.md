# 单次延迟证据实验：自动汇总

仅描述性结果，正式结论见后续综合分析。

| 方法 | 整题正确率 | 种子间标准差 |
|---|---:|---:|
| wait_ar | 74.45% | 3.12 pp |
| wait_ar/frozen_early | 0.23% | 0.35 pp |
| wait_ar/permutation_repair | 2.89% | 2.30 pp |
| wait_ar/no_reveal | 2.73% | 0.55 pp |
| wait_ar/wait_k2 | 79.53% | 2.81 pp |
| restart_hard | 66.64% | 6.33 pp |
| restart_hard/frozen_early | 0.31% | 0.33 pp |
| restart_hard/permutation_repair | 3.83% | 4.08 pp |
| restart_hard/no_reveal | 0.94% | 0.65 pp |
| revise_hard | 80.16% | 2.18 pp |
| revise_hard/frozen_early | 0.70% | 0.85 pp |
| revise_hard/permutation_repair | 3.83% | 2.15 pp |
| revise_hard/no_reveal | 2.50% | 1.62 pp |
| revise_soft | 90.23% | 1.10 pp |
| revise_soft/frozen_early | 0.08% | 0.17 pp |
| revise_soft/permutation_repair | 2.42% | 1.67 pp |
| revise_soft/no_reveal | 4.84% | 2.10 pp |

## 配对差值

{
  "revise_hard - wait_ar": {
    "mean_pp": 5.703125,
    "positive_seeds": 5,
    "by_seed_pp": [
      5.078125,
      2.734375,
      8.203125,
      10.546875,
      1.953125
    ]
  },
  "revise_hard - restart_hard": {
    "mean_pp": 13.515625,
    "positive_seeds": 5,
    "by_seed_pp": [
      12.5,
      9.375,
      16.796875,
      6.640625,
      22.265625
    ]
  },
  "revise_hard - wait_ar/wait_k2": {
    "mean_pp": 0.625,
    "positive_seeds": 2,
    "by_seed_pp": [
      -2.34375,
      -0.78125,
      3.515625,
      5.078125,
      -2.34375
    ]
  },
  "revise_soft - revise_hard": {
    "mean_pp": 10.078125,
    "positive_seeds": 5,
    "by_seed_pp": [
      10.546875,
      10.15625,
      5.859375,
      10.15625,
      13.671875
    ]
  },
  "revise_hard - revise_hard/permutation_repair": {
    "mean_pp": 76.328125,
    "positive_seeds": 5,
    "by_seed_pp": [
      73.828125,
      74.609375,
      80.859375,
      79.296875,
      73.046875
    ]
  }
}