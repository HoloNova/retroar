# 单次延迟证据实验：自动汇总

仅描述性结果，正式结论见后续综合分析。

| 方法 | 整题正确率 | 种子间标准差 |
|---|---:|---:|
| wait_ar | 18.75% | 0.00 pp |
| wait_ar/frozen_early | 0.00% | 0.00 pp |
| wait_ar/permutation_repair | 0.00% | 0.00 pp |
| wait_ar/no_reveal | 0.00% | 0.00 pp |
| wait_ar/wait_k2 | 25.00% | 0.00 pp |
| restart_hard | 31.25% | 0.00 pp |
| restart_hard/frozen_early | 0.00% | 0.00 pp |
| restart_hard/permutation_repair | 0.00% | 0.00 pp |
| restart_hard/no_reveal | 0.00% | 0.00 pp |
| revise_hard | 18.75% | 0.00 pp |
| revise_hard/frozen_early | 0.00% | 0.00 pp |
| revise_hard/permutation_repair | 0.00% | 0.00 pp |
| revise_hard/no_reveal | 0.00% | 0.00 pp |
| revise_soft | 25.00% | 0.00 pp |
| revise_soft/frozen_early | 0.00% | 0.00 pp |
| revise_soft/permutation_repair | 0.00% | 0.00 pp |
| revise_soft/no_reveal | 0.00% | 0.00 pp |

## 配对差值

{
  "revise_hard - wait_ar": {
    "mean_pp": 0.0,
    "positive_seeds": 0,
    "by_seed_pp": [
      0.0
    ]
  },
  "revise_hard - restart_hard": {
    "mean_pp": -12.5,
    "positive_seeds": 0,
    "by_seed_pp": [
      -12.5
    ]
  },
  "revise_hard - wait_ar/wait_k2": {
    "mean_pp": -6.25,
    "positive_seeds": 0,
    "by_seed_pp": [
      -6.25
    ]
  },
  "revise_soft - revise_hard": {
    "mean_pp": 6.25,
    "positive_seeds": 1,
    "by_seed_pp": [
      6.25
    ]
  },
  "revise_hard - revise_hard/permutation_repair": {
    "mean_pp": 18.75,
    "positive_seeds": 1,
    "by_seed_pp": [
      18.75
    ]
  }
}