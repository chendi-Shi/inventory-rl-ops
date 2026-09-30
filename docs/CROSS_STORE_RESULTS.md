# Version 2: cross-store results and baseline fallback

The [plan](CROSS_STORE_PLAN.md) was committed before CA_2 and CA_3 test results were inspected. CA_1 had already been inspected and is included only for context. Each store trained its own 64-SKU policy with the same code, economics, chronological split, three seeds and 60 episodes. The full [CA_2](m5-ca2-report.json) and [CA_3](m5-ca3-report.json) reports include selected items, source hash, validation choices and all 113 paired test-day profits.

| Store | RL simulated profit | Tuned rule | RL difference | 95% paired block interval | RL fill / rule fill | Active policy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CA_1, earlier inspected | 266,565.7 | 270,500.7 | −1.45% | [−8,691.3, 1,415.6] | 61.34% / 61.44% | Base-stock |
| CA_2, predeclared | 273,633.5 | 288,988.0 | −5.31% | [−23,611.6, −8,265.5] | 69.32% / 71.77% | Base-stock |
| CA_3, predeclared | 171,153.7 | 173,981.3 | −1.63% | [−8,152.7, 2,649.0] | 39.79% / 39.86% | Base-stock |

The RL candidate failed the release checks in both new stores. CA_2's paired interval is entirely below zero; CA_3's interval crosses zero. Both stores also had lower 10th-percentile daily profit than the rule. The version 2 API therefore returns feasible orders from the validation-tuned rule and labels them `base_stock`; the unpromoted RL policy is available only with an explicit local demonstration override.

These comparisons use observed M5 sales as a demand proxy. They are conditional on one simulated economic regime and do not estimate real business profit. The multi-store result is evidence **against** claiming that the current RL algorithm improves replenishment. It motivates a new, separately evaluated policy design rather than further tuning on these test periods.
