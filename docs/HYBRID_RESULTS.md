# CA_4 residual-RL result

The [CA_4 plan](HYBRID_PLAN.md) and [implementation commit](https://github.com/chendi-Shi/inventory-rl-ops/commit/beb970a) preceded this test run. The [machine-readable report](m5-ca4-hybrid-report.json) records all 15 validation candidates and 113 paired test-day profits. This was a separate model trained for CA_4, not a transferred CA_1/CA_2/CA_3 policy.

The validation-tuned rule used the long-run mean with three days of safety coverage. Among three training seeds and five residual weights, seed **22** at episode **60** with `alpha=0.25` had the highest validation profit: **213,763.60** simulated units versus **213,674.55** for `alpha=0`. The validation difference was only **89.05** units. `alpha=0` reproduces the rule exactly, and it was a candidate in every seed.

## One-time test result, days 1801–1913

| Metric | Selected residual policy | Tuned base-stock rule |
| --- | ---: | ---: |
| Simulated total profit | 241,755.05 | 242,135.60 |
| Fill rate | 92.76% | 93.02% |
| 10th-percentile daily profit | 1,469.54 | 1,472.23 |

The paired difference was **−380.55 simulated units**, or **−0.16%** of the rule's profit. The 95% paired 7-day-block bootstrap interval was **[−1,427.84, 626.12]**. The interval crosses zero, and the lower-tail daily profit was worse. The residual candidate therefore failed the fixed promotion gate. The version 3 decision bundle labels and serves the tuned rule as `base_stock`; `ALLOW_UNPROMOTED=1` labels the residual candidate `hybrid_unpromoted` for local demonstrations.

The small validation advantage did not survive the independent test. Selection among 15 candidates may have overfit 100 validation days. No alpha, seed, economics or gate was changed after reading CA_4 test results. The evidence supports a conservative baseline fallback, not a claim of RL uplift. M5 sales may be censored by historical stockouts, while costs, lead times, budgets and capacity here are simulated; these are benchmark figures, not observed retail profit.
