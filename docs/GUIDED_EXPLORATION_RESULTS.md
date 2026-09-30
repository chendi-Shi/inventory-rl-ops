# TX_1 heuristic-guided exploration: completed result

This study followed the [predeclared protocol](GUIDED_EXPLORATION_PLAN.md). It adapts the base-stock-guided exploration idea in [qihuazhong/multi-echelon-drl](https://github.com/qihuazhong/multi-echelon-drl/blob/main/hge.py) to the project's constrained discrete Double DQN and compares it with the unchanged trainer on a previously unused store. The [full machine-readable report](m5-tx1-guided-report.json) includes the source checksum, selected SKUs, all 30 validation candidates, training histories, daily test profits, and the exact release decision.

| TX_1 test, days 1801–1913 | Simulated profit | Difference vs tuned rule | Paired 95% interval | Fill rate | 10th-percentile daily profit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation-tuned base-stock rule | 286,375.65 | — | — | 70.73% | 1,930.86 |
| Unguided trainer, seed 22, alpha 0.5 | 286,560.30 | +184.65 (+0.06%) | [−1,754.91, +2,067.25] | 70.75% | 1,932.38 |
| Guided trainer, seed 22, alpha 1.0 | 285,975.90 | −399.75 (−0.14%) | [−3,299.18, +2,600.32] | 70.73% | 1,882.23 |

The ten baseline configurations were tuned on validation; the winner used 365-day sales and coverage 3.0. The fixed selection rule preferred the guided model on validation (253,419.25 vs 252,808.40 for the best unguided candidate), but that ranking reversed on test. The selected guided policy failed both the positive lower confidence bound and the daily downside condition, so the saved decision bundle serves `base_stock` by default. Even the unguided model's small positive test difference is statistically inconclusive and would not pass the release gate.

The experiment does **not** establish a profit improvement from guided exploration. The rule may reduce exploration diversity, and validation selection among 30 correlated candidates can favor noise; these are hypotheses rather than proven causes. The portfolio spends the full simulated daily purchasing budget under all three scored policies, so the learned score mainly changes how units are distributed across SKUs. Future work should target calibrated cross-SKU marginal value, stronger demand forecasts, and more held-out stores under a newly frozen protocol.

The 120 MB official M5 CSV is excluded from Git. This is an offline replay of observed sales, which may conceal past stockouts, using simulated procurement economics and lead times. None of the figures are observed retail profit.
