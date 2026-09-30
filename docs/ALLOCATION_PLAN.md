# Predeclared budget-allocation study: TX_2

The CA_1–CA_4 and TX_1 test results have already been inspected. TX_2 has not been evaluated in this project. This plan fixes a second improvement before the TX_2 test period is read.

## Motivation

On TX_1, the tuned rule and both RL candidates spent the entire simulated daily purchasing budget. The allocator therefore has to rank marginal packs across SKUs. The current rule divides squared distance from target stock by `(forecast_rate + 1)`, which can underweight high-volume items. We will test whether a less aggressive normalization improves portfolio profit. This extends the order-up-to heuristic used in the [OR-Gym inventory benchmark](https://github.com/r2barati/or-gym-inventory), while preserving its explicit baseline comparisons. It is our own scoring variant, not code copied from that project.

## Frozen protocol

- Use the same official M5 file, TX_2 store, 64 active SKUs selected from days 1–1700, economics, lead times, pack sizes, shared budget/capacity, and chronological split: 1–1700 training, 1701–1800 validation, 1801–1913 test.
- Generalize the rule's score to `−(pack − desired)^2 / (forecast_rate + 1)^beta`. The old rule is exactly `beta=1`; candidate `beta` values are `{0, 0.5, 1}`. The `recent_sales` options remain `{False, True}` and the coverage grid remains `{0.5, 1, 1.5, 2, 3}`. Select the highest validation-profit combination among 30 candidates; ties prefer higher beta, then lower coverage, then long-run sales.
- Separately select the old rule from its original ten `beta=1` candidates on validation. Compare both selected policies on the same TX_2 test days. Report total simulated profit, fill rate, spending, 10th-percentile daily profit, and a paired seven-day-block bootstrap 95% interval for the new rule minus old rule.
- Adopt the expanded rule for serving only if the selected beta is below 1, the lower confidence bound is positive, fill rate falls by at most two percentage points, and 10th-percentile daily profit does not decline. Otherwise retain the old rule. This is a baseline-policy improvement study; it does not imply RL improved.

No beta, coverage grid, economics, store or release threshold will be changed after the TX_2 test result is inspected. The input is observed sales as a censored demand proxy and the economics are simulated.
