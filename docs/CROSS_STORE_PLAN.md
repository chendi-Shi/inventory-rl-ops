# Predeclared cross-store evaluation, version 2

The CA_1 test results in [M5_RESULTS.md](M5_RESULTS.md) were inspected before this iteration. This document fixes the next evaluation before inspecting its test outcomes. The purpose is to test whether the decision pipeline and conservative fallback behave consistently in two additional stores. This is **not** cross-store model transfer: a separate policy is trained for each store.

## Fixed configuration

- Source: the same checksum-verified M5 `sales_train_validation.csv`.
- Stores: `CA_2` and `CA_3`; 64 active SKUs per store, selected by training-period sales only.
- Economics and constraints: unchanged `PortfolioConfig` defaults from version 1.
- Split: days 1–1700 for training, 1701–1800 for checkpoint/seed and baseline tuning, 1801–1913 for one final comparison.
- RL training: 60 episodes for each of seeds 11, 22 and 33, with an 84-day training window. Select the best validation checkpoint and seed.
- Baseline: the existing 2 forecast modes × 5 safety-cover levels, selected on validation profit.
- Comparison: RL, tuned base-stock and random feasible policy. Report paired total profit difference, 7-day-block 95% interval, fill rate and 10th-percentile daily profit.
- Release: promote RL only if the lower profit interval is positive, fill rate falls by no more than two percentage points, and 10th-percentile daily profit is not lower. Otherwise the decision API serves the tuned base-stock policy. `ALLOW_UNPROMOTED=1` is a labeled local demonstration override.

No test-period result will be used to change this configuration or choose a different store. If RL does not pass, that is the result. Follow-on algorithm changes require another predeclared evaluation set.
