# Predeclared heuristic-guided exploration study: TX_1

The CA_1 through CA_4 test results were inspected before this study. TX_1 has not been evaluated in this project. This plan fixes the method and decision rule before the TX_1 test period is read.

## Motivation and source

The [multi-echelon-drl project](https://github.com/qihuazhong/multi-echelon-drl) uses a base-stock heuristic during exploration and decays its use during training ([implementation](https://github.com/qihuazhong/multi-echelon-drl/blob/main/hge.py)). We adapt that idea to the existing discrete, shared-budget Double DQN. We do not copy its code or claim its reported results transfer to M5. The [OR-Gym inventory benchmark](https://github.com/r2barati/or-gym-inventory) motivates retaining a tuned heuristic comparator and operational metrics.

## Fixed design

- Use the same official M5 CSV and simulator economics as the earlier studies. Select 64 TX_1 SKUs from days 1–1700 only. Train on days 1–1700, select on 1701–1800, test once on 1801–1913.
- Use seeds 11, 22 and 33, 60 episodes of 84 days, the same Double DQN network and update settings. Compare the original unguided trainer to a guided trainer.
- In the guided trainer, each training day uses the fixed base-stock rule with `cover=2.0`, `recent=False` with probability `0.5 × (1 − episode / 60)`. Otherwise it uses the same decaying random exploration and Q-score allocation as the unguided trainer. The rule is computed from information available on that day, and the shared allocator still enforces budget and capacity.
- For each of the six trained models, select its checkpoint on validation profit. Tune the comparator's existing ten base-stock configurations on validation only. Evaluate each model as pure Q and with alpha in `{0, 0.25, 0.5, 1, 2}`. Select the trainer, seed and alpha with the highest validation profit; ties prefer lower alpha, then unguided, then lower seed. Alpha zero exactly matches the tuned comparator.
- On TX_1 test days, report the selected candidate, baseline, random feasible policy, and the best guided and unguided candidates selected on validation within their family. Compute paired seven-day-block bootstrap 95% intervals against baseline and report fill rate and 10th-percentile daily profit.
- Promote the selected RL policy only if alpha is nonzero, the lower profit interval bound is positive, fill rate falls by at most two percentage points, and 10th-percentile daily profit does not decline. Otherwise serve the tuned base-stock rule.

No policy, seed, alpha, economics or promotion threshold will be changed after the TX_1 test result is inspected. This is an offline demand replay using censored historical sales and simulated economics, not a real profit estimate.
