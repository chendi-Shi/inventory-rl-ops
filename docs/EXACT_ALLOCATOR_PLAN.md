# Exact allocation diagnostic: protocol fixed before execution

## Status and purpose

This document fixes the analysis protocol **before running the new exact-allocation diagnostic**. It is a **post-hoc diagnostic**, because this project's M5 days 1801–1913 and 1914–1941 have already been inspected. It cannot provide a fresh holdout, a new RL uplift claim, or grounds to promote a model. The published WI_1 and TX_3 results and release decisions remain as reported.

The design is motivated by [Inventory Control with MPC and RL](https://github.com/jjalcaraz-upct/inventory-control), which compares learned and optimization policies under shared evaluation conditions, and the [Google OR-Tools multiple-knapsack example](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/multiple_knapsack_sat.py), which illustrates exact constrained allocation. Our problem has a different 64-SKU portfolio, action set, and simulator. The implementation will be an original NumPy dynamic program; it will not copy either project's code or use an OR-Tools runtime dependency.

## Fixed policies, data, and windows

- Freeze the released WI_1 portfolio-return actor in `models/wi1-policy-search` and its original validation-selected base-stock rule. Freeze the released TX_3 context actor in `models/tx3-context` and its original validation-selected strong rule. Load their existing manifests, selected SKU order, source checksums, training features, and economics without refitting, changing coefficients, or selecting a new rule.
- Verify the official `sales_train_validation.csv` and `sales_train_evaluation.csv` identities and their common history as in the existing frozen future-period evaluator. Do not change the selected 64 SKUs, pack choices (`0, 4, 8, 16` units), simulator transitions, demand proxy, lead times, budget, capacity, or profit accounting.
- Replay each policy from the start of day 1801. Report days **1801–1913** and **1914–1941** separately. Continue each allocator's own inventory and pipeline state through day 1941; do not reset on day 1914. Both windows are already inspected, so every profit comparison is retrospective.
- For each store and policy, compare its current deterministic greedy allocator with an exact allocator using the **same score function**. The only intervention is the allocation algorithm. No training, validation selection, policy blending, or API deployment change is part of this study.

## Exact allocation objective

At each decision, select one of four packs for each SKU to maximize the sum of its supplied action scores, subject to the existing order budget and storage limits. Subtracting each SKU's zero-order score is allowed because it is constant across feasible allocations. The exact solver must return a feasible pack vector and a deterministic choice on ties.

Under the current simulator, all SKUs share one unit purchase cost and each unit consumes one unit of storage. For a given state, both constraints therefore reduce to one limit on **total ordered units**:

`U = min(floor(budget / unit_cost), capacity - on_hand_units - pipeline_units)`.

The four pack sizes are multiples of four, so a NumPy multiple-choice knapsack dynamic program can use four-unit capacity steps and backtrack one action per SKU. Its objective is exact **for the supplied score matrix**, not an oracle for realized economic profit or the multi-period control problem. The two existing resource limits are not independent dimensions under uniform per-unit costs and volumes. We will not describe the solver as a general two-resource optimizer.

## Comparisons and reporting

For each of WI_1 actor, WI_1 rule, TX_3 actor, and TX_3 rule, report:

1. Exact-versus-greedy total action-score gap at the **same state**, fraction of decisions with different orders, changed SKU packs, and feasibility checks. Compute this shadow comparison on both the greedy trajectory's states and the exact trajectory's states so path dependence is visible.
2. Allocator time per decision and full replay runtime under the same environment, with the timing method, hardware, and repeat count disclosed. Score computation and model loading should be excluded from allocator timing.
3. Continuous-replay simulated profit, fill rate, and 10th-percentile daily profit for greedy and exact allocation, reported separately for days 1801–1913 and 1914–1941. Include paired daily profit differences and the existing seven-day block interval as descriptive uncertainty summaries; retain daily values for audit.
4. The actor-versus-rule profit comparison under each allocator only as a **post-hoc sensitivity check**. Do not select whichever allocator yields a larger RL advantage or revise the original release gate from these inspected periods.

Unit tests will compare the dynamic program with brute-force enumeration on small SKU sets, cover tie handling and zero-order feasibility, and assert both hard limits and valid pack choices. A nonnegative exact score gap is an implementation check. A higher score need not produce higher simulated profit: scores are policy-dependent decision proxies, and different orders change later inventory states.

## Interpretation limits

M5 records observed sales, used here as an exogenous demand proxy; unavailable true demand and real supplier, cost, and capacity data limit business interpretation. Exact per-day score allocation does not solve a stochastic multi-period inventory optimization problem. Any observed profit difference may reflect the already inspected calendar, synthetic economics, and path-dependent replay. The diagnostic will be published regardless of sign and will not be presented as independent evidence that RL increases real-world profit.
