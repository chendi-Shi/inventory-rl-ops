# Exact allocation audit: retrospective results

This report follows the [protocol fixed before this diagnostic](EXACT_ALLOCATOR_PLAN.md). **Every M5 day in this audit had already been inspected in this project.** These are post-hoc comparisons of two allocation algorithms, not a new holdout, a new RL profit claim, or a release decision. The full [machine-readable report](m5-allocator-audit-report.json) contains daily profits, same-state score comparisons, decisions, and timing observations.

The audit keeps the released WI_1 and TX_3 actors, each original validation-selected rule, SKU order, synthetic economics, simulator, and order packs (`0, 4, 8, 16`) fixed. Each of the four policies is replayed once with the existing greedy allocator and once with a new exact NumPy multiple-choice knapsack allocator. Each path starts at day 1801 and continues through day 1941 without resetting at day 1914. The exact solver maximizes the policy's **one-day supplied action scores**, not realized or expected multi-day profit.

## Replay outcomes

Profit and spending are simulated units. Fill is sold units divided by the M5 sales proxy treated as demand. P10 is the 10th percentile of daily portfolio profit. `Actor` means the released RL actor; `rule` means its original frozen comparator.

### Days 1801–1913 (113 previously inspected days)

| Store | Policy | Allocator | Profit | Fill | P10 daily profit | Mean daily spend |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WI_1 | Actor | Greedy | 308,509.95 | 88.47% | 1,652.12 | 2,047.01 |
| WI_1 | Actor | Exact | 308,549.95 | 88.48% | 1,610.74 | 2,046.87 |
| WI_1 | Rule | Greedy | 303,919.15 | 87.53% | 1,639.12 | 2,017.13 |
| WI_1 | Rule | Exact | 303,952.95 | 87.54% | 1,640.89 | 2,017.27 |
| TX_3 | Actor | Greedy | 256,650.30 | 58.73% | 1,773.42 | 2,048.00 |
| TX_3 | Actor | Exact | 256,723.50 | 58.73% | 1,790.53 | 2,048.00 |
| TX_3 | Rule | Greedy | 252,881.50 | 58.22% | 1,739.18 | 2,030.87 |
| TX_3 | Rule | Exact | 252,760.45 | 58.21% | 1,770.73 | 2,031.29 |

### Days 1914–1941 (28 previously inspected days)

| Store | Policy | Allocator | Profit | Fill | P10 daily profit | Mean daily spend |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| WI_1 | Actor | Greedy | 84,233.30 | 86.66% | 1,946.37 | 2,048.00 |
| WI_1 | Actor | Exact | 84,206.00 | 86.64% | 1,898.11 | 2,048.00 |
| WI_1 | Rule | Greedy | 81,656.60 | 85.30% | 2,028.01 | 2,048.00 |
| WI_1 | Rule | Exact | 81,581.75 | 85.26% | 2,027.94 | 2,048.00 |
| TX_3 | Actor | Greedy | 64,151.65 | 55.21% | 1,840.73 | 2,048.00 |
| TX_3 | Actor | Exact | 63,947.20 | 55.15% | 1,816.85 | 2,048.00 |
| TX_3 | Rule | Greedy | 63,164.30 | 54.75% | 1,928.38 | 2,032.00 |
| TX_3 | Rule | Exact | 63,347.75 | 54.78% | 1,957.68 | 2,028.57 |

The exact-minus-greedy profit differences below use paired daily outcomes and the existing seven-day block bootstrap interval. All eight intervals cross zero. With only four seven-day blocks in the later window, they are descriptive rather than precise evidence of a lasting effect.

| Store | Window | Policy | Exact − greedy profit | Relative to greedy | Paired 95% interval |
| --- | --- | --- | ---: | ---: | ---: |
| WI_1 | 1801–1913 | Actor | +40.00 | +0.01% | [−405.21, +480.79] |
| WI_1 | 1801–1913 | Rule | +33.80 | +0.01% | [−398.28, +438.13] |
| WI_1 | 1914–1941 | Actor | −27.30 | −0.03% | [−242.26, +189.62] |
| WI_1 | 1914–1941 | Rule | −74.85 | −0.09% | [−230.57, +83.28] |
| TX_3 | 1801–1913 | Actor | +73.20 | +0.03% | [−963.33, +1,085.32] |
| TX_3 | 1801–1913 | Rule | −121.05 | −0.05% | [−966.42, +710.01] |
| TX_3 | 1914–1941 | Actor | −204.45 | −0.32% | [−519.94, +68.42] |
| TX_3 | 1914–1941 | Rule | +183.45 | +0.29% | [−276.91, +701.00] |

As a retrospective sensitivity check, actor-minus-rule profit with exact allocation is +4,597.00 (+1.51%) and +2,624.25 (+3.22%) for WI_1's old and later windows; it is +3,963.05 (+1.57%) and +599.45 (+0.95%) for TX_3. These figures **do not replace** the original greedy-allocation comparisons or create a fresh RL uplift result. In particular, the later TX_3 exact-allocation actor-versus-rule interval [−1,324.92, +2,807.22] crosses zero.

## Same-state allocation checks

For each visited state, the audit evaluates both allocators on **one identical score matrix** before executing the selected allocator. `Score gap` is the sum over the window of exact score minus greedy score; `changed` gives decisions and SKU packs that differ. Greedy-path and exact-path states are reported separately because earlier orders change later inventory and pipeline. All measured score gaps were nonnegative. The exact score advantage was small, and many days produced identical orders.

| Store | Window | Policy | On greedy-path states: score gap; changed days / packs | On exact-path states: score gap; changed days / packs |
| --- | --- | --- | ---: | ---: |
| WI_1 | 1801–1913 | Actor | 1.433; 11 / 33 | 2.002; 12 / 36 |
| WI_1 | 1801–1913 | Rule | 2.475; 16 / 48 | 1.196; 10 / 30 |
| WI_1 | 1914–1941 | Actor | 0.000; 0 / 0 | 0.210; 2 / 6 |
| WI_1 | 1914–1941 | Rule | 0.941; 5 / 15 | 0.868; 3 / 9 |
| TX_3 | 1801–1913 | Actor | 2.950; 15 / 45 | 4.350; 15 / 45 |
| TX_3 | 1801–1913 | Rule | 3.735; 15 / 42 | 3.409; 15 / 43 |
| TX_3 | 1914–1941 | Actor | 0.493; 3 / 9 | 0.430; 3 / 9 |
| TX_3 | 1914–1941 | Rule | 0.794; 2 / 6 | 1.087; 3 / 9 |

WI_1 actor's greedy-path states had **zero** score gap and zero changed decisions on all 28 later days, yet the exact path earned 27.30 less profit in that window. Its different orders during days 1801–1913 carried different inventory into day 1914. This illustrates why same-state action-score optimality does not imply higher path-dependent profit. Exact allocation earned lower simulated profit in four of the eight policy-window comparisons.

Budget was binding frequently, while warehouse capacity was binding on **zero** measured decisions in these paths. The current uniform unit cost and one-unit storage footprint make the two nominal resource constraints collapse to a single limit on ordered units; this audit does not establish performance under heterogeneous SKU costs or volumes.

## Runtime and reproducibility

The following medians are milliseconds **per allocator call**. Each state measured greedy first and exact second, once each; the columns show both algorithms on the greedy path and both on the exact path. Timing excludes score construction and model loading. Full replay includes scoring, selected and shadow allocator calls, and simulation.

| Store | Window | Policy | Greedy-path median: greedy / exact | Exact-path median: greedy / exact |
| --- | --- | --- | ---: | ---: |
| WI_1 | 1801–1913 | Actor | 17.28 / 6.40 | 18.96 / 6.75 |
| WI_1 | 1801–1913 | Rule | 18.52 / 6.73 | 18.34 / 6.69 |
| WI_1 | 1914–1941 | Actor | 16.64 / 6.24 | 21.46 / 6.69 |
| WI_1 | 1914–1941 | Rule | 20.99 / 6.86 | 16.98 / 6.03 |
| TX_3 | 1801–1913 | Actor | 20.32 / 6.77 | 16.51 / 6.18 |
| TX_3 | 1801–1913 | Rule | 14.52 / 5.84 | 18.35 / 6.59 |
| TX_3 | 1914–1941 | Actor | 17.89 / 6.89 | 15.64 / 6.09 |
| TX_3 | 1914–1941 | Rule | 14.86 / 5.88 | 14.48 / 5.93 |

Across the full 141-day paths, measured replay times were WI_1 actor 3.84 s (greedy) / 4.90 s (exact), WI_1 rule 4.35 / 4.22 s, TX_3 actor 6.25 / 4.00 s, and TX_3 rule 3.25 / 4.05 s. The run used Windows, Python 3.14.7, NumPy 2.5.3, and an Intel64 Family 6 Model 140 host with eight logical CPUs. The single measurement per state, fixed call order, cache and scheduler effects, and inclusion of both shadow calls in full replay mean these timings **do not support a production speed claim**.

Both source files, all 64-SKU orders, frozen bundle and rule metadata, and old/new calendar boundaries were checked. The greedy daily profit sequence and totals on days 1801–1913 matched the original [WI_1](PORTFOLIO_POLICY_SEARCH_RESULTS.md) and [TX_3](CONTEXT_ACTOR_RESULTS.md) reports. Greedy daily profit, spend, sales-proxy demand, and sold units on days 1914–1941 matched the [published frozen-period report](NEW_TIME_HOLDOUT_RESULTS.md). The audit's `greedy_reconciled` flag is true for both stores. Source identities are:

| Artifact | Digest |
| --- | --- |
| `sales_train_validation.csv` SHA-256 | `f368e66ed1dbecb48b2cc8fc589bf68b3deddbbb36bf5c88b4d6d0a09b9b6724` |
| `sales_train_evaluation.csv` MD5 | `b806dfc9f30a745102b708c09951f6aa` |
| `sales_train_evaluation.csv` SHA-256 | `4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0` |
| WI_1 actor model SHA-256 | `71ac12c3f3ddf294f7070cf2843d0b89b7202a33f395eca026455df923a18117` |
| TX_3 actor model SHA-256 | `f7abb861d378a6eecc557a3a371ad2de726a32fee1b0ef093a57bde7ed9755e5` |

The diagnostic is inspired by the shared-evaluation design of [Inventory Control with MPC and RL](https://github.com/jjalcaraz-upct/inventory-control) and the exact constrained-allocation pattern in [Google OR-Tools' multiple-knapsack sample](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/multiple_knapsack_sat.py). The solver here is an original NumPy dynamic program for this project's one-resource equivalent, not that project's MPC or OR-Tools code. M5 observed sales may be censored by real stockouts, and the economics and lead times are synthetic. **No actor bundle, API policy, or deployment decision changed.**
