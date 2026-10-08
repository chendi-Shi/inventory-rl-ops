# TX_3 context-aware RL: second store-level holdout improvement

This study followed the [predeclared TX_3 plan](CONTEXT_ACTOR_PLAN.md). The new actor was designed using WI_2 and WI_3 **training and validation data only**; their development reports are [WI_2](m5-wi2-context-dev-report.json) and [WI_3](m5-wi3-context-dev-report.json). The TX_3 [machine-readable final report](m5-tx3-context-report.json) contains the official M5 source checksum, selected SKUs, all 48 baseline candidates, training windows, actor checkpoints and daily test profits. The [integrity-checked model bundle](../models/tx3-context) is included for the guarded v4 API.

| TX_3 days 1801–1913 | Strong rule | Context-aware RL actor |
| --- | ---: | ---: |
| Simulated profit | 252,881.50 | **256,650.30** |
| Fill rate | 58.22% | **58.73%** |
| 10th-percentile daily profit | 1,739.18 | **1,773.42** |
| Mean daily spend | 2,030.87 | 2,048.00 |

The actor's paired simulated-profit difference was **+3,768.80 (+1.49%)**, with a seven-day-block bootstrap 95% interval of **[+763.18, +6,993.33]**. The selected validation baseline used recent sales, coverage 3.0 and score-normalization exponent 1.0. The actor was seed 22, iteration 2, selected at validation profit 229,366.05 versus 225,275.60 for the strong rule. All predeclared release checks passed: positive lower profit bound, no fill-rate drop and no lower-tail daily-profit drop. The bundle therefore marks `context_actor_rl` active for TX_3.

The zero-parameter actor exactly reproduces the strong rule, and both policies use the same hard budget and capacity allocator. The new actor adjusts SKU priority using local sales/inventory features and portfolio budget pressure; its coefficients were learned from whole-portfolio sequential return. This is a **separately trained** TX_3 policy, not the WI_1 actor transplanted across stores. The WI_1 coefficients failed frozen transfer on WI_2/WI_3, as [reported](TRANSFER_RESULTS.md). WI_1 and TX_3 provide two predeclared positive *store-specific* offline results, not evidence that one universal set of coefficients works across stores.

Both store tests cover the **same M5 calendar days 1801–1913**. The TX_3 method was developed after other stores' test outcomes were known, although TX_3 itself was not evaluated before its plan was committed. These are two store-level holdouts, not two independent future time periods or a fresh prospective replication.

M5 contains observed sales, not unconstrained demand, and may hide historical stockouts. Prices, procurement/holding costs, penalties, budget, capacity and lead times are simulated. The figures are not observed business profit, and the day-block interval does not capture uncertainty across economic assumptions or future stores.

A later [retrospective economic stress test](ECONOMICS_STRESS_RESULTS.md) keeps this actor and rule frozen. TX_3's point advantage stays positive in its tested scenarios but becomes only +91.65 simulated units with 25% higher unit procurement cost; the exploratory interval crosses zero. This is a diagnostic on the same test period, not a fresh holdout.

The [subsequently frozen later-period evaluation](NEW_TIME_HOLDOUT_RESULTS.md) replays the same actor and rule on M5 days 1914–1941. Its +1.56% actor profit point estimate has a paired interval crossing zero, and the actor's 10th-percentile daily profit is lower than the rule's; the original three-part release gate would not pass on that new window. A [quantile safety-stock comparator](SAFETY_FUTURE_RESULTS.md) also gives an inconclusive actor advantage in the new period. The previously published bundle is unchanged.
