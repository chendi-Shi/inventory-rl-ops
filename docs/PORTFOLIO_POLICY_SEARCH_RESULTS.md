# WI_1 portfolio-return RL: held-out improvement

This experiment followed the [predeclared WI_1 plan](PORTFOLIO_POLICY_SEARCH_PLAN.md). The [machine-readable report](m5-wi1-policy-search-report.json) records the official M5 file checksum, selected SKUs, fixed training windows, all 22 validation checkpoints, selected actor coefficients and each daily test profit. The compact [actor bundle](../models/wi1-policy-search) is included so the guarded v3 API can serve the exact selected policy without redistributing the M5 CSV.

| WI_1 days 1801–1913 | Simulated profit | Fill rate | 10th-percentile daily profit | Mean daily spend |
| --- | ---: | ---: | ---: | ---: |
| Validation-tuned base-stock rule | 303,919.15 | 87.53% | 1,639.12 | 2,017.13 |
| Portfolio-return policy-search RL | 308,509.95 | 88.47% | 1,652.12 | 2,047.01 |

The paired profit difference is **+4,590.80 simulated units (+1.51%)**, with a seven-day-block bootstrap 95% interval of **[+805.82, +8,490.15]**. The selected actor was seed 22, iteration 10; its validation profit was 265,601.90 versus 259,836.70 for the tuned rule. The baseline used long-run sales and coverage 3.0. All three predeclared promotion checks passed: positive lower profit bound, no fill-rate decline and no lower-tail daily-profit decline. The actor is marked `policy_search_rl` in the versioned model bundle and served by `/v3/portfolio/recommendations`.

This is **one held-out store** under simulated economics. The earlier factored DQN, residual and heuristic-guided experiments on CA and TX stores did not clear the release gate; they remain published. The result supports the precise statement that a portfolio-return RL actor improved **simulated offline replay profit** versus this tuned rule on WI_1. It does not establish an actual retailer profit change, general superiority across stores or causal impact. The [predeclared WI_2/WI_3 transfer check](TRANSFER_PLAN.md) assesses whether the frozen coefficients generalize.

A later [retrospective economic stress test](ECONOMICS_STRESS_RESULTS.md) finds that this fixed WI_1 actor falls below its fixed rule when purchasing budget is reduced by 25% or unit procurement cost rises by 25%. That analysis reuses the same test days and does not alter the original predeclared result.

M5 contains observed sales rather than unconstrained demand; historical stockouts could censor it. Unit prices, purchase and holding costs, penalties, budget, capacity and lead times are simulated. The bootstrap interval reflects variation across days in this one backtest, not across economics or stores.
