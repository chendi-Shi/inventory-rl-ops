# Frozen TX_3 safety-stock comparator on later M5 days 1914–1941

This supplementary comparison followed the [committed safety-period protocol](SAFETY_FUTURE_COMPARISON_PLAN.md). The safety-stock rule was introduced **after** days 1801–1913 had been inspected; its earlier apparent advantage is therefore a [post-hoc diagnostic](SAFETY_STOCK_BASELINE.md), not a fresh holdout. Its quantile `0.70`, cover `2`, recent-seven-day sales target, and score exponent `1.0` were fixed on days 1701–1800 before the later-period replay. The premium for each SKU uses only the final 365 training days ending at day 1700. The TX_3 actor and original rule are the same frozen policies used in the [two-store future-period report](NEW_TIME_HOLDOUT_RESULTS.md).

| TX_3, days 1914–1941 | Safety-stock rule | Context actor | Original strong rule |
| --- | ---: | ---: | ---: |
| Simulated profit | 63,223.45 | **64,151.65** | 63,164.30 |
| Fill rate | 54.75% | **55.21%** | 54.75% |
| Mean daily spend, simulated units | 2,048.00 | 2,048.00 | 2,032.00 |
| 10th-percentile daily profit, simulated units | 1,735.95 | 1,840.73 | **1,928.38** |

The actor earned **+928.20 simulated units (+1.47% of safety-rule profit)** more than the safety rule in these 28 days. In the report's fixed **safety-minus-actor** orientation, the paired profit difference is **−928.20**, with a seven-day circular moving-block bootstrap 95% interval of **[−2,978.25, +1,081.84]**. Safety minus the original strong rule is **+59.15**, with interval **[−1,873.34, +1,938.07]**. Both intervals cross zero; this short period does not establish that any of the three policies is reliably best. The safety rule exceeded the actor by 4,520.70 on the earlier, already-inspected 113 days, also with an interval crossing zero. The reversal of the point estimates is a reason to show both periods.

The [machine-readable report](m5-safety-future-report.json) contains all **28 paired daily profit rows** under `paired_daily_profit` and each policy's daily profit, demand, sales, and spend. The safety rule and actor share the same simulator, order sizes, budget/capacity allocator, and synthetic economics, but they do **not** use identical features: the safety rule has a static training-period quantile premium that the actor does not receive directly.

## Frozen checks and reproduction

The official evaluation CSV matched its predeclared MD5 `b806dfc9f30a745102b708c09951f6aa` and had SHA-256 `4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0`. The original validation CSV SHA-256 was `f368e66ed1dbecb48b2cc8fc589bf68b3deddbbb36bf5c88b4d6d0a09b9b6724`. Selected TX_3 SKU order and days 1–1913 sales matched the frozen context bundle and old file. The actor model SHA-256 remained `f7abb861d378a6eecc557a3a371ad2de726a32fee1b0ef093a57bde7ed9755e5`. The safety premiums computed from training data matched the published [retrospective report](m5-safety-stock-retro-report.json) exactly.

All three policies were replayed continuously from day 1801, carrying each policy's own inventory and in-transit orders into day 1914. Every safety-rule warm-up daily profit matched its published retrospective replay; every actor and original-rule warm-up daily profit matched the published TX_3 report. Only days 1914–1941 were scored here. The result did not change the model bundle, its activation, or any release decision.

After placing the official `sales_train_evaluation.csv` and `sales_train_validation.csv` under `data/m5/`, run from the repository root:

```powershell
$env:PYTHONPATH = "src"
..\.venv\Scripts\python.exe -m inventory_rl.cli m5-safety-future `
  --validation data/m5/sales_train_validation.csv `
  --evaluation data/m5/sales_train_evaluation.csv `
  --output artifacts/m5-safety-future
```

This 28-day sample spans only four nonoverlapping weeks, so its block-bootstrap intervals are **descriptive**. The comparator was designed after an earlier test was known, and M5 sales are a potentially censored demand proxy. Costs, prices, lead times, budget, and capacity are simulated. No real business-profit uplift or deployment readiness follows from these numbers.
