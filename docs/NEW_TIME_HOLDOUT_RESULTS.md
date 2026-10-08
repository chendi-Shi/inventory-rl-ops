# Frozen WI_1 and TX_3 policies on later M5 days 1914–1941

This evaluation followed the [committed future-period protocol](NEW_TIME_HOLDOUT_PLAN.md). The two released actor bundles, their 64-SKU orders, economics, and their own validation-selected rules were frozen before reading the added 28 days. The [machine-readable report](m5-future-holdout-report.json) includes the full **28 paired daily profit observations** for each store under `stores[].paired_daily_profit`, plus daily demand, sales, spending, and profit for both policies.

| Store, days 1914–1941 | Frozen original rule | Frozen actor | Actor minus rule |
| --- | ---: | ---: | ---: |
| WI_1 simulated profit | 81,656.60 | **84,233.30** | **+2,576.70 (+3.16%)** |
| TX_3 simulated profit | 63,164.30 | 64,151.65 | +987.35 (+1.56%) |

The paired seven-day circular moving-block bootstrap 95% interval for **total** actor-minus-rule profit is **[+848.73, +4,406.71]** for WI_1 and **[−740.04, +2,987.21]** for TX_3. WI_1 has a positive interval against its *frozen original rule* on this later period. TX_3 has a positive point estimate, but its interval crosses zero. These store results share the same 28 calendar days and are not independent time replications.

| Later-period operating metric | WI_1 rule | WI_1 actor | TX_3 rule | TX_3 actor |
| --- | ---: | ---: | ---: | ---: |
| Fill rate | 85.30% | **86.66%** | 54.75% | **55.21%** |
| Mean daily spend, simulated units | 2,048.00 | 2,048.00 | 2,032.00 | 2,048.00 |
| 10th-percentile daily profit, simulated units | **2,028.01** | 1,946.37 | **1,928.38** | 1,840.73 |

Both actors have **lower** 10th-percentile daily profit than their original rules in this period. The earlier three-part promotion gate required a positive lower profit interval bound, no material fill-rate decline, and no lower-tail daily-profit decline. Applied to this new window, **neither actor passes that same gate**: WI_1 fails the lower-tail condition, while TX_3 also lacks a positive interval lower bound. This observation does not retroactively alter either published bundle or its original release decision.

## Locked audit and reproduction

The downloaded official `sales_train_evaluation.csv` had MD5 `b806dfc9f30a745102b708c09951f6aa` and SHA-256 `4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0`. The original `sales_train_validation.csv` SHA-256 was `f368e66ed1dbecb48b2cc8fc589bf68b3deddbbb36bf5c88b4d6d0a09b9b6724`. For every selected SKU, days 1–1913 in both files matched exactly. The frozen model SHA-256 values were `71ac12c3f3ddf294f7070cf2843d0b89b7202a33f395eca026455df923a18117` for WI_1 and `f7abb861d378a6eecc557a3a371ad2de726a32fee1b0ef093a57bde7ed9755e5` for TX_3.

Each actor and rule was replayed **continuously from day 1801** with its own inventory and in-transit orders. All 113 warm-up daily profits on days 1801–1913 matched their original published reports, before the new-period metrics were written. Only days 1914–1941 contribute to the table above. No new actor, rule setting, store, or SKU was selected on these 28 days.

Download the M5 evaluation CSV from [Zenodo](https://zenodo.org/records/10203108) or [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), place both sales CSV files under `data/m5/`, then from the repository root run:

```powershell
$env:PYTHONPATH = "src"
..\.venv\Scripts\python.exe -m inventory_rl.cli m5-future-holdout `
  --validation data/m5/sales_train_validation.csv `
  --evaluation data/m5/sales_train_evaluation.csv `
  --output artifacts/m5-future-holdout
```

The 28 days contain only four nonoverlapping weeks. The seven-day-block interval is **descriptive** within this period; it does not account for store selection, model design, changing economics, or deployment uncertainty. M5 records observed sales that may be censored by historical stockouts, while price, procurement, holding cost, shortage penalty, lead time, budget, and capacity are simulated. These are **offline simulated profits**, not realized Walmart or other business profits. The [separately predeclared TX_3 safety-stock comparison](SAFETY_FUTURE_RESULTS.md) offers a stronger alternative baseline and is reported regardless of outcome.
