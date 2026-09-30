# M5 CA_1 backtest: 64 SKUs

## Reproducibility record

The source is `sales_train_validation.csv` from the [Zenodo M5 record](https://zenodo.org/records/10203108). Its MD5 is `26a366a25beb57b0a8f4c7b148758f94`, matching the checksum published by Zenodo. The tested local file's SHA-256 is `f368e66ed1dbecb48b2cc8fc589bf68b3deddbbb36bf5c88b4d6d0a09b9b6724`. The CSV is excluded from this repository.

Command (from the repository root, after installation):

```bash
inventory-rl m5-run --data data/m5/sales_train_validation.csv \
  --store CA_1 --skus 64 --episodes 60 --seeds 11 22 33 \
  --output artifacts/m5-ca1
```

Training used days 1–1700. Validation used days 1701–1800 to select a policy checkpoint and tune the base-stock rule. The final comparison used days 1801–1913 (113 days). SKU selection used only training-period sales. Seed 22 at episode 50 was selected on validation; the tuned rule used the long-run mean and two days of safety coverage. The [machine-readable report](m5-ca1-report.json) records selected items, configuration, validation history and paired daily test returns. The command also writes it to `artifacts/m5-ca1/report.json` locally.

## Held-out results

| Metric | Double DQN | Tuned base-stock rule | Random feasible policy |
| --- | ---: | ---: | ---: |
| Simulated total profit | 266,565.7 | 270,500.7 | −34,534.9 |
| Fill rate | 61.34% | 61.44% | 22.16% |
| Mean daily purchasing spend | 2,047.9 | 2,048.0 | 749.7 |
| 10th-percentile daily profit | 1,600.8 | 1,709.2 | −1,147.8 |

The paired profit difference (RL minus rule) is **−3,935.0 simulated units**, or **−1.45%** of the rule's profit. A paired, circular 7-day-block bootstrap gives a 95% interval of **[−8,691.3, 1,415.6]** for the total difference. This interval crosses zero. The fill-rate drop of 0.11 percentage points passes the two-point guard, while the 10th-percentile daily-profit guard fails. The RL model was **not promoted**. In the version 2 bundle, the default API serves the validation-tuned base-stock rule and labels it `base_stock`.

## Interpretation

This is an honest negative result against a tuned operations baseline. It shows that the shared-network Double DQN and heuristic allocator did not beat a simple replenishment rule under these assumptions. The test period was viewed once for this version, and no post-test tuning was used to select a new policy. Any subsequent design iteration should use a new predeclared holdout or another store and report it separately.

M5 contains historical **sales**, not unconstrained customer demand. Historical stockouts may censor sales, and this replay has no actual inventory records. Prices, costs, lead times, storage and budget are simulated. Thus the figures are useful for comparing algorithms inside this benchmark, but they are **not** estimates of real retail profit or deployable purchasing decisions. [Protocol and further work](M5_PROTOCOL.md).
