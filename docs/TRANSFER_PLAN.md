# Predeclared cross-store transfer check: WI_2 and WI_3

The WI_1 actor and its positive WI_1 test result were fixed before this plan. WI_2 and WI_3 have not been evaluated in this project. This study checks whether the exact WI_1 coefficients transfer, rather than training or selecting another actor after observing their tests.

- Load the published WI_1 actor coefficients unchanged. Use the same official M5 CSV, 64 SKUs per store selected from days 1–1700, simulated economics and day split (train 1–1700, validation 1701–1800, test 1801–1913).
- For each destination store separately, tune the original ten base-stock rule configurations on validation profit. Apply the frozen WI_1 actor as a bounded coverage residual to that selected rule. No coefficient, feature, training seed or policy-search iteration is selected on the destination stores.
- Evaluate both policies once on each store's test period. Report profit difference and percentage, paired seven-day-block bootstrap 95% interval, fill rate, 10th-percentile daily profit and spend. Report both stores regardless of sign.
- A broad statement that the actor transfers across stores requires positive lower confidence bounds and no fill-rate or downside deterioration on **both** stores. If either fails, describe the WI_1 result as a single-store simulated improvement only.

This check does not turn simulated M5 profit into observed retail profit. Historical sales may censor demand, and procurement economics and lead times are simulated.
