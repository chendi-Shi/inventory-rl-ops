# Frozen WI_1 actor transfer to WI_2 and WI_3

This check followed the [predeclared transfer plan](TRANSFER_PLAN.md). The WI_1 coefficients were loaded from the published model bundle without retraining or selection on either destination store. Each destination's original ten base-stock configurations were tuned on its validation days only. The [complete report](m5-wi-transfer-report.json) contains SKU IDs, daily test profits and source/model checksums.

| Store, days 1801–1913 | Tuned rule simulated profit | Frozen actor simulated profit | Actor difference | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| WI_2 | 204,026.70 | 201,558.15 | −2,468.55 (−1.21%) | [−7,855.58, +3,213.88] |
| WI_3 | 231,612.10 | 228,811.75 | −2,800.35 (−1.21%) | [−9,276.43, +4,107.00] |

Both transfer release gates failed. The frozen actor's fill rate was 45.71% versus 45.88% on WI_2 and 50.75% versus 50.90% on WI_3. Its 10th-percentile daily profit was 881.81 versus 946.89 on WI_2 and 1,251.11 versus 1,344.06 on WI_3. Both policies spent the full simulated daily purchasing budget in both stores.

The positive WI_1 result therefore supports a **single-store** offline simulated-profit statement, not a broad claim that the WI_1 coefficients transfer. The WI_1 rule used slightly less than the full daily budget, whereas both destination rules already exhausted it; this is an observed difference in the reports, not proof of the cause of transfer failure. Actual business profit remains unmeasured.
