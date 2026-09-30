# TX_2 budget-allocation normalization: completed result

This study followed the [predeclared TX_2 protocol](ALLOCATION_PLAN.md). The [machine-readable report](m5-tx2-allocation-report.json) contains the official M5 source checksum, selected SKUs, all 30 validation candidates, daily test profits and the release decision.

| TX_2 evaluation | Validation profit | Test simulated profit | Fill rate | Mean daily spend |
| --- | ---: | ---: | ---: | ---: |
| Expanded rule, selected `beta=1`, recent sales, cover 3 | 220,627.50 | 241,791.85 | 53.04% | 2,048.00 |
| Original rule, selected on its ten candidates | 220,627.50 | 241,791.85 | 53.04% | 2,048.00 |

The expanded grid selected the **exact original rule** on validation. Consequently the paired test uplift and bootstrap interval are both zero, and the release gate retained the old rule. The 10th-percentile daily test profit was 1,571.27 for both. The best lower-normalization candidate (`beta=0.5`, long-run sales, cover 1.5) had a validation profit of 220,143.95, below the old-rule winner.

This is a useful ablation: in a budget-saturated portfolio, reducing the score normalization did not improve the validation objective on this store. The result cannot establish that `beta=1` is universally optimal. It also does not measure an RL improvement.

The 120 MB M5 source CSV is excluded from Git. Observed sales may hide past stockouts, and the cost, lead-time and capacity settings are simulated. These are not observed retail profits.
