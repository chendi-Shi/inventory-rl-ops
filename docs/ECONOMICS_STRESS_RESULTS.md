# Retrospective economic stress test of frozen policies

This is a **post-hoc diagnostic** on the already inspected M5 days 1801–1913, not another held-out result. The [machine-readable report](m5-economics-stress-report.json) was generated from the official CSV by `inventory-rl m5-stress-run`. It checks the source checksum and SKU order against the released WI_1 and TX_3 bundles, holds both actors' coefficients and both rules' validation-selected parameters fixed, and replays both policies under each alternative simulated economy. The `original` rows exactly reproduce both previously published comparisons.

| Simulated economy; one change from original | TX_3 actor minus rule | TX_3 paired 95% lower bound | WI_1 actor minus rule | WI_1 paired 95% lower bound |
| --- | ---: | ---: | ---: | ---: |
| Original | +3,768.80 | +763.18 | +4,590.80 | +805.82 |
| Sale price 20% lower: 10 → 8 | +2,780.80 | +307.94 | +3,376.80 | −122.66 |
| Sale price 20% higher: 10 → 12 | +4,756.80 | +1,253.29 | +5,804.80 | +1,682.01 |
| Lost-sale penalty removed: 2 → 0 | +2,780.80 | +307.94 | +3,376.80 | −122.66 |
| Lost-sale penalty doubled: 2 → 4 | +4,756.80 | +1,253.29 | +5,804.80 | +1,682.01 |
| Holding cost doubled: 0.15 → 0.30 | +3,545.60 | +582.21 | +5,273.60 | +1,159.88 |
| Purchase budget 25% lower: 32 → 24 per SKU | +657.45 | −2,242.76 | **−3,087.00** | −7,419.81 |
| Purchase budget 25% higher: 32 → 40 per SKU | +1,915.25 | −3,765.73 | +7,152.70 | +4,038.81 |
| Unit procurement cost 25% higher: 4 → 5 | +91.65 | −3,455.73 | **−1,963.65** | −6,552.84 |

All numbers are **simulated profit units**. Each lower bound uses the same paired circular seven-day-block bootstrap as the main reports, applied to that scenario's 113 daily profit differences. These exploratory intervals are not adjusted for multiple scenarios and do not cover uncertainty in the demand proxy, new stores or real economics.

Changing sale price or lost-sale penalty does not enter either policy's observation or allocator, so those runs keep the same orders and reprice the resulting sales. The two corresponding rows happen to have equal profit differences because each change is two units per additionally served sale. Changing budget or unit procurement cost can change the orders themselves, so those scenarios replay both state trajectories under the altered constraints.

The stress test reveals material limits: WI_1's point estimate turns negative with a lower purchasing budget or higher unit procurement cost; TX_3's point estimate stays positive in this grid but becomes small, and its interval crosses zero in all three purchasing-resource changes. The original +1.51% and +1.49% claims therefore apply to the **original simulated economy**. This exercise does not change the historical promotion gates or validate the policies for changed budgets. A new evaluation period with measured prices, shortages and lead times is needed before a real release decision.
