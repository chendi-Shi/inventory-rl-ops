# Post-hoc accounting audit of the WI_1 and TX_3 test replays

This is an **explanatory audit after both test results were known**, not another held-out evaluation or an additional profit claim. The script loads the published, checksum-verified model bundles and official M5 CSV, replays each frozen actor and its own frozen rule on days 1801–1913, and checks **every daily profit**, total profit, fill rate and spending against the original [WI_1](m5-wi1-policy-search-report.json) and [TX_3](m5-tx3-context-report.json) reports. The complete [machine-readable accounting report](m5-accounting-audit.json) records source and model checksums, units, monetary components and terminal inventory.

All monetary values below are **simulated units**, not actual retailer revenue or profit. A positive cost difference means the actor incurred more of that cost; a positive *profit effect* means the component contributed to higher actor profit.

| Actor minus its rule, days 1801–1913 | WI_1 | TX_3 |
| --- | ---: | ---: |
| Units sold | +607 | +494 |
| Units ordered | +844 | +484 |
| Lost-sale units | −607 | −494 |
| Held inventory unit-days | −4,552 | +1,488 |
| Sales revenue effect | +6,070.00 | +4,940.00 |
| Procurement cost effect | −3,376.00 | −1,936.00 |
| Holding cost effect | +682.80 | −223.20 |
| Avoided lost-sale penalty effect | +1,214.00 | +988.00 |
| **Reconciled profit difference** | **+4,590.80** | **+3,768.80** |
| Difference from published profit difference | **0.00** | **0.00** |

The underlying accounts are `profit = sold_units × unit_price − ordered_units × unit_cost − held_inventory_unit_days × holding_cost − lost_units × lost_sale_penalty`. The reported differences equal the sum of the four profit effects exactly to cents. The actors sold more under the same historical-sales demand proxy; the WI_1 actor also reduced cumulative inventory holding. This accounting does not establish that the same decisions would increase profit with real demand or another cost structure.

| End of day 1913, on-hand + pipeline units | WI_1 rule | WI_1 actor | TX_3 rule | TX_3 actor |
| --- | ---: | ---: | ---: | ---: |
| On-hand | 1,320 | 1,569 | 1,080 | 1,070 |
| Pipeline | 528 | 516 | 540 | 540 |
| **Total inventory position** | **1,848** | **2,085** | **1,620** | **1,610** |

WI_1 finishes with 237 more units in stock or transit despite having already paid for its additional orders; TX_3 finishes with 10 fewer. The simulator charges procurement when an order is placed, holding after sales and before that day's arrivals, and **no terminal salvage value**. Terminal treatment can affect a finite-window comparison, so these balances should accompany the profit numbers. The net position difference also reconciles to `extra orders − extra sales` for each store.

## Reproduce

From the repository root, install the package and place the official `sales_train_validation.csv` at `data/m5/`. Then run:

```bash
python -m inventory_rl.m5_accounting \
  --data data/m5/sales_train_validation.csv \
  --output docs/m5-accounting-audit.json
```

The auditor rejects a source checksum or SKU order mismatch, a report/bundle coefficient or economics mismatch, and any discrepancy in published daily profit. It does not retrain or select models. The official M5 CSV is excluded from Git; the audited file's SHA-256 is recorded in the JSON report.

Both stores use the **same M5 calendar window** (days 1801–1913), although each store was evaluated under a separately predeclared, store-specific policy. Earlier stores' results informed later method development. These are not independent time replications or evidence of a universal cross-store model. Observed M5 sales may be censored by historical stockouts, while prices, procurement and holding costs, penalties, budgets, capacity and lead times are synthetic. New time periods or external data and calibrated economics are required for an independent operational claim.
