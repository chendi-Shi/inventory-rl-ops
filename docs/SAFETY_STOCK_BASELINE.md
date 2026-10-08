# Safety-stock baseline: retrospective TX_3 diagnostic

This report adds an independently implemented, validation-tuned inventory
baseline. It was designed **after** the project's original d_1801–d_1913
TX_3 results were inspected. The comparison below is therefore a
**post-hoc diagnostic**, not a new held-out result or a policy promotion test.
The machine-readable [report](m5-safety-stock-retro-report.json) contains all
48 validation candidates, daily profits, provenance, and paired intervals.

## Method

The baseline follows the empirical lead-time-demand quantile idea used by
[VN2](https://github.com/MatiasAlvo/vn2) and the
[gym-invmgmt newsvendor heuristic](https://github.com/r2barati/gym-invmgmt-paper/blob/main/agents/newsvendor_heuristic_agent.py).
This implementation is original, depends only on NumPy, and uses the same
`PortfolioEnv`, four order sizes, shared-budget and warehouse-capacity
allocator, and simulated economics as the existing TX_3 actor and rule.

For each SKU, only the final 365 **training** days (d_1336–d_1700) are used
to form overlapping sums of historical sales over a lead-time-plus-cover
window. The positive excess of the empirical quantile above the corresponding
training mean is its fixed safety-stock premium:

```text
horizon_i = lead_time_i + cover
safety_i = max(0, quantile_q(rolling horizon_i-day sales sums)
                  - mean_train_i * horizon_i)
target_i,t = rate_i,t * horizon_i + safety_i
desired_order_i,t = max(0, target_i,t - on_hand_i,t - pipeline_i,t)
```

`rate` is either the fixed training mean or the latest seven simulated sales
days, selected on validation. The desired orders are scored for 0/4/8/16-unit
packs with the original rule's normalized squared-distance score. The common
deterministic allocator makes the final portfolio orders feasible. This is a
practical quantile approximation; classic single-product newsvendor optimality
does not extend to this multi-period, 64-SKU shared-budget setting.

The validation grid has 48 candidates: quantiles 0.70/0.85/0.95, cover days
1/2/3/4, recent-sales switch on/off, and score beta 0/1. Selection uses only
d_1701–d_1800 profit. The selected parameters were **q = 0.70, cover = 2,
recent = true, beta = 1**, with validation simulated profit **226,518.00**.
For context, the frozen actor's validation profit was 229,366.05 and the
original validation-selected rule's was 225,275.60.

## Already-inspected d_1801–d_1913 replay

| Policy | Simulated profit | Fill rate | Mean daily spend |
| --- | ---: | ---: | ---: |
| Safety-stock quantile rule | **261,171.00** | **58.92%** | 2,048.00 |
| Frozen TX_3 context actor | 256,650.30 | 58.73% | 2,048.00 |
| Original strong mean-demand rule | 252,881.50 | 58.22% | 2,030.87 |

The safety-stock rule exceeded the actor by **4,520.70 simulated units**
(**+1.76%** relative to actor profit). Their paired seven-day block-bootstrap
95% interval for the **total** difference is **[−2,066.06, +9,901.48]**:
the point estimate favors safety stock, but this interval crosses zero.
Against the original rule, the difference is **+8,289.50** with interval
**[+1,581.14, +14,332.46]**.

These results narrow the earlier TX_3 claim: the actor beat the **original
48-grid mean-demand rule** by 1.49%, but did **not** beat this subsequently
introduced safety-stock baseline in the already-inspected replay. The paired
interval does not establish which of actor and safety stock is better.
Because this baseline was conceived after seeing the replay, its intervals
do not remove researcher-selection bias. A separately frozen future period is
needed to compare the policies credibly.

That [later, predeclared 28-day comparison](SAFETY_FUTURE_RESULTS.md) is now
available. On days 1914–1941 the actor's simulated profit was 64,151.65
versus 63,223.45 for this safety rule, a reversal of the earlier point
ordering. The paired safety-minus-actor interval crosses zero; neither period
settles which policy is better. The comparison leaves the published bundle
and release decision unchanged.

## Reproduction and limits

Keep the original `sales_train_validation.csv` under `data/m5/`; the raw M5
file is excluded from Git. From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
..\.venv\Scripts\python.exe -m inventory_rl.m5_safety_stock `
  --data data/m5/sales_train_validation.csv `
  --model models/tx3-context `
  --output docs/m5-safety-stock-retro-report.json
```

The command rejects `sales_train_evaluation.csv` and any input extending
beyond d_1913. The report records the source SHA-256 and frozen actor model
SHA-256. M5 unit sales are a censored demand proxy; prices, purchasing costs,
penalties, lead times, and capacity remain synthetic. This diagnostic says
nothing about realized business profit.
