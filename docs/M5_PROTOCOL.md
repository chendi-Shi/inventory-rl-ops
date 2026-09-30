# M5 portfolio experiment protocol

## Source and scope

Use `sales_train_validation.csv` from the [M5 dataset](https://doi.org/10.5281/zenodo.10203108), originally distributed for the [M5 Forecasting competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data). The file contains 1,913 daily unit-sales columns for each product/store series. The repo does not redistribute it. Store `CA_1` is the default. The loader filters items with at least 90 nonzero days in the last 365 training days and selects the top 64 by training-period sales volume. This represents active, popular items in one store, not all M5 products.

## Data contract and leakage controls

- The loader validates consecutive `d_1…d_1913` columns, nonnegative sales and item/store IDs.
- Training is days 1–1700. Model checkpoints, training seed and base-stock coverage are chosen using days 1701–1800 only. Days 1801–1913 are read for final test evaluation after selections are fixed.
- SKU selection and static sales-rate features use training data only.
- During replay, past **simulated sold units** enter the observation. The controller cannot see future test sales. Actual M5 sales drive the exogenous sequence only after each action.
- A source SHA-256 and the selected item IDs are written to the report and model manifest.

M5 records sales, not unconstrained demand. Historical stockouts can censor it. The replay treats sales as a demand proxy and is therefore a decision-system benchmark, not a causal estimate of business profit.

## Policy and economics

One shared 8→64→4 network scores order packs 0, 4, 8 and 16 for each SKU. A marginal-value allocator upgrades pack choices while respecting `budget_per_sku × SKU count` and `capacity_per_sku × SKU count`. It is a fast feasible heuristic for a coupled portfolio; it is not a proof of globally optimal knapsack allocation. Replenishment has deterministic synthetic lead times of one to three days. Prices, purchase costs, holding costs and lost-sale penalties are configurable synthetic units and recorded in every report.

The simulator uses lost sales and no returns or spoilage. Revenue and costs are summed across SKUs. The learned per-SKU Q functions approximate the coupled budget problem; the allocator handles the shared constraints at decision time. This approximation can underperform an operations-research optimizer.

## Baseline and uncertainty

The rule orders toward average sales during lead time plus a safety coverage period. It has two forecast modes (long-run training mean or most recent 7-day mean) and five safety levels, all chosen on validation profit. The test report compares the locked RL policy with this tuned rule and a random feasible action baseline.

The uncertainty interval resamples **paired** daily profit differences in 7-day moving blocks to retain short weekly correlation. It is an interval conditional on this one store and selected SKU set; it does not capture uncertainty across stores, economic assumptions or training seeds. Three training seeds are tried by default, with seed/checkpoint selection on validation. A more complete claim requires multiple store and cost regimes.

## Release rule and next steps

The service promotes RL only if it passes all three test checks: positive lower 95% paired profit bound, fill-rate drop no greater than two percentage points, and 10th-percentile daily profit at least as high as the rule. Otherwise version 2 serves the validation-tuned base-stock rule. The test set is used to accept or reject the fixed candidate, not to retune it.

For a production study, obtain actual inventory, stockout, supplier lead-time and margin records; compare tuned `(s,S)` and mixed-integer replenishment; calibrate a censored-demand model; evaluate multiple stores and time origins; add authenticated serving, audit trails, drift and service-level monitoring, a human override, and a shadow rollout. No production impact is claimed by this repository.
