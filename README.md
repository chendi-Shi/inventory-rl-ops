# Inventory RL Ops

A portfolio-scale reinforcement learning system for **64 SKUs sharing one warehouse budget and storage capacity**. It replays item-level sales from the [M5 Walmart retail dataset](https://doi.org/10.5281/zenodo.10203108), compares multiple RL methods with validation-tuned replenishment rules, and evaluates on future test periods. It includes versioned decision bundles, a constrained batch recommendation API, CI, and conservative promotion rules.

**Verified offline result:** an actor trained on **total portfolio return** by episodic policy search improved simulated profit on the predeclared WI_1 test by **4,590.80 units (+1.51%)** over the tuned rule. The paired seven-day-block 95% interval was **[+805.82, +8,490.15]**; fill rate and 10th-percentile daily profit also improved, so the actor passed the offline promotion gate. [WI_1 plan](docs/PORTFOLIO_POLICY_SEARCH_PLAN.md) · [Result](docs/PORTFOLIO_POLICY_SEARCH_RESULTS.md) · [Model bundle](models/wi1-policy-search). This is **simulated** profit, not an observed business result.

**Second store-level holdout result:** a separately trained context-aware actor improved TX_3 simulated profit by **3,768.80 units (+1.49%)** over a stronger rule selected from 48 validation candidates. The paired 95% interval was **[+763.18, +6,993.33]**; fill rate and downside daily profit also improved. [TX_3 plan](docs/CONTEXT_ACTOR_PLAN.md) · [Result](docs/CONTEXT_ACTOR_RESULTS.md) · [Model bundle](models/tx3-context). This actor adjusts SKU priority when the shared purchasing budget is scarce.

The exact WI_1 coefficients **did not transfer** to WI_2 or WI_3: both test comparisons were −1.21% against their own tuned rules. [Predeclared transfer result](docs/TRANSFER_RESULTS.md). A [post-hoc stronger-rule sensitivity check](docs/STRONGER_BASELINE_SENSITIVITY.md) still left a +1.50% WI_1 actor advantage, but is labeled diagnostic because the WI_1 test had already been inspected.

WI_1 and TX_3 were held out at the **store level**, with separate policy training and plans fixed before each store's test. They share M5 calendar days 1801–1913, and the methods were refined after results from other stores became known. They are not two independent future time periods.

**Later-calendar check of frozen policies:** using the official M5 evaluation file, we continuously replayed both already released actors and their original rules, then scored only the previously unseen days **1914–1941**. WI_1 gained **2,576.70 simulated units (+3.16%)** over its rule (84,233.30 vs 81,656.60; paired seven-day-block 95% interval **[+848.73, +4,406.71]**). TX_3 gained **987.35 (+1.56%)** over its rule (64,151.65 vs 63,164.30), but its interval **[−740.04, +2,987.21]** crosses zero. Both actors had lower 10th-percentile daily profit than their rules in these 28 days (WI_1: 1,946.37 vs 2,028.01; TX_3: 1,840.73 vs 1,928.38), so **neither passes the original no-downside-drop release gate on this new period**. The existing model bundles were not changed. Four seven-day blocks give limited precision, and all values remain simulated. [Frozen-period plan](docs/NEW_TIME_HOLDOUT_PLAN.md) · [Full result](docs/NEW_TIME_HOLDOUT_RESULTS.md) · [Daily report](docs/m5-future-holdout-report.json).

For TX_3, a separately frozen safety-stock rule scored **63,223.45** on those same 28 days: higher than the original rule, but below the actor by **928.20 (+1.47% relative to safety)**. The paired safety-minus-actor interval **[−2,978.25, +1,081.84]** crosses zero, so this comparison does not establish that either policy is better. Its stronger result on the already inspected 113-day period was **post-hoc**, and its interval also crossed zero. [Safety comparison](docs/SAFETY_FUTURE_RESULTS.md) · [Plan](docs/SAFETY_FUTURE_COMPARISON_PLAN.md).

A [post-hoc economic stress test](docs/ECONOMICS_STRESS_RESULTS.md) replays the frozen policies under changed prices, penalties and purchasing resources. WI_1's simulated profit advantage reverses when the budget falls by 25% or procurement unit cost rises by 25%; TX_3 remains slightly positive in point estimate in those cases but its exploratory interval crosses zero. These limits are part of the published result, not additional independent validations.

A separate [accounting audit](docs/ACCOUNTING_AUDIT.md) replays every published test day and reconciles the profit differences to sales revenue, purchase cost, inventory holding and lost-sale penalties. It also reports terminal on-hand and in-transit units, which the finite-window simulator gives no salvage value.

**Earlier experiments:** the complete M5 pipeline passes automated end-to-end tests and has been run on the official 120 MB M5 CSV. Across three CA stores, the pure Double DQN candidate underperformed a validation-tuned replenishment rule by **1.45%, 5.31% and 1.63%** in simulated profit on separate 113-day test windows. Those bundles still serve the validated base-stock rule. [Cross-store results](docs/CROSS_STORE_RESULTS.md) · [CA_1 report](docs/m5-ca1-report.json).

The separately predeclared [CA_4 residual-RL study](docs/HYBRID_RESULTS.md) selected a nonzero RL adjustment on validation but finished **0.16% below** the tuned rule on its untouched test period. Its release gate also chose the baseline. All outcomes, including unfavorable ones, are published.

After reviewing [heuristic-guided inventory RL](https://github.com/qihuazhong/multi-echelon-drl) and the [OR-Gym inventory benchmark](https://github.com/r2barati/or-gym-inventory), we added a guided-exploration ablation on the previously unused TX_1 store. The guided model won on validation but was **0.14% below** the tuned rule on test; the unguided model was **0.06% above**, with a confidence interval crossing zero. Neither qualified for deployment. [Plan](docs/GUIDED_EXPLORATION_PLAN.md) · [Results](docs/GUIDED_EXPLORATION_RESULTS.md).

A second [TX_2 budget-allocation study](docs/ALLOCATION_RESULTS.md) tested three ways to normalize the rule's SKU scores while preserving the existing rule as an exact candidate. Validation selected the original normalization, so the deployed fallback was unchanged. This result is published with its [predeclared plan](docs/ALLOCATION_PLAN.md).

## Why this is an RL problem

Each daily order affects future inventory because item lead times span one to three days. The agent trades service level, holding cost, purchasing cost and lost sales while it competes for a **shared** budget and storage space. It receives each SKU's stock, outstanding orders, recent sales, historical sales level, lead time, and weekly phase. The network scores four pack choices per item (0, 4, 8, 16 units). A deterministic marginal-value allocator coordinates all 64 scores into one feasible portfolio order. The simulator checks both hard constraints before every transition.

This factorization avoids a joint action space of `4^64`. The policy uses shared neural weights, replay, Double DQN targets, a target network, Huber loss and Adam. The NumPy implementation is inspectable in [agent.py](src/inventory_rl/agent.py), and the allocation logic is in [portfolio.py](src/inventory_rl/portfolio.py).

## Reproduce the M5 experiment

1. Download `sales_train_validation.csv` from the [M5 dataset record](https://zenodo.org/records/10203108) or [Kaggle competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data). Place it at `data/m5/sales_train_validation.csv`. The source data is excluded from Git. The loader records its SHA-256 in the report.
2. Install and run:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev,api]'
ruff check src tests
pytest -q
inventory-rl m5-run --data data/m5/sales_train_validation.csv \
  --store CA_1 --skus 64 --episodes 60 --seeds 11 22 33 \
  --output artifacts/m5-ca1
```

The command writes `report.json`, `portfolio_model.npz`, and `portfolio_manifest.json`. The 64 SKUs are selected using **training-period sales only**. Days 1–1700 train the policy, 1701–1800 select checkpoints and tune the base-stock rule, and 1801–1913 are held out for the final comparison. The report includes SKU IDs, source checksum, all daily test profits, total profit, fill rate, mean spending and a 7-day-block paired bootstrap interval. [Full protocol and limitations](docs/M5_PROTOCOL.md).

The official CSV is excluded from Git. Its Zenodo MD5 is `26a366a25beb57b0a8f4c7b148758f94`; the tested copy's SHA-256 is recorded in [the results](docs/M5_RESULTS.md). Without the CSV, `pytest` runs an end-to-end M5-schema fixture through the same loader, training loop, artifact checks and constrained recommendation path. A separate 3-SKU synthetic smoke test remains in the repo to validate simulator invariants; its [numbers](docs/synthetic-results.json) are explicitly separate from the M5 experiment.

To reproduce the **frozen** later-calendar check, also download `sales_train_evaluation.csv` from the same M5 source and place it in `data/m5/`. The command verifies both source files, the shared day-1–1913 history, SKU order, released bundles and the published warm-up daily profits before scoring days 1914–1941:

```bash
inventory-rl m5-future-holdout \
  --validation data/m5/sales_train_validation.csv \
  --evaluation data/m5/sales_train_evaluation.csv \
  --output artifacts/m5-future-holdout
inventory-rl m5-safety-future \
  --validation data/m5/sales_train_validation.csv \
  --evaluation data/m5/sales_train_evaluation.csv \
  --output artifacts/m5-safety-future
```

The second command reproduces the TX_3 safety-stock comparison using its validation-frozen parameters. [Later-calendar results](docs/NEW_TIME_HOLDOUT_RESULTS.md) · [Safety-stock results](docs/SAFETY_FUTURE_RESULTS.md).

## Residual-RL experiment on CA_4

The [version 3 plan](docs/HYBRID_PLAN.md) was committed before CA_4 test evaluation. It combines the tuned rule with a scaled Double DQN residual; `alpha=0` reproduces the rule exactly. Candidate weights and seeds are selected on validation only. Run the fixed experiment with:

```bash
inventory-rl m5-hybrid-run --data data/m5/sales_train_validation.csv \
  --store CA_4 --skus 64 --episodes 60 --seeds 11 22 33 \
  --output artifacts/m5-ca4-hybrid
```

The CA_4 test period was used once for the [published comparison](docs/HYBRID_RESULTS.md). An unpromoted residual policy falls back to the tuned rule in the API.

## Heuristic-guided exploration ablation on TX_1

The [predeclared TX_1 plan](docs/GUIDED_EXPLORATION_PLAN.md) compares the original trainer with a trainer that sometimes follows a fixed base-stock rule during early exploration. Both use the same M5 split, seeds, constrained allocator and evaluation gate. Reproduce it with:

```bash
inventory-rl m5-guided-run --data data/m5/sales_train_validation.csv \
  --store TX_1 --skus 64 --episodes 60 --seeds 11 22 33 \
  --output artifacts/m5-tx1-guided
```

The [TX_1 result](docs/GUIDED_EXPLORATION_RESULTS.md) is a negative ablation result: rule guidance did not improve out-of-sample profit. The report preserves both training families and all validation candidates. The active API policy remains the validation-tuned rule.

## Budget-allocation score study on TX_2

The [TX_2 plan](docs/ALLOCATION_PLAN.md) tests whether changing how the rule scales marginal scores across SKUs improves allocation when the shared purchasing budget is full. It includes the original rule exactly in the candidate grid. Reproduce it with:

```bash
inventory-rl m5-allocation-run --data data/m5/sales_train_validation.csv \
  --store TX_2 --skus 64 --output artifacts/m5-tx2-allocation
```

Validation chose the original scoring on TX_2. The [complete result](docs/ALLOCATION_RESULTS.md) records zero difference on its untouched test period, so the service's rule was not changed.

## Portfolio-return policy search on WI_1

The [predeclared WI_1 study](docs/PORTFOLIO_POLICY_SEARCH_PLAN.md) trains a five-feature, state-dependent residual around the tuned rule with a cross-entropy episodic RL search. Each candidate is scored on **whole-portfolio sequential profit**, so the learner directly sees the effect of shared budget allocation. A zero residual exactly reproduces the rule. Reproduce training and the fixed test with:

```bash
inventory-rl m5-policy-search-run --data data/m5/sales_train_validation.csv \
  --store WI_1 --skus 64 --output artifacts/m5-wi1-policy-search
```

The command saves `report.json`, `actor_model.npz` and `actor_manifest.json`. The published [model bundle](models/wi1-policy-search) lets the v3 API serve the reviewed WI_1 actor without shipping the raw M5 CSV. [Full WI_1 result](docs/PORTFOLIO_POLICY_SEARCH_RESULTS.md).

The frozen-coefficient transfer check is reproducible with `inventory-rl m5-transfer-run --data data/m5/sales_train_validation.csv`; the [WI_2/WI_3 report](docs/TRANSFER_RESULTS.md) documents the failures. The stronger-rule diagnostic is reproducible with `inventory-rl m5-sensitivity-run --data data/m5/sales_train_validation.csv` and is explicitly post-hoc.

## Context-aware SKU allocation on TX_3

The [TX_3 protocol](docs/CONTEXT_ACTOR_PLAN.md) extends portfolio-return policy search with observable stock gaps, pipeline coverage, sales velocity and global budget pressure. The rule comparator is chosen from 48 validation configurations. Zero actor coefficients exactly reproduce that strong rule. Reproduce the predeclared holdout with:

```bash
inventory-rl m5-context-run --data data/m5/sales_train_validation.csv \
  --store TX_3 --skus 64 --output artifacts/m5-tx3-context
```

The two development validation runs are reproduced with `inventory-rl m5-context-dev --data data/m5/sales_train_validation.csv --store WI_2` and the same command for `WI_3`; these commands do not evaluate test days. The [TX_3 result](docs/CONTEXT_ACTOR_RESULTS.md) passed the same offline release gate. Its reviewed [model bundle](models/tx3-context) can be served without the M5 source CSV.

## Post-hoc economics stress test

Run the released WI_1 and TX_3 models with unchanged coefficients and unchanged rule parameters across a fixed set of alternative simulated prices, penalties and purchasing resources:

```bash
inventory-rl m5-stress-run --data data/m5/sales_train_validation.csv \
  --output artifacts/m5-stress
```

The [report](docs/ECONOMICS_STRESS_RESULTS.md) includes each replay's profit difference, fill rate and paired seven-day-block interval. These are retrospective diagnostics on already inspected test days and cannot be used as fresh performance confirmation.

## Deployment boundary

The optional FastAPI service has versioned health and recommendation endpoints. It checks bundle integrity and portfolio dimensions, enforces budget and capacity, and falls back to the validation-tuned rule when a candidate fails the offline promotion gate. `/v3/portfolio/recommendations` loads the promoted WI_1 actor; `/v4/portfolio/recommendations` loads the promoted TX_3 context actor. v2 bundles from failed studies report `base_stock`. Set `POLICY_SEARCH_MODEL_DIR` or `CONTEXT_MODEL_DIR` to override the included bundles and run `uvicorn inventory_rl.api:app --host 127.0.0.1 --port 8000`. `ALLOW_UNPROMOTED=1` explicitly labels and enables an unpromoted RL policy for local demonstrations only. The repository also contains a Dockerfile.

Version 4 requires each request to include the exact SKU order or its fingerprint. `/v4/health` returns both, plus the source and model fingerprints. See the [v4 request and response contract](docs/API_V4.md) for the input shapes and a shape-only client example.

The gate requires the lower 95% bound of paired profit uplift to be positive, a fill-rate drop of at most two percentage points, and no drop in 10th-percentile daily profit against the validation-tuned base-stock rule. Passing the gate is a research result, not purchasing approval. The older test outcomes decided which policy bundle was marked active; the later 28-day frozen-policy check provides new calendar evidence but does not meet the original downside-profit condition for either actor or change the bundles. A longer new period with calibrated operational data and an operational shadow run are still needed. A [predeclared cross-store evaluation](docs/CROSS_STORE_PLAN.md) tests the pipeline in CA_2 and CA_3. The API does not include authentication, monitoring or integration with an ERP.

## What can go on a resume

> Built a 64-SKU, shared-budget inventory RL platform on M5 item-level sales. Separately trained whole-portfolio policy-search actors improved **simulated profit by 1.51% on WI_1 and 1.49% on TX_3** in predeclared store-level holdouts against validation-tuned rules. A later 28-day frozen-policy replay showed **+3.16%** on WI_1 and **+1.56%** on TX_3, with the latter interval crossing zero and both policies losing on the downside daily-profit check. Published negative DQN and frozen-transfer results, versioned model bundles, constrained FastAPI serving and CI.

Do not describe the simulated differences as real retail profit changes or claim that one fixed actor generalizes across stores. M5 provides observed sales, which may be censored by historical stockouts; procurement costs, lead times and capacity in this project are simulated. Both actors are offline research results, not purchasing authorization.

[中文简历与面试表述](docs/RESUME_CN.md).

## Repository map

| Area | Files |
| --- | --- |
| M5 data ingestion and time split | `src/inventory_rl/m5.py` |
| Multi-SKU simulator and constrained allocator | `src/inventory_rl/portfolio.py` |
| Training, baseline tuning, test and bootstrap | `src/inventory_rl/m5_experiment.py` |
| Residual RL experiment | `src/inventory_rl/m5_hybrid.py`, `docs/HYBRID_PLAN.md`, `docs/HYBRID_RESULTS.md` |
| Guided exploration ablation | `src/inventory_rl/m5_guided.py`, `docs/GUIDED_EXPLORATION_PLAN.md`, `docs/GUIDED_EXPLORATION_RESULTS.md` |
| Allocation score study | `src/inventory_rl/m5_allocation.py`, `docs/ALLOCATION_PLAN.md`, `docs/ALLOCATION_RESULTS.md` |
| Portfolio-return RL and transfer | `src/inventory_rl/m5_policy_search.py`, `src/inventory_rl/m5_transfer.py`, `docs/PORTFOLIO_POLICY_SEARCH_RESULTS.md`, `docs/TRANSFER_RESULTS.md` |
| Context-aware actor on TX_3 | `src/inventory_rl/m5_context_actor.py`, `docs/CONTEXT_ACTOR_PLAN.md`, `docs/CONTEXT_ACTOR_RESULTS.md` |
| Retrospective stress and accounting audits | `src/inventory_rl/m5_stress.py`, `src/inventory_rl/m5_accounting.py`, `docs/ECONOMICS_STRESS_RESULTS.md`, `docs/ACCOUNTING_AUDIT.md` |
| Post-hoc stronger baseline check | `src/inventory_rl/m5_sensitivity.py`, `docs/STRONGER_BASELINE_SENSITIVITY.md` |
| Portable policy bundles and inference | `src/inventory_rl/portfolio_artifact.py`, `src/inventory_rl/policy_search_artifact.py`, `src/inventory_rl/context_artifact.py`, `src/inventory_rl/api.py` |
| Automated checks | `tests/`, `.github/workflows/ci.yml` |
| Official-data evaluation and limitations | `docs/CROSS_STORE_RESULTS.md`, `docs/M5_RESULTS.md`, `docs/M5_PROTOCOL.md`, `docs/NEW_TIME_HOLDOUT_RESULTS.md`, `docs/SAFETY_FUTURE_RESULTS.md` |
| Small synthetic component benchmark | `src/inventory_rl/env.py`, `docs/synthetic-results.json` |
