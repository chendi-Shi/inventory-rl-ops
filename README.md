# Inventory RL Ops

A portfolio-scale reinforcement learning system for **64 SKUs sharing one warehouse budget and storage capacity**. Its main experiment replays item-level sales from the [M5 Walmart retail dataset](https://doi.org/10.5281/zenodo.10203108), trains a shared-network factored Double DQN, tunes a credible replenishment baseline on a separate validation period, and evaluates once on a future test period. It includes a versioned decision bundle, a constrained batch recommendation API, CI, and conservative promotion rules.

**Verification status:** the complete M5 pipeline passes automated end-to-end tests and has been run on the official 120 MB M5 CSV. Across three CA stores, the pure Double DQN candidate underperformed a validation-tuned replenishment rule by **1.45%, 5.31% and 1.63%** in simulated profit on separate 113-day test windows. The RL policy failed the offline promotion gate; the decision service serves the validated base-stock rule by default. [Cross-store results](docs/CROSS_STORE_RESULTS.md) · [CA_1 report](docs/m5-ca1-report.json). These are simulated economics, not observed business profit.

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

## Residual-RL experiment on CA_4

The [version 3 plan](docs/HYBRID_PLAN.md) was committed before CA_4 test evaluation. It combines the tuned rule with a scaled Double DQN residual; `alpha=0` reproduces the rule exactly. Candidate weights and seeds are selected on validation only. Run the fixed experiment with:

```bash
inventory-rl m5-hybrid-run --data data/m5/sales_train_validation.csv \
  --store CA_4 --skus 64 --episodes 60 --seeds 11 22 33 \
  --output artifacts/m5-ca4-hybrid
```

The CA_4 test period is reserved for one final comparison. An unpromoted residual policy falls back to the tuned rule in the API.

## Deployment boundary

The optional FastAPI service has `/v2/health` and `/v2/portfolio/recommendations`. It checks bundle integrity and portfolio dimensions, enforces budget and capacity, and **falls back to the validation-tuned base-stock rule when RL fails the offline promotion gate**. Responses name the active policy. Set `PORTFOLIO_MODEL_DIR` to the M5 artifact directory and run `uvicorn inventory_rl.api:app --host 127.0.0.1 --port 8000`. `ALLOW_UNPROMOTED=1` explicitly labels and enables the unpromoted RL policy for local demonstrations only. The repository also contains a Dockerfile; mount the reviewed artifact at `/models`.

The gate requires the lower 95% bound of paired profit uplift to be positive, a fill-rate drop of at most two percentage points, and no drop in 10th-percentile daily profit against the validation-tuned base-stock rule. Passing the gate is a research result, not purchasing approval. A [predeclared cross-store evaluation](docs/CROSS_STORE_PLAN.md) tests the pipeline in CA_2 and CA_3. The API does not include authentication, monitoring or integration with an ERP.

## What can go on a resume

> Built a 64-SKU, shared-budget inventory RL platform on M5 item-level sales with a factored Double DQN, action feasibility checks, chronological model selection, tuned operations baseline, block-bootstrap evaluation, model manifest and guarded FastAPI serving. The first real-data backtest found a −1.45% simulated profit gap to the tuned rule; the release gate selected the rule for serving.

Do not describe the −1.45% as a real retail profit change. M5 provides observed sales, which may be censored by historical stockouts; procurement costs, lead times and capacity in this project are simulated. The project demonstrates rigorous decision-system engineering and a release decision, not a proven business uplift.

## Repository map

| Area | Files |
| --- | --- |
| M5 data ingestion and time split | `src/inventory_rl/m5.py` |
| Multi-SKU simulator and constrained allocator | `src/inventory_rl/portfolio.py` |
| Training, baseline tuning, test and bootstrap | `src/inventory_rl/m5_experiment.py` |
| Residual RL experiment | `src/inventory_rl/m5_hybrid.py`, `docs/HYBRID_PLAN.md` |
| Portable policy bundle and inference | `src/inventory_rl/portfolio_artifact.py`, `src/inventory_rl/api.py` |
| Automated checks | `tests/`, `.github/workflows/ci.yml` |
| Official-data evaluation and limitations | `docs/CROSS_STORE_RESULTS.md`, `docs/M5_RESULTS.md`, `docs/M5_PROTOCOL.md` |
| Small synthetic component benchmark | `src/inventory_rl/env.py`, `docs/synthetic-results.json` |
