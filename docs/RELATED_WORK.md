# GitHub benchmark review and project changes

The projects below informed this repository's **evaluation design**. Their reported profits cannot be compared numerically with ours because demand traces, network topology, reward accounting, constraints and action spaces differ. No external code or model weights were copied.

| Project | Relevant practice | What applies here |
| --- | --- | --- |
| [gym-invmgmt-paper](https://github.com/r2barati/gym-invmgmt-paper) | M5-style data adapter and common benchmark scenarios for learned, heuristic and optimization policies | Require the same sales trace, simulator and KPI definitions for all of our policies; keep source and model checksums. |
| [VN2 inventory planning](https://github.com/MatiasAlvo/vn2) | Direct policy-performance optimization and classical base-stock / newsvendor-style comparators | Our episodic CEM actor already optimizes whole-portfolio simulated return; a quantile-based replenishment rule is a useful **future stronger comparator**. We do not implement VN2's differentiable optimizer. |
| [Inventory Control with MPC and RL](https://github.com/jjalcaraz-upct/inventory-control) | Receding-horizon optimization and RL evaluated on common stochastic scenarios | An MPC or mixed-integer baseline would be informative, but must model our **64-SKU shared budget** and pack constraints rather than reusing the single-SKU problem. |
| [OR-Gym](https://github.com/hubbs5/or-gym) | Explicit inventory dynamics, lead times, lost sales and OR comparisons | Keep environment accounting transparent and preserve negative baselines; our existing accounting audit verifies the published replays. |

The immediate improvement prompted by this review is the [predeclared future-calendar check](NEW_TIME_HOLDOUT_PLAN.md): load the already released WI_1 and TX_3 bundles, verify the official M5 evaluation file, continuously replay through day 1941, and score **only** the previously unused days 1914–1941. This addresses a more important evidence gap than adding another learner to the already inspected days 1801–1913. The new results, whether positive or negative, must be published and must not be used to retune these frozen bundles.

The M5 source files remain outside Git. The [Kaggle M5 data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) describes `sales_train_evaluation.csv` as extending through day 1941 and marks the dataset subject to its competition rules; the [Zenodo record](https://zenodo.org/records/10203108) provides the downloadable mirror and checksum. Dataset use does not turn observed sales into unconstrained demand or synthetic economics into actual retailer costs.
