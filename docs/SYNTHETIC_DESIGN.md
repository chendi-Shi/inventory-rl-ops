# Design, evaluation, and operational limits

## Decision contract

The state contains on-hand inventory, every lead-time pipeline slot, prior realized demand, seasonal phase, and normalized day. Three SKUs have fixed lead times of one or two days. The action space is the Cartesian product of 0, 4, and 8 units per SKU. Purchasing budget and total on-hand plus outstanding inventory form hard constraints. The order is booked before today's demand and arrives after its lead-time periods. Invalid orders raise an error before changing state.

The simulator draws Poisson demand from SKU-specific means with a 14-day seasonal component. No historical company data is used. The seasonal phase and pipeline make the simulated transition observable; demand noise remains stochastic. A fixed seed yields the same trajectory under the same policy.

## Learning and baselines

The learner uses epsilon-greedy exploration, uniform replay, a target network, Huber loss, Adam updates, and Double DQN action selection. All Bellman targets mask invalid future actions. The rule baseline orders toward average demand over lead time plus one safety period and selects the nearest feasible discrete order. A random feasible policy checks that the task is learnable, but the rule is the promotion comparator.

## Evaluation protocol

Training uses seeds 42–241. The holdout uses seeds 10000–10029; a demand-surge stress scenario uses seeds 20000–20029 with demand multiplied by 1.25. Each policy sees the same stochastic demand stream for a given seed. The report includes profit components, fill rates, mean, standard deviation, lower decile, and a paired bootstrap interval (2,000 resamples, fixed bootstrap seed). The bootstrap estimates uncertainty across simulated episodes, not across a wider population of real warehouses.

The promotion gate is deliberately conservative:

1. Lower bound of paired profit-uplift 95% interval must exceed zero.
2. Mean fill rate cannot fall by more than two percentage points versus base-stock.
3. 10th-percentile profit must be at least the base-stock value.

Passing this offline gate is only a prerequisite for shadow evaluation, never automatic purchasing authorization. For a real rollout, add authenticated API access, audit logging, artifact registry/signing, inventory reconciliation, incident response, a human override, and gradual experiment allocation.

## Model risk

- The demand generator, product economics and lead times are invented; the model can exploit simulator assumptions that are absent in a real warehouse.
- Only one training seed and 30 evaluation seeds were used for the reported run; variance and model-selection effects remain.
- The base-stock benchmark is a simple rule, not an optimized operations-research solver.
- The action grid caps purchases at eight units per SKU per day and would need redesign for larger catalogs.
- Purchase costs are charged when ordering, so late orders near the horizon can reduce measured profit; finite-horizon effects matter.
- A SHA-256 checksum detects accidental artifact corruption but does not authenticate a publisher.
- The API is a demonstration service, without authentication, rate limits, monitoring or production integrations.

## Next validation before real use

Replay a real, timestamped SKU dataset with time-based splits; model supplier delays and returns; compare tuned base-stock and mixed-integer optimization; run several training seeds; examine cost and service-level Pareto curves; run in shadow mode before a controlled experiment. Keep the same hard constraints and promotion checks.
