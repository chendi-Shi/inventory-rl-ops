# Predeclared residual-RL evaluation in CA_4

Version 2's CA_1, CA_2 and CA_3 test results were inspected before this design. CA_4 is reserved as the next untouched store-level test. The aim is to let the learned Q-values adjust the strong replenishment rule while retaining the rule as an exact zero-residual option. A separate CA_4 model will be trained; this is not transfer from the other stores.

## Fixed policy and selection

- Keep the same M5 source, 64 active SKUs, simulated economics, shared constraints, day split (1–1700 train, 1701–1800 validation, 1801–1913 test), three seeds (11, 22, 33) and 60 episodes per seed.
- Tune the existing two-mode, five-coverage base-stock grid on validation profit. Keep the pure Double DQN checkpoint chosen by validation profit for each seed.
- For each seed, score each daily order with `rule_scores + alpha × scale × (Q − Q_for_zero_pack)`. `scale` is the median absolute marginal rule score divided by the median absolute marginal Q score on that day, with a `1e-6` denominator floor. This uses only current observations and model scores, never future sales.
- Choose `alpha` from **0, 0.25, 0.5, 1, 2** and choose the seed/alpha pair with highest validation profit. `alpha=0` exactly reproduces the tuned rule. Ties prefer the smaller alpha and then smaller seed.
- Run the selected residual policy, tuned rule and random feasible policy once on CA_4 days 1801–1913. Report paired 7-day-block bootstrap interval, fill rate, 10th-percentile daily profit, selected seed/alpha, and the number of evaluated validation candidates.
- Apply the existing promotion gate. If it fails, serve the tuned rule and label it `base_stock`; the candidate remains visible only under the explicit local demonstration override.

Selection over multiple candidates can overfit the validation period. The CA_4 test period is the only new final check for this design. No post-test alpha, seed, economics or gate changes will be made to this result.
