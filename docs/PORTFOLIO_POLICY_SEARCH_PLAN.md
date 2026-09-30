# Predeclared portfolio-return policy-search study: WI_1

The CA_1–CA_4, TX_1 and TX_2 test results were inspected before this design. WI_1 has not been evaluated in this project. This plan fixes the method, selection rule and promotion gate before any WI_1 test replay.

## Why change the learner

The current factored DQN learns from per-SKU rewards although SKUs compete for a shared purchasing budget. We will train a residual policy against **total portfolio profit** instead. This is an episodic, derivative-free RL policy search, inspired by the shared-resource formulation in [Ding et al.](https://arxiv.org/abs/2212.07684) and the direct policy-performance objective in [Alvo et al.](https://github.com/MatiasAlvo/vn2). It does not implement either paper's specific algorithm.

## Frozen design

- Use the official M5 CSV, 64 WI_1 SKUs selected using days 1–1700, the existing simulator and synthetic economics. Days 1–1700 train; days 1701–1800 select the baseline and actor; days 1801–1913 are read once for the final test.
- Tune the existing ten base-stock configurations on validation profit. Use the selected rule as a fixed anchor. The learned actor adds a bounded per-SKU coverage residual `2 × tanh(theta · features)` to that rule's desired stock. Its five features are the clipped recent-seven-day trend against the training mean, within-week sales momentum, within-week sales volatility, normalized lead time, and weekly sine phase. `theta=0` exactly reproduces the tuned rule. The same allocator enforces budget and capacity.
- Train `theta` by cross-entropy episodic policy search using seeds 11 and 22. Each seed fixes two 84-day windows sampled from days 366–1700. Run 10 iterations, each with 8 Gaussian candidates, take the two highest total training-return candidates as elites, update the Gaussian mean to their average and multiply its standard deviation by 0.85, with a floor of 0.05. Start at zero mean and standard deviation 0.5. The fixed zero actor is also evaluated. This optimizer uses training windows only.
- Evaluate the initial actor and each iteration's mean on validation; select the seed and checkpoint with the highest validation profit. Ties prefer a smaller coefficient norm and then lower seed. Record all candidate training returns and validation checkpoints.
- Compare the selected actor and tuned rule once on WI_1 test days. Report simulated profit, fill rate, 10th-percentile daily profit, mean spend and paired seven-day-block bootstrap 95% interval. Promote the actor only if its coefficient norm is nonzero, the interval's lower bound is positive, fill rate falls by at most two percentage points, and downside daily profit does not decline.

The WI_1 test will not be used to adjust features, optimizer settings, economics or selection. A positive result would support only a **simulated profit** claim on this held-out store. It cannot establish actual business profit because M5 sales may censor demand and costs/lead times are simulated.
