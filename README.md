# TSM-Simulation

Simulation for two-sided bipartite matching algorithms.

The unweighted 0.7299 algorithm from Brubach et al., *Online Stochastic
Matching: New Algorithms and Bounds* (Algorithm 8), is registered as
`brubach_vw` (aliases: `brubach`, `vw`, and `unweighted_vw`). It can be used by
any modular simulation command that accepts `--algorithms`, for example:

```bash
python simulations/compare_random_edge_prob.py \
  --A 20 --I 20 --T 20 --algorithms brubach_vw,tsm
```

This implementation is the unweighted known-IID version with integral arrival
rates; it does not implement edge weights or stochastic edge rewards.
