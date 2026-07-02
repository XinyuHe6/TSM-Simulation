# TSM Simulation

Reproducible experiments for two-sided online stochastic matching algorithms. The repository is self-contained: the complete implementations of TSM, Manshadi/offline statistics, and Correlated Sampling live under `matching_algorithms/` and no longer import Git-ignored scripts from the repository root.

Each experiment can produce:

- a CSV file containing the mean result at every sweep or grid point;
- a PNG containing the corresponding 2D curve or 3D surface.

## 1. Install from a fresh clone

Python 3.10 or newer is required.

```bash
git clone <repo-url>
cd TSM-Simulation

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`pip install -e .` installs `networkx`, `scipy`, `matplotlib`, and `tqdm`, then exposes the `tsm-sim` command. Alternatively, install with `python -m pip install -r requirements.txt` and invoke the package through `python -m simulations ...`.

Run a small smoke test containing every registered algorithm:

```bash
tsm-sim random-edge \
  --A 8 --I 8 --T 8 --edge_points 3 \
  --num_graphs_per_point 1 --runs_per_graph 1 --mc_trials 3 \
  --algorithms all \
  --out_csv out/smoke.csv --out_fig out/smoke.png
```

Run the test suite with:

```bash
python -m unittest discover -v
```

## 2. Comparing algorithms

The following command sweeps random bipartite-graph edge probability from 0 to 1, computes `ALG / OPT`, and draws every selected algorithm on the same figure:

```bash
tsm-sim random-edge \
  --A 100 --I 100 --T 100 --edge_points 21 \
  --algorithms random_matching,degree_matching,tsm,manshadi2,correlated_sampling,brubach_vw \
  --num_graphs_per_point 5 --runs_per_graph 3 --mc_trials 20 \
  --seed 0 \
  --out_csv out/random_edge.csv \
  --out_fig out/random_edge.png
```

Within a graph/run pair, every algorithm receives the same realized arrival sequence. `OPT` is the maximum bipartite matching after the complete realization is known. CSV values are averaged first across runs and then across graphs, so columns can be compared directly.

To compare matched-advertiser counts instead of competitive ratios:

```bash
tsm-sim raw-counts \
  --A 100 --I 100 --T 100 --edge_points 21 \
  --algorithms fluid_lp,tsm,manshadi2,brubach_vw \
  --out_csv out/raw_counts.csv --out_fig out/raw_counts.png
```

Here `fluid_lp` represents realized offline OPT. It is an upper-bound benchmark, not an online algorithm.

## 3. Available matching algorithms

`--algorithms` accepts comma-separated names, `all`, or `default`.

| Name | Description |
|---|---|
| `random_matching` | Uniformly chooses an unmatched neighboring advertiser |
| `degree_matching` | Chooses an available advertiser with minimum degree in the original graph |
| `tsm` | TSM based on expanded types, max flow, and blue/red matchings |
| `manshadi2` | Monte Carlo offline statistics followed by two shifted-partition attempts |
| `manshadi3` | The same method with up to three attempts |
| `manshadi4` | The same method with up to four attempts |
| `correlated_sampling` | Unweighted Correlated Sampling using an offline LP |
| `brubach_vw` | Unweighted known-IID implementation of Brubach et al. Algorithm 8 |
| `fluid_lp` | Offline OPT for the current realization; intended as a raw-count benchmark |

Common aliases are accepted, including `random`, `degree`, `manshadi`, `correlated`, `brubach`, `vw`, and `opt`.

All algorithms use the same two-phase interface: `prepare_matching_states(...)` performs graph-level offline preparation once, and `run_matching(...)` processes a realized arrival sequence. The registry is defined in `matching_algorithms/registry.py`.

## 4. Experiment commands

List the commands or inspect one command's arguments:

```bash
tsm-sim --help
tsm-sim random-edge --help
```

Without the installed console command, replace `tsm-sim` with `python -m simulations`:

```bash
python -m simulations random-edge --A 20 --I 20 --T 20
```

| Command | Experiment | Command-specific arguments |
|---|---|---|
| `random-edge` | Random-graph edge probability versus `ALG/OPT` | `--A`, `--I`, `--T`, `--edge_points`, `--out_csv`, `--out_fig` |
| `raw-counts` | Random-graph edge probability versus raw `ALG` | Same as `random-edge` |
| `aeqieqt` | Edge-probability comparison with `A=I=T=N` | `--N`, `--edge_points`, `--out_csv`, `--out_fig` |
| `k-regular` | k-regular degree versus `ALG/OPT` | `--A`, `--I`, `--T`, `--degree_start/end/step`, output paths |
| `erdos-renyi` | `k` versus `ALG/OPT` on `G(n,n,k/n)` | `--n`, `--T`, `--k_start/end/step`, output paths |
| `tsm-surface` | Arrival horizon and edge probability versus one algorithm | `--A`, `--I`, `--T_start/end/step`, `--edge_points`, output paths |
| `manshadi-surface` | Arrival horizon and graph parameter versus one algorithm | `--A`, `--I`, T range, graph sweep, output paths |
| `correlated-2d` | Edge probability or regular degree versus one algorithm | `--N`, graph sweep, and required output paths |
| `correlated-surface` | Arrival horizon and graph parameter versus one algorithm | `--A`, `--I`, T range, graph sweep, output paths |
| `manshadi-tries` | Direct comparison of arbitrary Manshadi retry counts | Uses the independent argument set documented below |

### Size, sweep, and output arguments

| Argument | Meaning |
|---|---|
| `--A` | Number of advertisers, the offline-side vertices |
| `--I` | Number of impression types, the online arrival types |
| `--T` | Expected or fixed total arrivals; the default arrival mode realizes exactly `T` arrivals |
| `--N` | Shortcut setting `A=I=T=N` |
| `--n` | Size of both sides in an Erdos-Renyi experiment; `T` also defaults to `n` |
| `--edge_points` | Number of evenly spaced edge-probability values in the closed interval `[0,1]` |
| `--degree_start`, `--degree_end`, `--degree_step` | Inclusive k-regular degree sweep; the end defaults to `A` |
| `--k_start`, `--k_end`, `--k_step` | Expected-degree sweep for Erdos-Renyi; actual edge probability is `k/n` |
| `--T_start`, `--T_end`, `--T_step` | Inclusive arrival-horizon sweep used by surface experiments |
| `--out_csv` | CSV output path; its parent directory is created automatically |
| `--out_fig` | PNG output path; its parent directory is created automatically |

### Arguments shared by regular experiments

| Argument | Default | Meaning |
|---|---:|---|
| `--num_graphs_per_point` | `5` | Number of independently generated graphs at each sweep/grid point |
| `--runs_per_graph` | `3` | Number of independent arrival realizations evaluated on each graph |
| `--mc_trials` | `20` | Monte Carlo trials used to prepare Manshadi algorithms; this is not `runs_per_graph` |
| `--algorithms` | Command-specific | Comma-separated algorithms; `all` selects every algorithm and `default` restores the command default |
| `--seed` | `0` | Base seed for graph, arrival, and algorithm randomness |
| `--use_poisson_len` | Off | Draw the realized arrival length from `Poisson(T)` |
| `--no_use_poisson_len` | On | Explicitly use exactly `T` realized arrivals |
| `--no_plot` | Off | Write CSV only and do not create a PNG |
| `--show` | Off | Display the figure interactively after saving; avoid this on servers and in CI |
| `--skip_failed_points` | Off | Write `NaN` and continue if algorithm preparation fails; the default raises the error |

### Graph-mode arguments

`manshadi-surface`, `correlated-2d`, and `correlated-surface` support:

| Argument | Default | Meaning |
|---|---:|---|
| `--graph_mode` | `random` | `random` sweeps edge probability; `k_regular` sweeps type degree |
| `--edge_points` | `21` | Number of `[0,1]` samples in `random` mode |
| `--regular_degree_start` | `0` | First k-regular degree |
| `--regular_degree_end` | `A` | Last k-regular degree |
| `--regular_degree_step` | `1` | k-regular degree step |

When `A=I`, `k_regular` uses a random union of cyclic perfect matchings, giving degree `k` on both sides. When `A!=I`, it uses the project's one-sided convention: every impression type has exactly `k` advertiser neighbors.

### Correlated Sampling LP arguments

These arguments matter only when `correlated_sampling` is selected:

| Argument | Default | Meaning |
|---|---:|---|
| `--corr_lp_constraint_mode` | `pair_approx` | `pair_approx` is a fast low-order approximation; `natural` solves the natural LP with cutting planes |
| `--corr_lp_max_rounds` | `20` | Maximum cutting-plane rounds in `natural` mode |
| `--corr_lp_separation_tol` | `1e-9` | Tolerance for adding violated constraints in `natural` mode |
| `--corr_lp_pair_cap` | Automatic | Optional override for the pairwise-constraint right-hand side; normally leave unset |

Example using the paper-style natural constraints:

```bash
tsm-sim random-edge \
  --A 30 --I 30 --T 30 --edge_points 11 \
  --algorithms correlated_sampling \
  --corr_lp_constraint_mode natural --corr_lp_max_rounds 200 \
  --out_csv out/corr_natural.csv --out_fig out/corr_natural.png
```

`natural` is substantially slower than `pair_approx`. Test it at a small size before scaling the experiment.

### Comparing arbitrary Manshadi retry counts

`manshadi2`, `manshadi3`, and `manshadi4` are registered algorithms. Use `manshadi-tries` to compare other retry counts:

```bash
tsm-sim manshadi-tries \
  --A 50 --I 50 --T 50 --edge_points 11 \
  --tries 2,3,4,5,8 \
  --num_graphs_per_point 3 --runs_per_graph 3 --mc_trials 50 \
  --out_csv out/manshadi_tries.csv \
  --out_fig out/manshadi_tries.png
```

The complete argument set for this command is `--A`, `--I`, `--T`, `--edge_points`, `--num_graphs_per_point`, `--runs_per_graph`, `--mc_trials`, `--tries`, `--seed`, `--use_poisson_len`, `--no_use_poisson_len`, `--out_csv`, `--out_fig`, `--no_plot`, and `--show`. Their meanings match the tables above. The only differences are that Poisson arrival length is enabled by default and both output paths are required.

## 5. Arrival generation and result aggregation

For each graph, the program first generates integer expected-arrival counts `e_i >= 0` satisfying `sum(e_i)=T`. The IID type probability is `p_i=e_i/T`. It then:

1. prepares every algorithm requiring offline state once for that graph;
2. samples an arrival sequence for each run;
3. runs all selected algorithms on that same sequence;
4. computes offline maximum matching `OPT` on the realized graph;
5. records `ALG/OPT`, or raw `ALG` for the `raw-counts` command.

When `OPT=0`, the ratio is recorded as `NaN` and excluded from the final mean. Generated files should normally be placed under `out/`; this directory is Git-ignored.

## 6. Package structure

```text
arrival_algorithms/          Arrival models and sampling
graph_algorithms/            Random, k-regular, and Erdos-Renyi generators
matching_algorithms/         Complete algorithms, internal cores, and registry
simulations/                 Experiment entry points, CSV output, and plotting
tests/                       Unit and full-registry integration tests
```

The dependency direction is now `simulations -> matching_algorithms -> package-internal cores`. No tracked Python source imports root-level `function_*.py`, local notebooks, or legacy drivers. A fresh Git clone therefore contains everything required to install, run comparisons, and generate figures.
