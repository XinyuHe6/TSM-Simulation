#!/usr/bin/env python3

import argparse
import math
import os
import random
import warnings

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from function_Correlated_Sampling import (
    prepare_correlated_sampling_state,
    simulate_correlated_sampling_on_arrivals,
)
from function_ei import (
    compute_blue_red_matchings,
    compute_integer_eis,
    compute_opt_from_realization,
    random_probability_vector,
)
from function_offline_statistics import sample_arrival_sequence
from function_offline_statistics_multiple import (
    prepare_offline_statistics_multi_state,
    simulate_offline_statistics_k_on_arrivals,
)


DEFAULT_ALGORITHMS = (
    "tsm,correlated_sampling,manshadi2,manshadi3,manshadi4,random,degree_matching"
)

ALGORITHM_SPECS = {
    "tsm": {"csv": "tsm", "label": "TSM", "marker": "s"},
    "correlated_sampling": {
        "csv": "correlated_sampling",
        "label": "Correlated Sampling",
        "marker": "P",
    },
    "manshadi2": {
        "csv": "manshadi_2try",
        "label": "Manshadi (2 tries)",
        "marker": "^",
    },
    "manshadi3": {
        "csv": "manshadi_3try",
        "label": "Manshadi (3 tries)",
        "marker": "D",
    },
    "manshadi4": {
        "csv": "manshadi_4try",
        "label": "Manshadi (4 tries)",
        "marker": "v",
    },
    "random": {
        "csv": "random_greedy",
        "label": "Random Greedy",
        "marker": "x",
    },
    "degree_matching": {
        "csv": "degree_matching",
        "label": "Degree Matching",
        "marker": "o",
    },
}

ALGORITHM_ALIASES = {
    "tsm": "tsm",
    "correlated": "correlated_sampling",
    "correlated_sampling": "correlated_sampling",
    "corr": "correlated_sampling",
    "manshadi": "manshadi2",
    "manshadi2": "manshadi2",
    "manshadi_2try": "manshadi2",
    "manshadi3": "manshadi3",
    "manshadi_3try": "manshadi3",
    "manshadi4": "manshadi4",
    "manshadi_4try": "manshadi4",
    "random": "random",
    "random_greedy": "random",
    "random_matching": "random",
    "degree": "degree_matching",
    "degree_matching": "degree_matching",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare ALG/OPT over Erdos-Renyi bipartite graphs G(n,n,p=k/n)."
        )
    )

    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--T", type=int, default=None)

    parser.add_argument("--k_start", type=float, default=0.0)
    parser.add_argument("--k_end", type=float, default=100.0)
    parser.add_argument("--k_step", type=float, default=1.0)

    parser.add_argument("--num_graphs_per_k", type=int, default=3)
    parser.add_argument("--runs_per_graph", type=int, default=3)

    parser.add_argument("--mc_trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--algorithms",
        type=str,
        default=DEFAULT_ALGORITHMS,
        help=(
            "Comma-separated algorithms. Options: tsm, correlated_sampling, "
            "manshadi2, manshadi3, manshadi4, random, degree_matching. Use all "
            "for the default full comparison."
        ),
    )

    parser.set_defaults(use_poisson_len=False)
    parser.add_argument("--use_poisson_len", dest="use_poisson_len", action="store_true")
    parser.add_argument("--no_use_poisson_len", dest="use_poisson_len", action="store_false")

    parser.add_argument(
        "--corr_lp_constraint_mode",
        type=str,
        choices=["natural", "pair_approx"],
        default="pair_approx",
    )
    parser.add_argument("--corr_lp_max_rounds", type=int, default=20)
    parser.add_argument("--corr_lp_separation_tol", type=float, default=1e-9)
    parser.add_argument("--corr_lp_pair_cap", type=float, default=None)
    parser.add_argument(
        "--skip_failed_points",
        action="store_true",
        help="Record NaN for correlated-sampling failures instead of stopping.",
    )

    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument(
        "--out_fig",
        type=str,
        default=None,
        help="Deprecated single-figure path; parent directory is used if --out_fig_dir is omitted.",
    )
    parser.add_argument(
        "--out_fig_dir",
        type=str,
        default=None,
        help="Directory where one figure per algorithm will be written.",
    )
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    if args.T is None:
        args.T = args.n
    args.algorithms = parse_algorithms(args.algorithms)
    if args.out_fig_dir is None:
        if args.out_fig is not None:
            args.out_fig_dir = os.path.dirname(os.path.abspath(args.out_fig))
        else:
            args.out_fig_dir = os.getcwd()
    return args


def parse_algorithms(raw_algorithms):
    if raw_algorithms.strip().lower() == "all":
        raw_algorithms = DEFAULT_ALGORITHMS

    algorithms = []
    seen = set()
    for part in raw_algorithms.split(","):
        name = part.strip().lower()
        if not name:
            continue
        canonical = ALGORITHM_ALIASES.get(name)
        if canonical is None:
            valid = ", ".join(sorted(ALGORITHM_SPECS))
            raise ValueError(f"Unknown algorithm '{part}'. Valid options: {valid}.")
        if canonical not in seen:
            algorithms.append(canonical)
            seen.add(canonical)

    if not algorithms:
        raise ValueError("--algorithms must include at least one algorithm.")
    return algorithms


def build_k_values(k_start, k_end, k_step):
    if k_step <= 0:
        raise ValueError("--k_step must be positive.")
    if k_end < k_start:
        raise ValueError("--k_end must be >= --k_start.")

    values = []
    cur = float(k_start)
    while cur <= float(k_end) + 1e-12:
        values.append(round(cur, 12))
        cur += float(k_step)
    return values


def mean(values):
    valid = [value for value in values if not math.isnan(value)]
    return (sum(valid) / len(valid)) if valid else float("nan")


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def generate_er_bipartite_graph(n, k):
    edge_prob = float(k) / float(n)
    if edge_prob < -1e-12 or edge_prob > 1.0 + 1e-12:
        raise ValueError(f"k/n must be in [0, 1], got k={k}, n={n}.")
    edge_prob = min(1.0, max(0.0, edge_prob))

    neighbors = []
    for _i in range(n):
        neigh_i = [a for a in range(n) if random.random() < edge_prob]
        neighbors.append(neigh_i)
    return neighbors, edge_prob


def build_realization(neighbors, arrivals):
    hat_I = []
    edges_real = []
    for i in arrivals:
        imp_id = len(hat_I)
        hat_I.append(imp_id)
        for a in neighbors[i]:
            edges_real.append((a, imp_id))
    return hat_I, edges_real


def advertiser_degrees(n, neighbors):
    degrees = [0] * n
    for neigh_i in neighbors:
        for a in neigh_i:
            degrees[a] += 1
    return degrees


def selected_manshadi_try_counts(algorithms):
    return sorted(
        int(algorithm[len("manshadi"):])
        for algorithm in algorithms
        if algorithm.startswith("manshadi")
    )


def prepare_tsm_state(n, neighbors, p, T):
    e = compute_integer_eis(T, p)
    blue_for_copy, red_for_copy, copies_of_type = compute_blue_red_matchings(
        n,
        n,
        neighbors,
        e,
    )
    return {
        "e": e,
        "blue_for_copy": blue_for_copy,
        "red_for_copy": red_for_copy,
        "copies_of_type": copies_of_type,
    }


def simulate_tsm_on_arrivals(n, arrivals, state):
    matched_A = [False] * n
    ALG = 0

    e = state["e"]
    blue_for_copy = state["blue_for_copy"]
    red_for_copy = state["red_for_copy"]
    copies_of_type = state["copies_of_type"]
    x_copy = [0] * len(blue_for_copy)

    for i in arrivals:
        if e[i] <= 0 or not copies_of_type[i]:
            continue

        j = random.choice(copies_of_type[i])
        if x_copy[j] == 0:
            a = blue_for_copy[j]
        elif x_copy[j] == 1:
            a = red_for_copy[j]
        else:
            a = None

        if a is not None and not matched_A[a]:
            matched_A[a] = True
            ALG += 1

        x_copy[j] += 1

    return ALG


def simulate_random_on_arrivals(n, neighbors, arrivals):
    matched_A = [False] * n
    ALG = 0

    for i in arrivals:
        candidates = [a for a in neighbors[i] if not matched_A[a]]
        if candidates:
            a = random.choice(candidates)
            matched_A[a] = True
            ALG += 1

    return ALG


def simulate_degree_on_arrivals(n, neighbors, arrivals, adv_degrees):
    matched_A = [False] * n
    ALG = 0

    for i in arrivals:
        candidates = [a for a in neighbors[i] if not matched_A[a]]
        if candidates:
            a = min(candidates, key=lambda value: adv_degrees[value])
            matched_A[a] = True
            ALG += 1

    return ALG


def build_algorithm_states(args, neighbors, p, graph_seed):
    states = {}
    manshadi_try_counts = selected_manshadi_try_counts(args.algorithms)

    if "degree_matching" in args.algorithms:
        states["degree_matching"] = {"adv_degrees": advertiser_degrees(args.n, neighbors)}

    if "tsm" in args.algorithms:
        states["tsm"] = prepare_tsm_state(args.n, neighbors, p, args.T)

    if manshadi_try_counts:
        states["manshadi"] = prepare_offline_statistics_multi_state(
            A_size=args.n,
            I_size=args.n,
            neighbors=neighbors,
            p=p,
            T=args.T,
            max_tries=max(manshadi_try_counts),
            mc_trials=args.mc_trials,
            seed=graph_seed + 1000,
            use_poisson_len=args.use_poisson_len,
        )

    if "correlated_sampling" in args.algorithms:
        states["correlated_sampling"] = prepare_correlated_sampling_state(
            A_size=args.n,
            I_size=args.n,
            neighbors=neighbors,
            p=p,
            T=args.T,
            lp_max_rounds=args.corr_lp_max_rounds,
            lp_separation_tol=args.corr_lp_separation_tol,
            lp_constraint_mode=args.corr_lp_constraint_mode,
            lp_pair_cap=args.corr_lp_pair_cap,
        )

    return states


def evaluate_one_graph(args, neighbors, p, graph_seed):
    values_by_algorithm = {algorithm: [] for algorithm in args.algorithms}

    try:
        states = build_algorithm_states(args, neighbors, p, graph_seed)
    except RuntimeError as exc:
        if "correlated_sampling" in args.algorithms and args.skip_failed_points:
            warnings.warn(f"Skipping graph because setup failed: {exc}", RuntimeWarning)
            return {algorithm: float("nan") for algorithm in args.algorithms}
        raise

    for run_index in range(args.runs_per_graph):
        arrival_seed = graph_seed + 1000000 + run_index
        random.seed(arrival_seed)
        arrivals = sample_arrival_sequence(
            args.n,
            p,
            args.T,
            use_poisson_len=args.use_poisson_len,
        )
        hat_I, edges_real = build_realization(neighbors, arrivals)
        OPT = compute_opt_from_realization(args.n, hat_I, edges_real)

        for algorithm in args.algorithms:
            if algorithm == "tsm":
                random.seed(arrival_seed + 11)
                ALG = simulate_tsm_on_arrivals(args.n, arrivals, states["tsm"])
            elif algorithm == "correlated_sampling":
                random.seed(arrival_seed + 22)
                ALG = simulate_correlated_sampling_on_arrivals(
                    args.n,
                    neighbors,
                    arrivals,
                    states["correlated_sampling"],
                )
            elif algorithm == "random":
                random.seed(arrival_seed + 33)
                ALG = simulate_random_on_arrivals(args.n, neighbors, arrivals)
            elif algorithm == "degree_matching":
                random.seed(arrival_seed + 44)
                ALG = simulate_degree_on_arrivals(
                    args.n,
                    neighbors,
                    arrivals,
                    states["degree_matching"]["adv_degrees"],
                )
            elif algorithm.startswith("manshadi"):
                try_count = int(algorithm[len("manshadi"):])
                random.seed(arrival_seed + 55 + try_count)
                ALG = simulate_offline_statistics_k_on_arrivals(
                    args.n,
                    arrivals,
                    states["manshadi"],
                    try_count,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            ratio = (ALG / OPT) if OPT > 0 else 0.0
            values_by_algorithm[algorithm].append(float(ratio))

    return {
        algorithm: mean(values)
        for algorithm, values in values_by_algorithm.items()
    }


def evaluate_one_k(k_value, k_index, args):
    values_by_algorithm = {algorithm: [] for algorithm in args.algorithms}

    for graph_index in range(args.num_graphs_per_k):
        cell_seed = args.seed + 100000 * k_index + graph_index
        random.seed(cell_seed)

        neighbors, _edge_prob = generate_er_bipartite_graph(args.n, k_value)
        p = random_probability_vector(args.n)

        graph_values = evaluate_one_graph(
            args=args,
            neighbors=neighbors,
            p=p,
            graph_seed=cell_seed + 17,
        )
        for algorithm in args.algorithms:
            values_by_algorithm[algorithm].append(graph_values[algorithm])

    return {
        algorithm: mean(values)
        for algorithm, values in values_by_algorithm.items()
    }


def write_csv(path, k_values, values_by_algorithm, algorithms, n):
    with open(path, "w", encoding="utf-8") as f:
        header = ["k", "edge_prob"] + [ALGORITHM_SPECS[a]["csv"] for a in algorithms]
        f.write(",".join(header) + "\n")
        for idx, k_value in enumerate(k_values):
            row = [f"{float(k_value):.10f}", f"{float(k_value) / float(n):.10f}"]
            for algorithm in algorithms:
                value = values_by_algorithm[algorithm][idx]
                row.append("nan" if math.isnan(value) else f"{float(value):.10f}")
            f.write(",".join(row) + "\n")


def plot_results(args, k_values, values_by_algorithm):
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out_fig_dir, exist_ok=True)

    for algorithm in args.algorithms:
        spec = ALGORITHM_SPECS[algorithm]
        fig_path = os.path.join(args.out_fig_dir, f"{spec['csv']}.png")

        plt.figure(figsize=(10, 6))
        plt.plot(
            k_values,
            values_by_algorithm[algorithm],
            marker=spec["marker"],
            linewidth=2,
            label=spec["label"],
        )

        plt.xlabel("k (edge probability p = k / n)")
        plt.ylabel("ALG / OPT ratio")
        plt.title(
            f"{spec['label']} on Erdos-Renyi G(n,n,k/n) "
            f"(n={args.n}, T={args.T})"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)

        if args.show:
            plt.show()
        else:
            plt.close()


def main():
    args = parse_args()
    k_values = build_k_values(args.k_start, args.k_end, args.k_step)
    values_by_algorithm = {algorithm: [] for algorithm in args.algorithms}

    for k_index, k_value in enumerate(tqdm(k_values, desc="Sweep k", unit="point")):
        values = evaluate_one_k(k_value, k_index, args)
        for algorithm in args.algorithms:
            values_by_algorithm[algorithm].append(values[algorithm])

    ensure_parent_dir(args.out_csv)
    write_csv(args.out_csv, k_values, values_by_algorithm, args.algorithms, args.n)
    plot_results(args, k_values, values_by_algorithm)


if __name__ == "__main__":
    main()
