#!/usr/bin/env python3

import argparse
import os
import random

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from function_ei import (
    compute_opt_from_realization,
    generate_random_graph,
    random_probability_vector,
    simulate_many_runs,
    simulate_many_runs_degree,
)
from function_offline_statistics_multiple import (
    simulate_many_runs_offline_statistics_multi,
)


DEFAULT_ALGORITHMS = (
    "random_matching,degree_matching,tsm,manshadi2,manshadi3,manshadi4"
)

ALGORITHM_SPECS = {
    "random_matching": {
        "csv": "random_matching",
        "label": "Random Matching",
        "marker": "x",
    },
    "degree_matching": {
        "csv": "degree_matching",
        "label": "Degree Matching",
        "marker": "o",
    },
    "tsm": {
        "csv": "tsm",
        "label": "TSM",
        "marker": "s",
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
}

ALGORITHM_ALIASES = {
    "random": "random_matching",
    "random_matching": "random_matching",
    "degree": "degree_matching",
    "degree_matching": "degree_matching",
    "tsm": "tsm",
    "offline_statistics": "manshadi2",
    "offline_statistics_2try": "manshadi2",
    "manshadi": "manshadi2",
    "manshadi2": "manshadi2",
    "manshadi_2try": "manshadi2",
    "manshadi3": "manshadi3",
    "manshadi_3try": "manshadi3",
    "manshadi4": "manshadi4",
    "manshadi_4try": "manshadi4",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot ALG/OPT ratio vs edge_prob for selectable online matching "
            "algorithms."
        )
    )

    parser.add_argument("--A", type=int, default=100)
    parser.add_argument("--I", type=int, default=100)
    parser.add_argument("--T", type=int, default=100)

    parser.add_argument("--edge_points", type=int, default=21)
    parser.add_argument("--num_graphs_per_point", type=int, default=5)
    parser.add_argument("--runs_per_graph", type=int, default=3)

    parser.add_argument("--mc_trials", type=int, default=20)
    parser.add_argument(
        "--algorithms",
        type=str,
        default=DEFAULT_ALGORITHMS,
        help=(
            "Comma-separated algorithms to compare. Options: "
            "random_matching, degree_matching, tsm, manshadi2, manshadi3, "
            "manshadi4. Use all for the default full comparison."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_fig", type=str, default=None)
    parser.add_argument("--show", action="store_true")

    parser.set_defaults(use_poisson_len=True)
    parser.add_argument(
        "--use_poisson_len",
        dest="use_poisson_len",
        action="store_true",
        help="Use Poisson(T) arrival length for Manshadi matching.",
    )
    parser.add_argument(
        "--no_use_poisson_len",
        dest="use_poisson_len",
        action="store_false",
        help="Use exactly T arrivals for Manshadi matching.",
    )

    args = parser.parse_args()
    args.algorithms = parse_algorithms(args.algorithms)

    if args.out_csv is None:
        args.out_csv = f"edge_prob_comparison_A{args.A}_I{args.I}_T{args.T}.csv"
    if args.out_fig is None:
        args.out_fig = f"edge_prob_comparison_A{args.A}_I{args.I}_T{args.T}.png"

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


def seed_all(seed):
    random.seed(seed)


def linspace(start, stop, num):
    if num <= 0:
        raise ValueError("--edge_points must be positive.")
    if num == 1:
        return [float(start)]

    step = (stop - start) / (num - 1)
    return [float(start + step * idx) for idx in range(num)]


def mean(values):
    return (sum(values) / len(values)) if values else 0.0


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def simulate_random_matching_once(A_size, I_size, neighbors, p, T):
    matched_A = [False] * A_size
    ALG = 0

    hat_I = []
    edges_real = []

    for _ in range(T):
        i = random.choices(range(I_size), weights=p, k=1)[0]

        imp_id = len(hat_I)
        hat_I.append(imp_id)

        for a in neighbors[i]:
            edges_real.append((a, imp_id))

        candidates = [a for a in neighbors[i] if not matched_A[a]]
        if candidates:
            a = random.choice(candidates)
            matched_A[a] = True
            ALG += 1

    OPT = compute_opt_from_realization(A_size, hat_I, edges_real)
    return ALG, OPT


def simulate_many_runs_random_matching(A_size, I_size, neighbors, p, T, num_runs=1):
    ratios = []
    for _ in range(num_runs):
        ALG, OPT = simulate_random_matching_once(A_size, I_size, neighbors, p, T)
        if OPT > 0:
            ratios.append(ALG / OPT)
        elif T > 0:
            ratios.append(1.0)

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return avg_ratio, ratios


def selected_manshadi_try_counts(algorithms):
    try_counts = []
    for algorithm in algorithms:
        if algorithm.startswith("manshadi"):
            try_counts.append(int(algorithm[len("manshadi"):]))
    return sorted(try_counts)


def evaluate_one_edge_prob(edge_prob, edge_index, args):
    ratios_by_algorithm = {algorithm: [] for algorithm in args.algorithms}
    manshadi_try_counts = selected_manshadi_try_counts(args.algorithms)

    for graph_index in range(args.num_graphs_per_point):
        cell_seed = args.seed + 100000 * edge_index + graph_index
        seed_all(cell_seed)

        neighbors = generate_random_graph(args.A, args.I, float(edge_prob))
        p = random_probability_vector(args.I)

        if "random_matching" in ratios_by_algorithm:
            seed_all(cell_seed + 11)
            avg_ratio_random, _ = simulate_many_runs_random_matching(
                args.A,
                args.I,
                neighbors,
                p,
                args.T,
                num_runs=args.runs_per_graph,
            )
            ratios_by_algorithm["random_matching"].append(float(avg_ratio_random))

        if "degree_matching" in ratios_by_algorithm:
            seed_all(cell_seed + 22)
            avg_ratio_degree, _ = simulate_many_runs_degree(
                args.A,
                args.I,
                neighbors,
                p,
                args.T,
                num_runs=args.runs_per_graph,
            )
            ratios_by_algorithm["degree_matching"].append(float(avg_ratio_degree))

        if "tsm" in ratios_by_algorithm:
            seed_all(cell_seed + 33)
            avg_ratio_tsm, _ = simulate_many_runs(
                args.A,
                args.I,
                neighbors,
                p,
                args.T,
                num_runs=args.runs_per_graph,
            )
            ratios_by_algorithm["tsm"].append(float(avg_ratio_tsm))

        if manshadi_try_counts:
            manshadi_results = simulate_many_runs_offline_statistics_multi(
                A_size=args.A,
                I_size=args.I,
                neighbors=neighbors,
                p=p,
                T=args.T,
                try_counts=manshadi_try_counts,
                mc_trials=args.mc_trials,
                num_runs=args.runs_per_graph,
                seed=cell_seed + 44,
                use_poisson_len=args.use_poisson_len,
            )
            for try_count in manshadi_try_counts:
                algorithm = f"manshadi{try_count}"
                avg_ratio, _ = manshadi_results[try_count]
                ratios_by_algorithm[algorithm].append(float(avg_ratio))

    return {
        algorithm: float(mean(ratios))
        for algorithm, ratios in ratios_by_algorithm.items()
    }


def write_csv(path, edge_probs, values_by_algorithm, algorithms):
    with open(path, "w", encoding="utf-8") as f:
        header = ["edge_prob"] + [ALGORITHM_SPECS[a]["csv"] for a in algorithms]
        f.write(",".join(header) + "\n")

        for idx, edge_prob in enumerate(edge_probs):
            row = [f"{float(edge_prob):.10f}"]
            row.extend(
                f"{float(values_by_algorithm[algorithm][idx]):.10f}"
                for algorithm in algorithms
            )
            f.write(",".join(row) + "\n")


def plot_results(args, edge_probs, values_by_algorithm):
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent_dir(args.out_fig)
    plt.figure(figsize=(10, 6))

    for algorithm in args.algorithms:
        spec = ALGORITHM_SPECS[algorithm]
        plt.plot(
            edge_probs,
            values_by_algorithm[algorithm],
            marker=spec["marker"],
            linewidth=2,
            label=spec["label"],
        )

    plt.xlabel("edge_prob")
    plt.ylabel("ALG / OPT ratio")
    plt.title(f"ALG / OPT vs edge_prob (A={args.A}, I={args.I}, T={args.T})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=200)

    if args.show:
        plt.show()
    else:
        plt.close()


def main():
    args = parse_args()

    edge_probs = linspace(0.0, 1.0, args.edge_points)
    values_by_algorithm = {algorithm: [] for algorithm in args.algorithms}

    for edge_index, edge_prob in enumerate(
        tqdm(edge_probs, desc="Sweep edge_prob", unit="point")
    ):
        values = evaluate_one_edge_prob(
            edge_prob=edge_prob,
            edge_index=edge_index,
            args=args,
        )
        for algorithm in args.algorithms:
            values_by_algorithm[algorithm].append(values[algorithm])

    ensure_parent_dir(args.out_csv)
    write_csv(args.out_csv, edge_probs, values_by_algorithm, args.algorithms)
    plot_results(args, edge_probs, values_by_algorithm)


if __name__ == "__main__":
    main()
