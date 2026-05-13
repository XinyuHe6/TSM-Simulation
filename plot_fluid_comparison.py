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
    compute_blue_red_matchings,
    compute_integer_eis,
    compute_opt_from_realization,
    generate_random_graph,
    random_probability_vector,
)
from function_offline_statistics import sample_arrival_sequence
from function_offline_statistics_multiple import (
    prepare_offline_statistics_multi_state,
    simulate_offline_statistics_k_on_arrivals,
)


DEFAULT_ALGORITHMS = (
    "fluid,random_matching,degree_matching,tsm,manshadi2,manshadi3,manshadi4"
)

ALGORITHM_SPECS = {
    "fluid": {
        "csv": "fluid",
        "label": "Fluid LP Upper Bound",
        "marker": "*",
    },
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
    "fluid": "fluid",
    "fluid_lp": "fluid",
    "opt": "fluid",
    "random": "random_matching",
    "random_matching": "random_matching",
    "degree": "degree_matching",
    "degree_matching": "degree_matching",
    "tsm": "tsm",
    "offline_statistics": "manshadi2",
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
            "Compare average online ALG counts and a fluid/offline upper bound "
            "over edge probability."
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
            "Comma-separated algorithms. Options: fluid, random_matching, "
            "degree_matching, tsm, manshadi2, manshadi3, manshadi4. Use all "
            "for the default full comparison."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_fig", type=str, default=None)
    parser.add_argument("--show", action="store_true")

    parser.set_defaults(use_poisson_len=False)
    parser.add_argument(
        "--use_poisson_len",
        dest="use_poisson_len",
        action="store_true",
        help="Use a Poisson(T) number of arrivals for every algorithm.",
    )
    parser.add_argument(
        "--no_use_poisson_len",
        dest="use_poisson_len",
        action="store_false",
        help="Use exactly T arrivals for every algorithm.",
    )

    args = parser.parse_args()
    args.algorithms = parse_algorithms(args.algorithms)

    if args.out_csv is None:
        args.out_csv = f"fluid_comparison_A{args.A}_I{args.I}_T{args.T}.csv"
    if args.out_fig is None:
        args.out_fig = f"fluid_comparison_A{args.A}_I{args.I}_T{args.T}.png"

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


def selected_manshadi_try_counts(algorithms):
    try_counts = []
    for algorithm in algorithms:
        if algorithm.startswith("manshadi"):
            try_counts.append(int(algorithm[len("manshadi"):]))
    return sorted(try_counts)


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


def build_realization(neighbors, arrivals):
    hat_I = []
    edges_real = []
    for i in arrivals:
        imp_id = len(hat_I)
        hat_I.append(imp_id)
        for a in neighbors[i]:
            edges_real.append((a, imp_id))
    return hat_I, edges_real


def simulate_random_matching_alg(A_size, neighbors, arrivals):
    matched_A = [False] * A_size
    ALG = 0

    for i in arrivals:
        candidates = [a for a in neighbors[i] if not matched_A[a]]
        if candidates:
            a = random.choice(candidates)
            matched_A[a] = True
            ALG += 1

    return ALG


def advertiser_degrees(A_size, I_size, neighbors):
    degrees = [0] * A_size
    for i in range(I_size):
        for a in neighbors[i]:
            degrees[a] += 1
    return degrees


def simulate_degree_matching_alg(A_size, neighbors, arrivals, adv_degrees):
    matched_A = [False] * A_size
    ALG = 0

    for i in arrivals:
        candidates = [a for a in neighbors[i] if not matched_A[a]]
        if candidates:
            a = min(candidates, key=lambda value: adv_degrees[value])
            matched_A[a] = True
            ALG += 1

    return ALG


def prepare_tsm_state(A_size, I_size, neighbors, p, T):
    e = compute_integer_eis(T, p)
    blue_for_copy, red_for_copy, copies_of_type = compute_blue_red_matchings(
        A_size,
        I_size,
        neighbors,
        e,
    )
    return {
        "e": e,
        "blue_for_copy": blue_for_copy,
        "red_for_copy": red_for_copy,
        "copies_of_type": copies_of_type,
    }


def simulate_tsm_alg(A_size, arrivals, state):
    matched_A = [False] * A_size
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


def evaluate_one_graph(args, algorithms, neighbors, p, graph_seed):
    values_by_algorithm = {algorithm: [] for algorithm in algorithms}
    manshadi_try_counts = selected_manshadi_try_counts(algorithms)

    adv_degrees = None
    if "degree_matching" in algorithms:
        adv_degrees = advertiser_degrees(args.A, args.I, neighbors)

    tsm_state = None
    if "tsm" in algorithms:
        tsm_state = prepare_tsm_state(args.A, args.I, neighbors, p, args.T)

    manshadi_state = None
    if manshadi_try_counts:
        manshadi_state = prepare_offline_statistics_multi_state(
            A_size=args.A,
            I_size=args.I,
            neighbors=neighbors,
            p=p,
            T=args.T,
            max_tries=max(manshadi_try_counts),
            mc_trials=args.mc_trials,
            seed=graph_seed + 1000,
            use_poisson_len=args.use_poisson_len,
        )

    for run_index in range(args.runs_per_graph):
        arrival_seed = graph_seed + 1000000 + run_index
        random.seed(arrival_seed)
        arrivals = sample_arrival_sequence(
            args.I,
            p,
            args.T,
            use_poisson_len=args.use_poisson_len,
        )

        if "fluid" in values_by_algorithm:
            hat_I, edges_real = build_realization(neighbors, arrivals)
            fluid_alg = compute_opt_from_realization(args.A, hat_I, edges_real)
            values_by_algorithm["fluid"].append(float(fluid_alg))

        if "random_matching" in values_by_algorithm:
            random.seed(arrival_seed + 11)
            alg = simulate_random_matching_alg(args.A, neighbors, arrivals)
            values_by_algorithm["random_matching"].append(float(alg))

        if "degree_matching" in values_by_algorithm:
            random.seed(arrival_seed + 22)
            alg = simulate_degree_matching_alg(
                args.A,
                neighbors,
                arrivals,
                adv_degrees,
            )
            values_by_algorithm["degree_matching"].append(float(alg))

        if "tsm" in values_by_algorithm:
            random.seed(arrival_seed + 33)
            alg = simulate_tsm_alg(args.A, arrivals, tsm_state)
            values_by_algorithm["tsm"].append(float(alg))

        for try_count in manshadi_try_counts:
            algorithm = f"manshadi{try_count}"
            random.seed(arrival_seed + 44 + try_count)
            alg = simulate_offline_statistics_k_on_arrivals(
                args.A,
                arrivals,
                manshadi_state,
                try_count,
            )
            values_by_algorithm[algorithm].append(float(alg))

    return {
        algorithm: float(mean(values))
        for algorithm, values in values_by_algorithm.items()
    }


def evaluate_one_edge_prob(edge_prob, edge_index, args):
    values_by_algorithm = {algorithm: [] for algorithm in args.algorithms}

    for graph_index in range(args.num_graphs_per_point):
        cell_seed = args.seed + 100000 * edge_index + graph_index
        random.seed(cell_seed)

        neighbors = generate_random_graph(args.A, args.I, float(edge_prob))
        p = random_probability_vector(args.I)

        graph_values = evaluate_one_graph(
            args=args,
            algorithms=args.algorithms,
            neighbors=neighbors,
            p=p,
            graph_seed=cell_seed + 17,
        )

        for algorithm in args.algorithms:
            values_by_algorithm[algorithm].append(graph_values[algorithm])

    return {
        algorithm: float(mean(values))
        for algorithm, values in values_by_algorithm.items()
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
    plt.ylabel("ALG")
    plt.title(f"Online ALG vs Fluid Upper Bound (A={args.A}, I={args.I}, T={args.T})")
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
