#!/usr/bin/env python3

import argparse
import random

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from arrival_algorithms import random_arrival_counts
from graph_algorithms import generate_graph

from .common import ensure_parent_dir, format_float, linspace, mean, plot_series


def parse_try_counts(raw_tries):
    try_counts = []
    for part in raw_tries.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("--tries values must be positive integers.")
        try_counts.append(value)
    if not try_counts:
        raise ValueError("--tries must contain at least one positive integer.")
    return sorted(set(try_counts))


def build_parser(default_tries="2,3,4,5,6,7,8,9,10"):
    parser = argparse.ArgumentParser(
        description="Compare Manshadi/offline-statistics try counts over random edge probability."
    )
    parser.add_argument("--A", type=int, default=100)
    parser.add_argument("--I", type=int, default=100)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--edge_points", type=int, default=21)
    parser.add_argument("--num_graphs_per_point", type=int, default=5)
    parser.add_argument("--runs_per_graph", type=int, default=3)
    parser.add_argument("--mc_trials", type=int, default=20)
    parser.add_argument("--tries", type=str, default=default_tries)
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(use_poisson_len=True)
    parser.add_argument("--use_poisson_len", dest="use_poisson_len", action="store_true")
    parser.add_argument("--no_use_poisson_len", dest="use_poisson_len", action="store_false")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_fig", type=str, required=True)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def evaluate_one_edge_prob(args, edge_prob, edge_index, try_counts):
    from matching_algorithms.manshadi_multiple import (
        simulate_many_runs_offline_statistics_multi,
    )

    values_by_try = {try_count: [] for try_count in try_counts}
    for graph_index in range(args.num_graphs_per_point):
        cell_seed = args.seed + 100000 * edge_index + graph_index
        random.seed(cell_seed)
        neighbors = generate_graph("random", args.A, args.I, edge_prob=edge_prob)
        e = random_arrival_counts(args.I, args.T)
        results = simulate_many_runs_offline_statistics_multi(
            A_size=args.A,
            I_size=args.I,
            neighbors=neighbors,
            p=e,
            T=args.T,
            try_counts=try_counts,
            mc_trials=args.mc_trials,
            num_runs=args.runs_per_graph,
            seed=cell_seed + 17,
            use_poisson_len=args.use_poisson_len,
        )
        for try_count in try_counts:
            avg_ratio, _ = results[try_count]
            values_by_try[try_count].append(float(avg_ratio))
    return {
        try_count: mean(values)
        for try_count, values in values_by_try.items()
    }


def write_csv(path, edge_probs, values_by_try, try_counts):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        header = ["edge_prob"] + [f"manshadi_{try_count}try" for try_count in try_counts]
        f.write(",".join(header) + "\n")
        for idx, edge_prob in enumerate(edge_probs):
            row = [format_float(edge_prob)]
            for try_count in try_counts:
                row.append(format_float(values_by_try[try_count][idx]))
            f.write(",".join(row) + "\n")


def main(default_tries="2,3,4,5,6,7,8,9,10"):
    args = build_parser(default_tries=default_tries).parse_args()
    if args.A <= 0 or args.I <= 0:
        raise ValueError("--A and --I must be positive.")
    if args.T < 0:
        raise ValueError("--T must be non-negative.")
    if args.edge_points <= 0:
        raise ValueError("--edge_points must be positive.")
    if args.num_graphs_per_point <= 0:
        raise ValueError("--num_graphs_per_point must be positive.")
    if args.runs_per_graph <= 0:
        raise ValueError("--runs_per_graph must be positive.")
    if args.mc_trials <= 0:
        raise ValueError("--mc_trials must be positive.")
    try_counts = parse_try_counts(args.tries)
    edge_probs = linspace(0.0, 1.0, args.edge_points)
    values_by_try = {try_count: [] for try_count in try_counts}

    for edge_index, edge_prob in enumerate(tqdm(edge_probs, desc="Sweep edge_prob", unit="point")):
        values = evaluate_one_edge_prob(args, edge_prob, edge_index, try_counts)
        for try_count in try_counts:
            values_by_try[try_count].append(values[try_count])

    write_csv(args.out_csv, edge_probs, values_by_try, try_counts)
    series = {
        f"Manshadi ({try_count} tries)": values_by_try[try_count]
        for try_count in try_counts
    }
    plot_series(
        args.out_fig,
        "edge_prob",
        "ALG / OPT ratio",
        f"Manshadi try-count comparison (A={args.A}, I={args.I}, T={args.T})",
        edge_probs,
        series,
        show=args.show,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
