#!/usr/bin/env python3

import argparse
import os
import random

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from function_offline_statistics import (
    generate_random_graph,
    random_probability_vector,
    simulate_many_runs_offline_statistics,
)
from function_offline_statistics3 import simulate_many_runs_offline_statistics3


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare 2-try and 3-try offline statistics matching over edge probability."
        )
    )

    parser.add_argument("--A", type=int, default=100)
    parser.add_argument("--I", type=int, default=100)
    parser.add_argument("--T", type=int, default=100)

    parser.add_argument("--edge_points", type=int, default=21)
    parser.add_argument("--num_graphs_per_point", type=int, default=5)
    parser.add_argument("--runs_per_graph", type=int, default=3)
    parser.add_argument("--mc_trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_fig", type=str, required=True)
    parser.add_argument("--show", action="store_true")

    parser.set_defaults(use_poisson_len=True)
    parser.add_argument(
        "--use_poisson_len",
        dest="use_poisson_len",
        action="store_true",
        help="Use Poisson(T) arrival length.",
    )
    parser.add_argument(
        "--no_use_poisson_len",
        dest="use_poisson_len",
        action="store_false",
        help="Use exactly T arrivals.",
    )

    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def linspace(start, stop, num):
    if num <= 0:
        raise ValueError("--edge_points must be positive.")
    if num == 1:
        return [float(start)]

    step = (stop - start) / (num - 1)
    return [float(start + step * idx) for idx in range(num)]


def mean(values):
    return (sum(values) / len(values)) if values else 0.0


def evaluate_one_edge_prob(edge_prob, edge_index, args):
    ratios_2 = []
    ratios_3 = []

    for graph_index in range(args.num_graphs_per_point):
        cell_seed = args.seed + 100000 * edge_index + graph_index
        random.seed(cell_seed)

        neighbors = generate_random_graph(args.A, args.I, float(edge_prob))
        p = random_probability_vector(args.I)

        avg_ratio_2, _ = simulate_many_runs_offline_statistics(
            args.A,
            args.I,
            neighbors,
            p,
            args.T,
            mc_trials=args.mc_trials,
            num_runs=args.runs_per_graph,
            seed=cell_seed + 11,
            use_poisson_len=args.use_poisson_len,
        )
        ratios_2.append(float(avg_ratio_2))

        avg_ratio_3, _ = simulate_many_runs_offline_statistics3(
            args.A,
            args.I,
            neighbors,
            p,
            args.T,
            mc_trials=args.mc_trials,
            num_runs=args.runs_per_graph,
            seed=cell_seed + 22,
            use_poisson_len=args.use_poisson_len,
        )
        ratios_3.append(float(avg_ratio_3))

    return float(mean(ratios_2)), float(mean(ratios_3))


def write_csv(path, edge_probs, values_2, values_3):
    with open(path, "w", encoding="utf-8") as f:
        f.write("edge_prob,offline_statistics_2try,offline_statistics_3try\n")
        for edge_prob, value_2, value_3 in zip(edge_probs, values_2, values_3):
            f.write(
                f"{float(edge_prob):.10f},"
                f"{float(value_2):.10f},"
                f"{float(value_3):.10f}\n"
            )


def plot_results(args, edge_probs, values_2, values_3):
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent_dir(args.out_fig)
    plt.figure(figsize=(10, 6))
    plt.plot(edge_probs, values_2, marker="o", linewidth=2, label="Offline Statistics (2 tries)")
    plt.plot(edge_probs, values_3, marker="^", linewidth=2, label="Offline Statistics (3 tries)")
    plt.xlabel("edge_prob")
    plt.ylabel("ALG / OPT ratio")
    plt.title(f"Offline Statistics 2 tries vs 3 tries (A={args.A}, I={args.I}, T={args.T})")
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

    values_2 = []
    values_3 = []

    for edge_index, edge_prob in enumerate(
        tqdm(edge_probs, desc="Sweep edge_prob", unit="point")
    ):
        value_2, value_3 = evaluate_one_edge_prob(edge_prob, edge_index, args)
        values_2.append(value_2)
        values_3.append(value_3)

    ensure_parent_dir(args.out_csv)
    write_csv(args.out_csv, edge_probs, values_2, values_3)
    plot_results(args, edge_probs, values_2, values_3)


if __name__ == "__main__":
    main()
