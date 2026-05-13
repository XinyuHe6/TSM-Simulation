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
    generate_random_graph,
    random_probability_vector,
    simulate_many_runs,
    simulate_many_runs_degree,
)
from function_offline_statistics import simulate_many_runs_offline_statistics


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot ALG/OPT ratio vs edge_prob for TSM, function_offline_statistics, "
            "and Degree Matching when A = I = T = N."
        )
    )

    parser.add_argument("--N", type=int, default=100)

    parser.add_argument("--edge_points", type=int, default=21)
    parser.add_argument("--num_graphs_per_point", type=int, default=5)
    parser.add_argument("--runs_per_graph", type=int, default=3)

    parser.add_argument("--mc_trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_fig", type=str, default=None)
    parser.add_argument("--show", action="store_true")

    parser.set_defaults(use_poisson_len=True)
    parser.add_argument(
        "--use_poisson_len",
        dest="use_poisson_len",
        action="store_true",
        help="Use Poisson(T) arrival length for function_offline_statistics.",
    )
    parser.add_argument(
        "--no_use_poisson_len",
        dest="use_poisson_len",
        action="store_false",
        help="Use exactly T arrivals for function_offline_statistics.",
    )

    args = parser.parse_args()

    if args.N < 0:
        raise ValueError("--N must be non-negative.")

    if args.out_csv is None:
        args.out_csv = f"AeqIeqT_comparison_N{args.N}.csv"
    if args.out_fig is None:
        args.out_fig = f"AeqIeqT_comparison_N{args.N}.png"

    return args


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


def evaluate_one_edge_prob(edge_prob, edge_index, args):
    A_size = args.N
    I_size = args.N
    T = args.N

    tsm_ratios = []
    offstat_ratios = []
    degree_ratios = []

    for graph_index in range(args.num_graphs_per_point):
        cell_seed = args.seed + 100000 * edge_index + graph_index
        seed_all(cell_seed)

        neighbors = generate_random_graph(A_size, I_size, float(edge_prob))
        p = random_probability_vector(I_size)

        tsm_seed = cell_seed + 11
        offstat_seed = cell_seed + 22
        degree_seed = cell_seed + 33

        seed_all(tsm_seed)
        avg_ratio_tsm, _ = simulate_many_runs(
            A_size,
            I_size,
            neighbors,
            p,
            T,
            num_runs=args.runs_per_graph,
        )
        tsm_ratios.append(float(avg_ratio_tsm))

        avg_ratio_offstat, _ = simulate_many_runs_offline_statistics(
            A_size,
            I_size,
            neighbors,
            p,
            T,
            mc_trials=args.mc_trials,
            num_runs=args.runs_per_graph,
            seed=offstat_seed,
            use_poisson_len=args.use_poisson_len,
        )
        offstat_ratios.append(float(avg_ratio_offstat))

        seed_all(degree_seed)
        avg_ratio_degree, _ = simulate_many_runs_degree(
            A_size,
            I_size,
            neighbors,
            p,
            T,
            num_runs=args.runs_per_graph,
        )
        degree_ratios.append(float(avg_ratio_degree))

    return (
        float(mean(tsm_ratios)),
        float(mean(offstat_ratios)),
        float(mean(degree_ratios)),
    )


def write_csv(path, edge_probs, tsm_values, offstat_values, degree_values):
    with open(path, "w", encoding="utf-8") as f:
        f.write("edge_prob,tsm,function_offline_statistics,degree_matching\n")
        for edge_prob, tsm_val, offstat_val, degree_val in zip(
            edge_probs,
            tsm_values,
            offstat_values,
            degree_values,
        ):
            f.write(
                f"{float(edge_prob):.10f},"
                f"{float(tsm_val):.10f},"
                f"{float(offstat_val):.10f},"
                f"{float(degree_val):.10f}\n"
            )


def plot_results(args, edge_probs, tsm_values, offstat_values, degree_values):
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent_dir(args.out_fig)
    plt.figure(figsize=(10, 6))
    plt.plot(edge_probs, tsm_values, marker="s", linewidth=2, label="TSM")
    plt.plot(
        edge_probs,
        offstat_values,
        marker="^",
        linewidth=2,
        label="function_offline_statistics",
    )
    plt.plot(
        edge_probs,
        degree_values,
        marker="o",
        linewidth=2,
        label="Degree Matching",
    )

    plt.xlabel("edge_prob")
    plt.ylabel("ALG / OPT ratio")
    plt.title(f"ALG / OPT vs edge_prob (A=I=T={args.N})")
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
    tsm_values = []
    offstat_values = []
    degree_values = []

    for edge_index, edge_prob in enumerate(
        tqdm(edge_probs, desc="Sweep edge_prob", unit="point")
    ):
        tsm_val, offstat_val, degree_val = evaluate_one_edge_prob(
            edge_prob=edge_prob,
            edge_index=edge_index,
            args=args,
        )
        tsm_values.append(tsm_val)
        offstat_values.append(offstat_val)
        degree_values.append(degree_val)

    ensure_parent_dir(args.out_csv)
    write_csv(args.out_csv, edge_probs, tsm_values, offstat_values, degree_values)
    plot_results(args, edge_probs, tsm_values, offstat_values, degree_values)


if __name__ == "__main__":
    main()
