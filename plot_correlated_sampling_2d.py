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
    generate_graph,
    random_probability_vector,
    simulate_many_runs_correlated_sampling,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot ALG/OPT ratio for Correlated Sampling as a 2D curve over either "
            "edge_prob or k-regular degree."
        )
    )

    parser.add_argument("--N", type=int, required=True, help="Use A = I = T = N.")
    parser.add_argument(
        "--graph_mode",
        type=str,
        choices=["random", "k_regular"],
        required=True,
    )

    parser.add_argument("--edge_points", type=int, default=31)
    parser.add_argument("--regular_degree_start", type=int, default=0)
    parser.add_argument("--regular_degree_end", type=int, default=100)
    parser.add_argument("--regular_degree_step", type=int, default=1)

    parser.add_argument("--num_graphs_per_point", type=int, default=2)
    parser.add_argument("--runs_per_graph", type=int, default=3)

    parser.add_argument("--use_poisson_len", action="store_true")
    parser.add_argument("--lp_max_rounds", type=int, default=200)
    parser.add_argument("--lp_separation_tol", type=float, default=1e-9)
    parser.add_argument(
        "--lp_constraint_mode",
        type=str,
        choices=["natural", "pair_approx"],
        default="natural",
        help="LP constraints to use before correlated sampling.",
    )
    parser.add_argument(
        "--lp_pair_cap",
        type=float,
        default=None,
        help=(
            "Optional constant pair cap for pair_approx mode. By default each "
            "pair uses 1 - exp(-(lambda_i + lambda_j))."
        ),
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_fig", type=str, required=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--skip_failed_points",
        action="store_true",
        help="If the LP does not converge for a point, skip it and record NaN instead of failing.",
    )

    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def mean(values):
    return (sum(values) / len(values)) if values else 0.0


def build_x_values(args):
    if args.graph_mode == "random":
        if args.edge_points <= 0:
            raise ValueError("--edge_points must be positive.")
        if args.edge_points == 1:
            return [0.0], "edge_prob"
        step = 1.0 / float(args.edge_points - 1)
        return [step * idx for idx in range(args.edge_points)], "edge_prob"

    if args.regular_degree_step <= 0:
        raise ValueError("--regular_degree_step must be positive.")
    if args.regular_degree_end < args.regular_degree_start:
        raise ValueError("--regular_degree_end must be >= --regular_degree_start.")
    if args.regular_degree_end > args.N:
        raise ValueError(f"regular degree cannot exceed N (= {args.N}).")

    values = list(
        range(
            args.regular_degree_start,
            args.regular_degree_end + 1,
            args.regular_degree_step,
        )
    )
    if not values:
        raise ValueError("k-regular sweep produced an empty degree grid.")
    return values, "regular_degree"


def evaluate_one_point(graph_param, point_index, args):
    ratios = []
    A_size = args.N
    I_size = args.N
    T = args.N

    for graph_index in range(args.num_graphs_per_point):
        cell_seed = args.seed + 100000 * point_index + graph_index
        random.seed(cell_seed)

        neighbors = generate_graph(A_size, I_size, args.graph_mode, graph_param)
        p = random_probability_vector(I_size)

        try:
            avg_ratio_corr, _ = simulate_many_runs_correlated_sampling(
                A_size=A_size,
                I_size=I_size,
                neighbors=neighbors,
                p=p,
                T=T,
                num_runs=args.runs_per_graph,
                seed=cell_seed + 17,
                use_poisson_len=args.use_poisson_len,
                lp_max_rounds=args.lp_max_rounds,
                lp_separation_tol=args.lp_separation_tol,
                lp_constraint_mode=args.lp_constraint_mode,
                lp_pair_cap=args.lp_pair_cap,
            )
        except RuntimeError as exc:
            if not args.skip_failed_points:
                raise
            warnings.warn(
                f"Skipping {args.graph_mode} point {graph_param} "
                f"(edge/grid index {point_index}, graph {graph_index}) because LP failed: {exc}",
                RuntimeWarning,
            )
            return float("nan")
        ratios.append(float(avg_ratio_corr))

    return float(mean(ratios))


def write_csv(path, x_label, x_values, y_values):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{x_label},mean_ratio\n")
        for x_value, y_value in zip(x_values, y_values):
            if math.isnan(y_value):
                f.write(f"{float(x_value):.10f},nan\n")
            else:
                f.write(f"{float(x_value):.10f},{float(y_value):.10f}\n")


def plot_results(args, x_label, x_values, y_values):
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent_dir(args.out_fig)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker="o", linewidth=2, label="Correlated Sampling")
    plt.xlabel(x_label)
    plt.ylabel("ALG / OPT ratio")
    plt.title(
        f"Correlated Sampling ({args.lp_constraint_mode}) vs {x_label} "
        f"(A=I=T={args.N})"
    )
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
    x_values, x_label = build_x_values(args)
    y_values = []

    for point_index, graph_param in enumerate(
        tqdm(x_values, desc=f"Sweep {x_label}", unit="point")
    ):
        y_values.append(evaluate_one_point(graph_param, point_index, args))

    ensure_parent_dir(args.out_csv)
    write_csv(args.out_csv, x_label, x_values, y_values)
    plot_results(args, x_label, x_values, y_values)


if __name__ == "__main__":
    main()
