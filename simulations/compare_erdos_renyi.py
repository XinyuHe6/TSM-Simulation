#!/usr/bin/env python3

import argparse

from common import (
    DEFAULT_RATIO_ALGORITHMS,
    add_common_run_args,
    evaluate_graph_parameter_point,
    float_range,
    parse_algorithms,
    plot_lines,
    tqdm,
    write_wide_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare matching algorithms over Erdos-Renyi G(n,n,p=k/n)."
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--k_start", type=float, default=0.0)
    parser.add_argument("--k_end", type=float, default=100.0)
    parser.add_argument("--k_step", type=float, default=1.0)
    parser.add_argument("--out_csv", type=str, default="erdos_renyi_comparison.csv")
    parser.add_argument("--out_fig", type=str, default="erdos_renyi_comparison.png")
    add_common_run_args(
        parser,
        default_algorithms="tsm,correlated_sampling,manshadi2,manshadi3,manshadi4,random_matching,degree_matching",
    )
    args = parser.parse_args()
    if args.n <= 0:
        raise ValueError("--n must be positive.")
    if args.T is None:
        args.T = args.n
    if args.T < 0:
        raise ValueError("--T must be non-negative.")
    if args.num_graphs_per_point <= 0:
        raise ValueError("--num_graphs_per_point must be positive.")
    if args.runs_per_graph <= 0:
        raise ValueError("--runs_per_graph must be positive.")
    if args.k_end > args.n:
        raise ValueError("--k_end cannot exceed n because edge_prob = k / n.")
    args.algorithms = parse_algorithms(
        args.algorithms,
        "tsm,correlated_sampling,manshadi2,manshadi3,manshadi4,random_matching,degree_matching",
    )
    return args


def main():
    args = parse_args()
    k_values = float_range(args.k_start, args.k_end, args.k_step)
    edge_probs = [float(k) / float(args.n) for k in k_values]
    values_by_algorithm = {algorithm: [] for algorithm in args.algorithms}

    for k_index, k_value in enumerate(tqdm(k_values, desc="Sweep k", unit="point")):
        values = evaluate_graph_parameter_point(
            args=args,
            A_size=args.n,
            I_size=args.n,
            T=args.T,
            graph_name="erdos_renyi",
            graph_params={"k": k_value},
            point_index=k_index,
            algorithms=args.algorithms,
            metric="ratio",
        )
        for algorithm in args.algorithms:
            values_by_algorithm[algorithm].append(values[algorithm])

    write_wide_csv(
        args.out_csv,
        "k",
        k_values,
        values_by_algorithm,
        args.algorithms,
        extra_columns={"edge_prob": edge_probs},
    )
    plot_lines(
        args.out_fig,
        "k (edge probability p = k / n)",
        "ALG / OPT ratio",
        f"ALG / OPT on Erdos-Renyi G(n,n,k/n) (n={args.n}, T={args.T})",
        k_values,
        values_by_algorithm,
        args.algorithms,
        show=args.show,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
