#!/usr/bin/env python3

import argparse

from common import (
    DEFAULT_RATIO_ALGORITHMS,
    add_common_run_args,
    evaluate_graph_parameter_point,
    linspace,
    parse_algorithms,
    plot_lines,
    tqdm,
    validate_common_args,
    write_wide_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare matching algorithms over random graph edge probability."
    )
    parser.add_argument("--A", type=int, default=100)
    parser.add_argument("--I", type=int, default=100)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--edge_points", type=int, default=21)
    parser.add_argument("--out_csv", type=str, default="random_edge_prob_comparison.csv")
    parser.add_argument("--out_fig", type=str, default="random_edge_prob_comparison.png")
    add_common_run_args(parser, default_algorithms=DEFAULT_RATIO_ALGORITHMS)
    args = parser.parse_args()
    validate_common_args(args)
    args.algorithms = parse_algorithms(args.algorithms, DEFAULT_RATIO_ALGORITHMS)
    return args


def main():
    args = parse_args()
    edge_probs = linspace(0.0, 1.0, args.edge_points)
    values_by_algorithm = {algorithm: [] for algorithm in args.algorithms}

    for edge_index, edge_prob in enumerate(tqdm(edge_probs, desc="Sweep edge_prob", unit="point")):
        values = evaluate_graph_parameter_point(
            args=args,
            A_size=args.A,
            I_size=args.I,
            T=args.T,
            graph_name="random",
            graph_params={"edge_prob": edge_prob},
            point_index=edge_index,
            algorithms=args.algorithms,
            metric="ratio",
        )
        for algorithm in args.algorithms:
            values_by_algorithm[algorithm].append(values[algorithm])

    write_wide_csv(args.out_csv, "edge_prob", edge_probs, values_by_algorithm, args.algorithms)
    plot_lines(
        args.out_fig,
        "edge_prob",
        "ALG / OPT ratio",
        f"ALG / OPT vs edge_prob (A={args.A}, I={args.I}, T={args.T})",
        edge_probs,
        values_by_algorithm,
        args.algorithms,
        show=args.show,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
