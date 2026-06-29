#!/usr/bin/env python3

import argparse

from common import (
    add_common_run_args,
    add_graph_mode_sweep_args,
    build_graph_mode_sweep_values,
    evaluate_graph_parameter_point,
    graph_params_from_mode,
    parse_algorithms,
    plot_lines,
    tqdm,
    write_wide_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correlated Sampling curve over edge_prob or k-regular degree with A = I = T = N."
    )
    parser.add_argument("--N", type=int, required=True)
    add_graph_mode_sweep_args(parser)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_fig", type=str, required=True)
    add_common_run_args(parser, default_algorithms="correlated_sampling")
    args = parser.parse_args()
    if args.N <= 0:
        raise ValueError("--N must be positive.")
    args.A = args.N
    args.I = args.N
    args.T = args.N
    args.algorithms = parse_algorithms(args.algorithms, "correlated_sampling")
    if len(args.algorithms) != 1:
        raise ValueError("This script expects exactly one algorithm.")
    return args


def main():
    args = parse_args()
    x_values, x_label = build_graph_mode_sweep_values(args, args.N)
    algorithm = args.algorithms[0]
    values_by_algorithm = {algorithm: []}

    for point_index, graph_param in enumerate(tqdm(x_values, desc=f"Sweep {x_label}", unit="point")):
        graph_name, graph_params = graph_params_from_mode(args.graph_mode, graph_param)
        values = evaluate_graph_parameter_point(
            args=args,
            A_size=args.N,
            I_size=args.N,
            T=args.N,
            graph_name=graph_name,
            graph_params=graph_params,
            point_index=point_index,
            algorithms=[algorithm],
            metric="ratio",
        )
        values_by_algorithm[algorithm].append(values[algorithm])

    write_wide_csv(args.out_csv, x_label, x_values, values_by_algorithm, [algorithm])
    plot_lines(
        args.out_fig,
        x_label,
        "ALG / OPT ratio",
        f"Correlated Sampling vs {x_label} (A=I=T={args.N})",
        x_values,
        values_by_algorithm,
        [algorithm],
        show=args.show,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
