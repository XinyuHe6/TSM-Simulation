#!/usr/bin/env python3

import argparse

from .common import (
    add_common_run_args,
    add_graph_mode_sweep_args,
    build_graph_mode_sweep_values,
    evaluate_graph_parameter_point,
    graph_params_from_mode,
    int_range,
    parse_algorithms,
    plot_surface,
    tqdm,
    validate_common_args,
    write_grid_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manshadi/offline-statistics surface over T and graph parameter."
    )
    parser.add_argument("--A", type=int, required=True)
    parser.add_argument("--I", type=int, required=True)
    parser.add_argument("--T_start", type=int, default=0)
    parser.add_argument("--T_end", type=int, default=100)
    parser.add_argument("--T_step", type=int, default=5)
    add_graph_mode_sweep_args(parser)
    parser.add_argument("--out_csv", type=str, default="manshadi_surface.csv")
    parser.add_argument("--out_fig", type=str, default="manshadi_surface.png")
    add_common_run_args(parser, default_algorithms="manshadi2")
    args = parser.parse_args()
    args.T = args.T_end
    validate_common_args(args)
    args.algorithms = parse_algorithms(args.algorithms, "manshadi2")
    if len(args.algorithms) != 1:
        raise ValueError("This surface script expects exactly one algorithm.")
    return args


def main():
    args = parse_args()
    Ts = int_range(args.T_start, args.T_end, args.T_step)
    x_values, x_label = build_graph_mode_sweep_values(args, args.A)
    algorithm = args.algorithms[0]
    grid = []

    total = len(Ts) * len(x_values)
    pbar = tqdm(total=total, desc=f"Grid (T, {x_label})", unit="cell")
    for t_index, T in enumerate(Ts):
        row = []
        for x_index, graph_param in enumerate(x_values):
            if T == 0:
                row.append(0.0)
            else:
                graph_name, graph_params = graph_params_from_mode(args.graph_mode, graph_param)
                values = evaluate_graph_parameter_point(
                    args=args,
                    A_size=args.A,
                    I_size=args.I,
                    T=T,
                    graph_name=graph_name,
                    graph_params=graph_params,
                    point_index=t_index * len(x_values) + x_index,
                    algorithms=[algorithm],
                    metric="ratio",
                )
                row.append(values[algorithm])
            pbar.update(1)
        grid.append(row)
    pbar.close()

    write_grid_csv(args.out_csv, "T", x_label, Ts, x_values, grid)
    plot_surface(
        args.out_fig,
        x_label,
        "T",
        "ALG / OPT ratio",
        f"{algorithm} performance vs T and {x_label} (A={args.A}, I={args.I})",
        x_values,
        Ts,
        grid,
        show=args.show,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
