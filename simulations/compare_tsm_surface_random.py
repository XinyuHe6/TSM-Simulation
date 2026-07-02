#!/usr/bin/env python3

import argparse

from .common import (
    add_common_run_args,
    evaluate_graph_parameter_point,
    int_range,
    linspace,
    parse_algorithms,
    plot_surface,
    tqdm,
    validate_common_args,
    write_grid_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TSM surface over T and random graph edge probability."
    )
    parser.add_argument("--A", type=int, required=True)
    parser.add_argument("--I", type=int, required=True)
    parser.add_argument("--T_start", type=int, default=0)
    parser.add_argument("--T_end", type=int, default=100)
    parser.add_argument("--T_step", type=int, default=5)
    parser.add_argument("--edge_points", type=int, default=21)
    parser.add_argument("--out_csv", type=str, default="tsm_surface_random.csv")
    parser.add_argument("--out_fig", type=str, default="tsm_surface_random.png")
    add_common_run_args(parser, default_algorithms="tsm")
    args = parser.parse_args()
    args.T = args.T_end
    validate_common_args(args)
    args.algorithms = parse_algorithms(args.algorithms, "tsm")
    if len(args.algorithms) != 1:
        raise ValueError("This surface script expects exactly one algorithm.")
    return args


def main():
    args = parse_args()
    Ts = int_range(args.T_start, args.T_end, args.T_step)
    edge_probs = linspace(0.0, 1.0, args.edge_points)
    algorithm = args.algorithms[0]
    grid = []

    total = len(Ts) * len(edge_probs)
    pbar = tqdm(total=total, desc="Grid (T, edge_prob)", unit="cell")
    for t_index, T in enumerate(Ts):
        row = []
        for edge_index, edge_prob in enumerate(edge_probs):
            if T == 0:
                row.append(0.0)
            else:
                values = evaluate_graph_parameter_point(
                    args=args,
                    A_size=args.A,
                    I_size=args.I,
                    T=T,
                    graph_name="random",
                    graph_params={"edge_prob": edge_prob},
                    point_index=t_index * len(edge_probs) + edge_index,
                    algorithms=[algorithm],
                    metric="ratio",
                )
                row.append(values[algorithm])
            pbar.update(1)
        grid.append(row)
    pbar.close()

    write_grid_csv(args.out_csv, "T", "edge_prob", Ts, edge_probs, grid)
    plot_surface(
        args.out_fig,
        "edge_prob",
        "T",
        "ALG / OPT ratio",
        f"TSM performance vs T and edge_prob (A={args.A}, I={args.I})",
        edge_probs,
        Ts,
        grid,
        show=args.show,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
