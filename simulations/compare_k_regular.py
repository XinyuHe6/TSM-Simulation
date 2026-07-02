#!/usr/bin/env python3

import argparse

from .common import (
    DEFAULT_RATIO_ALGORITHMS,
    add_common_run_args,
    evaluate_graph_parameter_point,
    int_range,
    parse_algorithms,
    plot_lines,
    tqdm,
    validate_common_args,
    write_wide_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare matching algorithms over k-regular graph degree."
    )
    parser.add_argument("--A", type=int, default=100)
    parser.add_argument("--I", type=int, default=100)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--degree_start", type=int, default=0)
    parser.add_argument("--degree_end", type=int, default=None)
    parser.add_argument("--degree_step", type=int, default=1)
    parser.add_argument("--out_csv", type=str, default="k_regular_comparison.csv")
    parser.add_argument("--out_fig", type=str, default="k_regular_comparison.png")
    add_common_run_args(parser, default_algorithms=DEFAULT_RATIO_ALGORITHMS)
    args = parser.parse_args()
    if args.degree_end is None:
        args.degree_end = args.A
    validate_common_args(args)
    if args.degree_end > args.A:
        raise ValueError("--degree_end cannot exceed A.")
    args.algorithms = parse_algorithms(args.algorithms, DEFAULT_RATIO_ALGORITHMS)
    return args


def main():
    args = parse_args()
    degrees = int_range(args.degree_start, args.degree_end, args.degree_step)
    values_by_algorithm = {algorithm: [] for algorithm in args.algorithms}

    for degree_index, degree in enumerate(tqdm(degrees, desc="Sweep degree", unit="point")):
        values = evaluate_graph_parameter_point(
            args=args,
            A_size=args.A,
            I_size=args.I,
            T=args.T,
            graph_name="k_regular",
            graph_params={"degree": degree},
            point_index=degree_index,
            algorithms=args.algorithms,
            metric="ratio",
        )
        for algorithm in args.algorithms:
            values_by_algorithm[algorithm].append(values[algorithm])

    write_wide_csv(args.out_csv, "degree", degrees, values_by_algorithm, args.algorithms)
    plot_lines(
        args.out_fig,
        "regular_degree",
        "ALG / OPT ratio",
        f"ALG / OPT vs k-regular degree (A={args.A}, I={args.I}, T={args.T})",
        degrees,
        values_by_algorithm,
        args.algorithms,
        show=args.show,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
