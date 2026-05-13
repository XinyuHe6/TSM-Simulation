#!/usr/bin/env python3

from function_Correlated_Sampling import (
    generate_graph,
    random_probability_vector,
    simulate_many_runs_correlated_sampling,
)

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm.auto import tqdm


def build_graph_params(args, A_size, I_size):
    if args.graph_mode == "random":
        graph_params = np.linspace(0, 1, args.edge_points, dtype=float)
        axis_label = "edge_prob"
        title_suffix = "edge_prob"
        csv_label = "edge_prob"
    elif args.graph_mode == "k_regular":
        if args.regular_degree_step <= 0:
            raise ValueError("--regular_degree_step must be positive.")
        if args.regular_degree_end < args.regular_degree_start:
            raise ValueError("--regular_degree_end must be >= --regular_degree_start.")

        graph_params = np.arange(
            args.regular_degree_start,
            args.regular_degree_end + 1,
            args.regular_degree_step,
            dtype=int,
        )
        if len(graph_params) == 0:
            raise ValueError("k-regular mode produced an empty degree grid.")
        if graph_params[-1] > A_size:
            raise ValueError(f"k-regular degree cannot exceed A (= {A_size}).")

        axis_label = "regular_degree"
        title_suffix = "regular_degree"
        csv_label = "regular_degree"
    else:
        raise ValueError(f"Unsupported graph mode: {args.graph_mode}")

    return graph_params, axis_label, title_suffix, csv_label


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--A", type=int, required=True)
    parser.add_argument("--I", type=int, required=True)

    parser.add_argument("--T_start", type=int, default=0)
    parser.add_argument("--T_end", type=int, default=100)
    parser.add_argument("--T_step", type=int, default=5)

    parser.add_argument("--edge_points", type=int, default=21)
    parser.add_argument("--graph_mode", type=str, choices=["random", "k_regular"], default="random")
    parser.add_argument("--regular_degree_start", type=int, default=1)
    parser.add_argument("--regular_degree_end", type=int, default=1)
    parser.add_argument("--regular_degree_step", type=int, default=1)

    parser.add_argument("--num_graphs_per_cell", type=int, default=3)
    parser.add_argument("--runs_per_graph", type=int, default=5)

    parser.add_argument("--use_poisson_len", action="store_true")
    parser.add_argument("--lp_max_rounds", type=int, default=200)
    parser.add_argument("--lp_separation_tol", type=float, default=1e-9)
    parser.add_argument(
        "--lp_constraint_mode",
        type=str,
        choices=["natural", "pair_approx"],
        default="natural",
    )
    parser.add_argument("--lp_pair_cap", type=float, default=None)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out_csv", type=str, default="results_correlated_sampling.csv")
    parser.add_argument("--out_fig", type=str, default="surface_correlated_sampling.png")

    args = parser.parse_args()

    np.random.seed(args.seed)

    A_size = args.A
    I_size = args.I

    Ts = np.arange(args.T_start, args.T_end + 1, args.T_step)
    graph_params, x_label, title_suffix, csv_label = build_graph_params(args, A_size, I_size)

    T_grid, GP_grid = np.meshgrid(Ts, graph_params, indexing="ij")
    Z = np.zeros_like(T_grid, dtype=float)

    total_cells = len(Ts) * len(graph_params)
    pbar_cells = tqdm(total=total_cells, desc=f"Grid (T, {x_label})", unit="cell")

    for i_T, T in enumerate(Ts):
        for j_gp, graph_param in enumerate(graph_params):
            if T == 0:
                Z[i_T, j_gp] = 0.0
                pbar_cells.update(1)
                continue

            ratios = []
            for g in range(args.num_graphs_per_cell):
                cell_seed = args.seed + 100000 * i_T + 1000 * j_gp + g
                np.random.seed(cell_seed)
                random.seed(cell_seed)

                neighbors = generate_graph(A_size, I_size, args.graph_mode, graph_param)
                p = random_probability_vector(I_size)

                avg_ratio_corr, _ = simulate_many_runs_correlated_sampling(
                    A_size=A_size,
                    I_size=I_size,
                    neighbors=neighbors,
                    p=p,
                    T=int(T),
                    num_runs=args.runs_per_graph,
                    seed=cell_seed,
                    use_poisson_len=args.use_poisson_len,
                    lp_max_rounds=args.lp_max_rounds,
                    lp_separation_tol=args.lp_separation_tol,
                    lp_constraint_mode=args.lp_constraint_mode,
                    lp_pair_cap=args.lp_pair_cap,
                )
                ratios.append(float(avg_ratio_corr))

            Z[i_T, j_gp] = float(np.mean(ratios)) if ratios else 0.0
            pbar_cells.update(1)

    pbar_cells.close()

    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write(f"T,{csv_label},mean_ratio\n")
        for i_T, T in enumerate(Ts):
            for j_gp, graph_param in enumerate(graph_params):
                f.write(f"{int(T)},{float(graph_param):.10f},{float(Z[i_T, j_gp]):.10f}\n")

    data = np.genfromtxt(args.out_csv, delimiter=",", skip_header=1)
    Ts2 = np.unique(data[:, 0]).astype(int)
    graph_params2 = np.unique(data[:, 1])

    T_grid2, GP_grid2 = np.meshgrid(Ts2, graph_params2, indexing="ij")
    Z2 = np.zeros_like(T_grid2, dtype=float)

    for row in data:
        T, graph_param, mr = int(row[0]), float(row[1]), float(row[2])
        i_T = np.where(Ts2 == T)[0][0]
        j_gp = np.where(graph_params2 == graph_param)[0][0]
        Z2[i_T, j_gp] = mr

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        GP_grid2,
        T_grid2,
        Z2,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("T (horizon)")
    ax.set_zlabel("ALG / OPT ratio (Correlated Sampling)")
    ax.set_title(
        f"Correlated Sampling ({args.lp_constraint_mode}) vs T and {title_suffix} "
        f"(A={A_size}, I={I_size})"
    )
    ax.set_zlim(bottom=0)

    fig.colorbar(surf, shrink=0.5, aspect=10, label="ALG / OPT ratio")
    plt.tight_layout()

    plt.savefig(args.out_fig, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
