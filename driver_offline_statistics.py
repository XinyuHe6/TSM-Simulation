#!/usr/bin/env python3

from function_offline_statistics import (
    generate_random_graph,
    random_probability_vector,
    simulate_many_runs_offline_statistics,
)

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()

    # ===== params you want to control from shell =====
    parser.add_argument("--A", type=int, required=True)
    parser.add_argument("--I", type=int, required=True)

    parser.add_argument("--T_start", type=int, default=0)
    parser.add_argument("--T_end", type=int, default=100)
    parser.add_argument("--T_step", type=int, default=5)

    parser.add_argument("--edge_points", type=int, default=21)  # linspace(0,1,edge_points)

    parser.add_argument("--num_graphs_per_cell", type=int, default=5)
    parser.add_argument("--runs_per_graph", type=int, default=3)

    # ===== offline-statistics specific =====
    parser.add_argument("--mc_trials", type=int, default=200)
    parser.add_argument("--use_poisson_len", action="store_true")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out_csv", type=str, default="results_offline_statistics.csv")
    parser.add_argument("--out_fig", type=str, default="surface_offline_statistics.png")

    args = parser.parse_args()

    np.random.seed(args.seed)

    A_size = args.A
    I_size = args.I

    Ts = np.arange(args.T_start, args.T_end + 1, args.T_step)
    edge_probs = np.linspace(0, 1, args.edge_points)

    T_grid, EP_grid = np.meshgrid(Ts, edge_probs, indexing="ij")
    Z = np.zeros_like(T_grid, dtype=float)

    total_cells = len(Ts) * len(edge_probs)
    pbar_cells = tqdm(total=total_cells, desc="Grid (T, edge_prob)", unit="cell")

    for i_T, T in enumerate(Ts):
        for j_ep, edge_prob in enumerate(edge_probs):

            if T == 0:
                Z[i_T, j_ep] = 0.0
                pbar_cells.update(1)
                continue

            ratios = []
            for g in range(args.num_graphs_per_cell):
                # make graph + p reproducible per cell
                np.random.seed(args.seed + 100000 * i_T + 1000 * j_ep + g)
                # also seed python random inside simulate function via seed argument

                neighbors = generate_random_graph(A_size, I_size, float(edge_prob))
                p = random_probability_vector(I_size)

                avg_ratio_alg2, _ = simulate_many_runs_offline_statistics(
                    A_size, I_size, neighbors, p, int(T),
                    mc_trials=args.mc_trials,
                    num_runs=args.runs_per_graph,
                    seed=args.seed + 100000 * i_T + 1000 * j_ep + g,
                    use_poisson_len=args.use_poisson_len
                )
                ratios.append(float(avg_ratio_alg2))

            Z[i_T, j_ep] = float(np.mean(ratios)) if ratios else 0.0
            pbar_cells.update(1)

    pbar_cells.close()

    # ===== write CSV =====
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write("T,edge_prob,mean_ratio\n")
        for i_T, T in enumerate(Ts):
            for j_ep, edge_prob in enumerate(edge_probs):
                f.write(f"{int(T)},{float(edge_prob):.10f},{float(Z[i_T, j_ep]):.10f}\n")

    # ===== plot from CSV =====
    data = np.genfromtxt(args.out_csv, delimiter=",", skip_header=1)
    Ts2 = np.unique(data[:, 0]).astype(int)
    edge_probs2 = np.unique(data[:, 1])

    T_grid2, EP_grid2 = np.meshgrid(Ts2, edge_probs2, indexing="ij")
    Z2 = np.zeros_like(T_grid2, dtype=float)

    for row in data:
        T, ep, mr = int(row[0]), float(row[1]), float(row[2])
        i_T = np.where(Ts2 == T)[0][0]
        j_ep = np.where(edge_probs2 == ep)[0][0]
        Z2[i_T, j_ep] = mr

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        EP_grid2, T_grid2, Z2,
        rstride=1, cstride=1,
        linewidth=0, antialiased=True
    )

    ax.set_xlabel("edge_prob")
    ax.set_ylabel("T (horizon)")
    ax.set_zlabel("ALG / OPT ratio (Offline Statistics Alg2)")
    ax.set_title(f"Offline Statistics Alg2 vs T and edge_prob (A={A_size}, I={I_size})")
    ax.set_zlim(bottom=0)

    fig.colorbar(surf, shrink=0.5, aspect=10, label="ALG / OPT ratio")
    plt.tight_layout()

    plt.savefig(args.out_fig, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
