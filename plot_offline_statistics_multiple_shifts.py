#!/usr/bin/env python3

import argparse
import os
import random

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from function_offline_statistics import (
    build_virtual_copies,
    compute_opt_from_realization,
    estimate_f_monte_carlo,
    generate_random_graph,
    pick_from_partition,
    random_probability_vector,
    sample_arrival_sequence,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare offline statistics matching with multiple shifted attempts "
            "over edge probability."
        )
    )

    parser.add_argument("--A", type=int, default=100)
    parser.add_argument("--I", type=int, default=100)
    parser.add_argument("--T", type=int, default=100)

    parser.add_argument("--edge_points", type=int, default=21)
    parser.add_argument("--num_graphs_per_point", type=int, default=5)
    parser.add_argument("--runs_per_graph", type=int, default=3)
    parser.add_argument("--mc_trials", type=int, default=20)
    parser.add_argument(
        "--tries",
        type=str,
        default="2,3,4,5,6,7,8,9,10",
        help="Comma-separated try counts to compare, e.g. 2,3,4,5,6,7,8,9,10.",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_fig", type=str, required=True)
    parser.add_argument("--show", action="store_true")

    parser.set_defaults(use_poisson_len=True)
    parser.add_argument(
        "--use_poisson_len",
        dest="use_poisson_len",
        action="store_true",
        help="Use Poisson(T) arrival length.",
    )
    parser.add_argument(
        "--no_use_poisson_len",
        dest="use_poisson_len",
        action="store_false",
        help="Use exactly T arrivals.",
    )

    return parser.parse_args()


def parse_try_counts(raw_tries):
    try_counts = []
    for part in raw_tries.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("--tries values must be positive integers.")
        try_counts.append(value)

    if not try_counts:
        raise ValueError("--tries must contain at least one positive integer.")

    return sorted(set(try_counts))


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def linspace(start, stop, num):
    if num <= 0:
        raise ValueError("--edge_points must be positive.")
    if num == 1:
        return [float(start)]

    step = (stop - start) / (num - 1)
    return [float(start + step * idx) for idx in range(num)]


def mean(values):
    return (sum(values) / len(values)) if values else 0.0


def _build_shifted_partition(items, f_dummy, r, shift):
    """
    Build one shifted partition.

    shift = 0 gives Iy:
      a1, a2, ..., dummy
    shift = 1 gives Jy:
      a2, a3, ..., dummy, a1
    shift = 2 gives the third-attempt partition:
      a3, a4, ..., dummy, a1, a2
    """
    if not items:
        return [None], [r]

    n = len(items)
    shift = max(0, min(int(shift), n))

    prefix = items[shift:]
    suffix = items[:shift]

    bins = []
    ends = []
    cur = 0.0

    for a, val in prefix:
        cur += val
        bins.append(a)
        ends.append(cur)

    cur += f_dummy
    bins.append(None)
    ends.append(cur)

    for a, val in suffix:
        cur += val
        bins.append(a)
        ends.append(cur)

    if ends:
        ends[-1] = r
    return bins, ends


def build_shifted_partitions_for_copy(
    j,
    neighbors,
    orig_type_of_copy,
    r_copy,
    f_dict,
    max_tries,
):
    i = orig_type_of_copy[j]
    r = r_copy[j]
    nbrs = neighbors[i]

    items = [(a, f_dict.get((j, a), 0.0)) for a in nbrs]
    items.sort(key=lambda item: item[1], reverse=True)

    sum_f = sum(val for _, val in items)
    if sum_f > r and sum_f > 0:
        scale = r / sum_f
        items = [(a, val * scale) for a, val in items]
        sum_f = r

    f_dummy = max(0.0, r - sum_f)
    base_bins, base_ends = _build_shifted_partition(items, f_dummy, r, shift=0)

    if not items or items[0][1] <= 0:
        return [(base_bins[:], base_ends[:]) for _ in range(max_tries)]

    return [
        _build_shifted_partition(items, f_dummy, r, shift=shift)
        for shift in range(max_tries)
    ]


def build_all_shifted_partitions(
    neighbors,
    r_copy_of_exp,
    orig_type_of_exp,
    f_dict,
    max_tries,
):
    bins_by_try = [[] for _ in range(max_tries)]
    ends_by_try = [[] for _ in range(max_tries)]

    for j in range(len(r_copy_of_exp)):
        partitions = build_shifted_partitions_for_copy(
            j=j,
            neighbors=neighbors,
            orig_type_of_copy=orig_type_of_exp,
            r_copy=r_copy_of_exp,
            f_dict=f_dict,
            max_tries=max_tries,
        )
        for try_index, (bins, ends) in enumerate(partitions):
            bins_by_try[try_index].append(bins)
            ends_by_try[try_index].append(ends)

    return bins_by_try, ends_by_try


def simulate_offline_statistics_k_once(
    A_size,
    I_size,
    neighbors,
    p,
    T,
    r_copy_of_exp,
    copies_of_type,
    bins_by_try,
    ends_by_try,
    try_count,
    use_poisson_len=False,
):
    matched_A = [False] * A_size
    ALG = 0

    hat_I = []
    edges_real = []

    arrivals = sample_arrival_sequence(I_size, p, T, use_poisson_len=use_poisson_len)

    for i in arrivals:
        imp_id = len(hat_I)
        hat_I.append(imp_id)

        for a in neighbors[i]:
            edges_real.append((a, imp_id))

        if copies_of_type[i]:
            j = random.choice(copies_of_type[i])
        else:
            continue

        r = r_copy_of_exp[j]
        if r <= 0:
            continue

        x = random.random() * r

        for try_index in range(try_count):
            a = pick_from_partition(
                bins_by_try[try_index][j],
                ends_by_try[try_index][j],
                x,
            )
            if a is not None and (not matched_A[a]):
                matched_A[a] = True
                ALG += 1
                break

    OPT = compute_opt_from_realization(A_size, hat_I, edges_real)
    return ALG, OPT


def evaluate_one_graph(args, try_counts, neighbors, p, graph_seed):
    max_tries = max(try_counts)

    r_copy_of_exp, copies_of_type, f_dict = estimate_f_monte_carlo(
        args.A,
        args.I,
        neighbors,
        p,
        args.T,
        mc_trials=args.mc_trials,
        seed=graph_seed,
        use_poisson_len=args.use_poisson_len,
    )

    r_copy_of_exp2, orig_type_of_exp, copies_of_type2 = build_virtual_copies(
        args.T,
        p,
    )
    if len(r_copy_of_exp2) != len(r_copy_of_exp):
        r_copy_of_exp = r_copy_of_exp2
    if copies_of_type2 is not None:
        copies_of_type = copies_of_type2

    bins_by_try, ends_by_try = build_all_shifted_partitions(
        neighbors=neighbors,
        r_copy_of_exp=r_copy_of_exp,
        orig_type_of_exp=orig_type_of_exp,
        f_dict=f_dict,
        max_tries=max_tries,
    )

    graph_values = {}
    for try_count in try_counts:
        ratios = []
        random.seed(graph_seed + 10000)
        for _ in range(args.runs_per_graph):
            ALG, OPT = simulate_offline_statistics_k_once(
                args.A,
                args.I,
                neighbors,
                p,
                args.T,
                r_copy_of_exp,
                copies_of_type,
                bins_by_try,
                ends_by_try,
                try_count,
                use_poisson_len=args.use_poisson_len,
            )
            if OPT > 0:
                ratios.append(ALG / OPT)
        graph_values[try_count] = float(mean(ratios))

    return graph_values


def evaluate_one_edge_prob(edge_prob, edge_index, args, try_counts):
    ratios_by_try = {try_count: [] for try_count in try_counts}

    for graph_index in range(args.num_graphs_per_point):
        cell_seed = args.seed + 100000 * edge_index + graph_index
        random.seed(cell_seed)

        neighbors = generate_random_graph(args.A, args.I, float(edge_prob))
        p = random_probability_vector(args.I)

        graph_values = evaluate_one_graph(
            args=args,
            try_counts=try_counts,
            neighbors=neighbors,
            p=p,
            graph_seed=cell_seed + 11,
        )

        for try_count in try_counts:
            ratios_by_try[try_count].append(graph_values[try_count])

    return {
        try_count: float(mean(values))
        for try_count, values in ratios_by_try.items()
    }


def csv_column_name(try_count):
    return f"offline_statistics_{try_count}try"


def write_csv(path, edge_probs, values_by_try, try_counts):
    with open(path, "w", encoding="utf-8") as f:
        header = ["edge_prob"] + [csv_column_name(k) for k in try_counts]
        f.write(",".join(header) + "\n")
        for idx, edge_prob in enumerate(edge_probs):
            row = [f"{float(edge_prob):.10f}"]
            row.extend(f"{float(values_by_try[k][idx]):.10f}" for k in try_counts)
            f.write(",".join(row) + "\n")


def plot_results(args, edge_probs, values_by_try, try_counts):
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = ["o", "^", "s", "D", "v", "P", "X", "*"]

    ensure_parent_dir(args.out_fig)
    plt.figure(figsize=(10, 6))
    for idx, try_count in enumerate(try_counts):
        marker = markers[idx % len(markers)]
        plt.plot(
            edge_probs,
            values_by_try[try_count],
            marker=marker,
            linewidth=2,
            label=f"Offline Statistics ({try_count} tries)",
        )

    plt.xlabel("edge_prob")
    plt.ylabel("ALG / OPT ratio")
    plt.title(
        f"Offline Statistics Multiple Shifts (A={args.A}, I={args.I}, T={args.T})"
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
    try_counts = parse_try_counts(args.tries)
    edge_probs = linspace(0.0, 1.0, args.edge_points)

    values_by_try = {try_count: [] for try_count in try_counts}

    for edge_index, edge_prob in enumerate(
        tqdm(edge_probs, desc="Sweep edge_prob", unit="point")
    ):
        values = evaluate_one_edge_prob(edge_prob, edge_index, args, try_counts)
        for try_count in try_counts:
            values_by_try[try_count].append(values[try_count])

    ensure_parent_dir(args.out_csv)
    write_csv(args.out_csv, edge_probs, values_by_try, try_counts)
    plot_results(args, edge_probs, values_by_try, try_counts)


if __name__ == "__main__":
    main()
