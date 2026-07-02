import csv
import math
import os
import random

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from arrival_algorithms import (
    arrival_probabilities_from_counts,
    random_arrival_counts,
    sample_arrival_sequence,
)
from graph_algorithms import generate_graph
from matching_algorithms import (
    MATCHING_SPECS,
    canonicalize_matching_algorithms,
    compute_offline_opt,
    prepare_matching_states,
    run_matching,
)


DEFAULT_RATIO_ALGORITHMS = (
    "random_matching,degree_matching,tsm,manshadi2,manshadi3,manshadi4"
)
DEFAULT_ALG_COUNT_ALGORITHMS = (
    "fluid_lp,random_matching,degree_matching,tsm,manshadi2,manshadi3,manshadi4"
)


def parse_algorithms(raw_algorithms, default_algorithms=DEFAULT_RATIO_ALGORITHMS):
    if raw_algorithms.strip().lower() == "all":
        raw_algorithms = ",".join(MATCHING_SPECS)
    elif raw_algorithms.strip().lower() == "default":
        raw_algorithms = default_algorithms
    return canonicalize_matching_algorithms(raw_algorithms)


def arrival_algorithm_from_args(args):
    return "poisson" if getattr(args, "use_poisson_len", False) else "fixed"


def add_common_run_args(parser, default_algorithms=DEFAULT_RATIO_ALGORITHMS):
    parser.add_argument(
        "--num_graphs_per_point",
        type=int,
        default=5,
        help="Independent random graphs evaluated at each x-axis/grid point.",
    )
    parser.add_argument(
        "--runs_per_graph",
        type=int,
        default=3,
        help="Independent arrival realizations evaluated on each graph.",
    )
    parser.add_argument(
        "--mc_trials",
        type=int,
        default=20,
        help="Offline Monte Carlo trials used to prepare Manshadi algorithms.",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=default_algorithms,
        help=(
            "Comma-separated matching algorithms. Use 'all' for every registered "
            "algorithm, or 'default' for this script's default set."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.set_defaults(use_poisson_len=False)
    parser.add_argument(
        "--use_poisson_len",
        dest="use_poisson_len",
        action="store_true",
        help="Sample the realized arrival length from Poisson(T).",
    )
    parser.add_argument(
        "--no_use_poisson_len",
        dest="use_poisson_len",
        action="store_false",
        help="Use exactly T realized arrivals (the default).",
    )
    parser.add_argument(
        "--corr_lp_constraint_mode",
        choices=["natural", "pair_approx"],
        default="pair_approx",
        help="Correlated Sampling LP: exact cutting-plane constraints or fast pair approximation.",
    )
    parser.add_argument(
        "--corr_lp_max_rounds",
        type=int,
        default=20,
        help="Maximum cutting-plane rounds for the natural Correlated Sampling LP.",
    )
    parser.add_argument(
        "--corr_lp_separation_tol",
        type=float,
        default=1e-9,
        help="Constraint-violation tolerance for the natural LP.",
    )
    parser.add_argument(
        "--corr_lp_pair_cap",
        type=float,
        default=None,
        help="Optional override for the pairwise LP constraint RHS.",
    )
    parser.add_argument(
        "--skip_failed_points",
        action="store_true",
        help="Write NaN and continue when an algorithm cannot prepare a point.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Write CSV only; do not create a figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure interactively after saving it.",
    )


def validate_common_args(args):
    if getattr(args, "A", 1) <= 0 or getattr(args, "I", 1) <= 0:
        raise ValueError("--A and --I must be positive.")
    if getattr(args, "T", 0) < 0:
        raise ValueError("--T must be non-negative.")
    if args.num_graphs_per_point <= 0:
        raise ValueError("--num_graphs_per_point must be positive.")
    if args.runs_per_graph <= 0:
        raise ValueError("--runs_per_graph must be positive.")
    if args.mc_trials <= 0:
        raise ValueError("--mc_trials must be positive.")
    if args.corr_lp_max_rounds <= 0:
        raise ValueError("--corr_lp_max_rounds must be positive.")
    if args.corr_lp_separation_tol < 0:
        raise ValueError("--corr_lp_separation_tol must be non-negative.")


def linspace(start, stop, num):
    if num <= 0:
        raise ValueError("number of points must be positive.")
    if num == 1:
        return [float(start)]
    step = (float(stop) - float(start)) / float(num - 1)
    return [float(start) + step * idx for idx in range(num)]


def float_range(start, end, step):
    if step <= 0:
        raise ValueError("step must be positive.")
    if end < start:
        raise ValueError("end must be >= start.")
    values = []
    cur = float(start)
    while cur <= float(end) + 1e-12:
        values.append(round(cur, 12))
        cur += float(step)
    return values


def int_range(start, end, step):
    if step <= 0:
        raise ValueError("step must be positive.")
    if end < start:
        raise ValueError("end must be >= start.")
    return list(range(int(start), int(end) + 1, int(step)))


def add_graph_mode_sweep_args(parser):
    parser.add_argument(
        "--graph_mode",
        choices=["random", "k_regular"],
        default="random",
        help="Sweep random edge probability or k-regular degree.",
    )
    parser.add_argument(
        "--edge_points",
        type=int,
        default=21,
        help="Number of evenly spaced edge probabilities in [0, 1].",
    )
    parser.add_argument("--regular_degree_start", type=int, default=0)
    parser.add_argument("--regular_degree_end", type=int, default=None)
    parser.add_argument("--regular_degree_step", type=int, default=1)


def build_graph_mode_sweep_values(args, A_size):
    if args.graph_mode == "random":
        return linspace(0.0, 1.0, args.edge_points), "edge_prob"

    regular_degree_end = args.regular_degree_end
    if regular_degree_end is None:
        regular_degree_end = A_size
    if regular_degree_end > A_size:
        raise ValueError("--regular_degree_end cannot exceed A.")
    values = int_range(
        args.regular_degree_start,
        regular_degree_end,
        args.regular_degree_step,
    )
    return values, "regular_degree"


def graph_params_from_mode(graph_mode, graph_param):
    if graph_mode == "random":
        return "random", {"edge_prob": graph_param}
    if graph_mode == "k_regular":
        return "k_regular", {"degree": int(graph_param)}
    raise ValueError(f"Unsupported graph_mode: {graph_mode}")


def mean(values, ignore_nan=True):
    if ignore_nan:
        values = [value for value in values if not math.isnan(value)]
    return (sum(values) / len(values)) if values else float("nan")


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def format_float(value):
    return "nan" if math.isnan(value) else f"{float(value):.10f}"


def metric_value(alg, opt, metric, zero_opt_ratio):
    if metric == "alg":
        return float(alg)
    if opt > 0:
        return float(alg) / float(opt)
    return float(zero_opt_ratio)


def evaluate_algorithms_on_graph(
    args,
    A_size,
    I_size,
    T,
    neighbors,
    e,
    graph_seed,
    algorithms,
    metric="ratio",
    zero_opt_ratio=float("nan"),
):
    p = arrival_probabilities_from_counts(e)
    T = sum(e)
    try:
        states = prepare_matching_states(
            algorithms=algorithms,
            A_size=A_size,
            I_size=I_size,
            neighbors=neighbors,
            p=p,
            T=T,
            e=e,
            seed=graph_seed + 1000,
            arrival_algorithm=arrival_algorithm_from_args(args),
            mc_trials=args.mc_trials,
            corr_lp_max_rounds=args.corr_lp_max_rounds,
            corr_lp_separation_tol=args.corr_lp_separation_tol,
            corr_lp_constraint_mode=args.corr_lp_constraint_mode,
            corr_lp_pair_cap=args.corr_lp_pair_cap,
        )
    except (ImportError, RuntimeError, ValueError):
        if not args.skip_failed_points:
            raise
        return {algorithm: float("nan") for algorithm in algorithms}

    values = {algorithm: [] for algorithm in algorithms}
    for run_index in range(args.runs_per_graph):
        arrival_seed = graph_seed + 1000000 + run_index
        random.seed(arrival_seed)
        arrivals = sample_arrival_sequence(
            arrival_algorithm_from_args(args),
            I_size=I_size,
            e=e,
        )
        opt = compute_offline_opt(A_size, neighbors, arrivals)

        for algorithm_index, algorithm in enumerate(algorithms):
            random.seed(arrival_seed + 100 + algorithm_index)
            result = run_matching(
                algorithm,
                A_size=A_size,
                I_size=I_size,
                neighbors=neighbors,
                arrivals=arrivals,
                state=states.get(algorithm),
            )
            values[algorithm].append(metric_value(result.alg, opt, metric, zero_opt_ratio))

    return {algorithm: mean(run_values) for algorithm, run_values in values.items()}


def evaluate_graph_parameter_point(
    args,
    A_size,
    I_size,
    T,
    graph_name,
    graph_params,
    point_index,
    algorithms,
    metric="ratio",
    zero_opt_ratio=float("nan"),
):
    values_by_algorithm = {algorithm: [] for algorithm in algorithms}

    for graph_index in range(args.num_graphs_per_point):
        cell_seed = args.seed + 100000 * point_index + graph_index
        random.seed(cell_seed)
        neighbors = generate_graph(
            graph_name,
            A_size=A_size,
            I_size=I_size,
            **graph_params,
        )
        e = random_arrival_counts(I_size, T)
        graph_values = evaluate_algorithms_on_graph(
            args=args,
            A_size=A_size,
            I_size=I_size,
            T=T,
            neighbors=neighbors,
            e=e,
            graph_seed=cell_seed + 17,
            algorithms=algorithms,
            metric=metric,
            zero_opt_ratio=zero_opt_ratio,
        )
        for algorithm in algorithms:
            values_by_algorithm[algorithm].append(graph_values[algorithm])

    return {
        algorithm: mean(values)
        for algorithm, values in values_by_algorithm.items()
    }


def write_wide_csv(path, x_label, x_values, values_by_algorithm, algorithms, extra_columns=None):
    ensure_parent_dir(path)
    extra_columns = extra_columns or {}
    header = [x_label] + list(extra_columns) + [
        MATCHING_SPECS[algorithm]["csv"] for algorithm in algorithms
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, x_value in enumerate(x_values):
            row = [format_float(float(x_value))]
            for values in extra_columns.values():
                row.append(format_float(float(values[idx])))
            for algorithm in algorithms:
                row.append(format_float(values_by_algorithm[algorithm][idx]))
            writer.writerow(row)


def write_grid_csv(path, row_label, col_label, rows, cols, grid):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row_label, col_label, "mean_ratio"])
        for row_index, row_value in enumerate(rows):
            for col_index, col_value in enumerate(cols):
                writer.writerow([
                    int(row_value) if float(row_value).is_integer() else format_float(row_value),
                    format_float(float(col_value)),
                    format_float(grid[row_index][col_index]),
                ])


def plot_lines(path, x_label, y_label, title, x_values, series_by_algorithm, algorithms, show=False, no_plot=False):
    if no_plot:
        return
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent_dir(path)
    plt.figure(figsize=(10, 6))
    for algorithm in algorithms:
        spec = MATCHING_SPECS[algorithm]
        plt.plot(
            x_values,
            series_by_algorithm[algorithm],
            marker=spec["marker"],
            linewidth=2,
            label=spec["label"],
        )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def plot_series(path, x_label, y_label, title, x_values, series, show=False, no_plot=False):
    if no_plot:
        return
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent_dir(path)
    plt.figure(figsize=(10, 6))
    for label, y_values in series.items():
        plt.plot(x_values, y_values, marker="o", linewidth=2, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def plot_surface(path, x_label, y_label, z_label, title, x_values, y_values, grid, show=False, no_plot=False):
    if no_plot:
        return
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_parent_dir(path)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    x_grid = [[x for x in x_values] for _ in y_values]
    y_grid = [[y for _ in x_values] for y in y_values]
    surf = ax.plot_surface(x_grid, y_grid, grid, linewidth=0, antialiased=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.set_zlim(bottom=0)
    fig.colorbar(surf, shrink=0.5, aspect=10, label=z_label)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
