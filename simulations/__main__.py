"""Command-line dispatcher for the simulation package."""

import argparse
import importlib
import sys


COMMANDS = {
    "random-edge": (
        "compare_random_edge_prob",
        "Compare ALG/OPT while sweeping independent edge probability.",
    ),
    "raw-counts": (
        "compare_random_edge_prob_alg_counts",
        "Compare raw matched counts while sweeping edge probability.",
    ),
    "aeqieqt": (
        "compare_AeqIeqT_edge_prob",
        "Edge-probability comparison with A = I = T = N.",
    ),
    "k-regular": (
        "compare_k_regular",
        "Compare ALG/OPT while sweeping k-regular degree.",
    ),
    "erdos-renyi": (
        "compare_erdos_renyi",
        "Compare ALG/OPT on G(n,n,k/n).",
    ),
    "tsm-surface": (
        "compare_tsm_surface_random",
        "Plot one algorithm over arrival horizon and edge probability.",
    ),
    "manshadi-surface": (
        "compare_manshadi_surface",
        "Plot one algorithm over arrival horizon and graph density.",
    ),
    "manshadi-tries": (
        "compare_manshadi_tries",
        "Compare arbitrary Manshadi retry counts.",
    ),
    "correlated-2d": (
        "compare_correlated_sampling_2d",
        "Plot a Correlated Sampling curve.",
    ),
    "correlated-surface": (
        "compare_correlated_sampling_surface",
        "Plot a Correlated Sampling surface.",
    ),
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="tsm-sim",
        description="Run reproducible two-sided matching simulations.",
    )
    parser.add_argument(
        "command",
        choices=COMMANDS,
        help="Experiment to run. Use 'tsm-sim COMMAND --help' for its arguments.",
    )
    return parser


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in COMMANDS:
        command = raw_args.pop(0)
    else:
        command = build_parser().parse_args(raw_args).command

    module_name, _description = COMMANDS[command]
    sys.argv = [f"tsm-sim {command}", *raw_args]
    module = importlib.import_module(f"simulations.{module_name}")
    return module.main()


if __name__ == "__main__":
    main()
