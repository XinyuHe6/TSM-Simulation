from .common import compute_offline_opt


SPEC = {
    "csv": "fluid_lp",
    "label": "Fluid LP / Offline OPT",
    "marker": "*",
}

ALIASES = ("fluid", "fluid_lp", "opt", "offline_opt")


def run(A_size, I_size, neighbors, arrivals, state=None):
    del I_size, state
    return compute_offline_opt(A_size, neighbors, arrivals)
