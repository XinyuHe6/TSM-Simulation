from . import erdos_renyi, k_regular, random_graph
from .common import require_param


GRAPH_MODULES = {
    "random": random_graph,
    "k_regular": k_regular,
    "erdos_renyi": erdos_renyi,
}

GRAPH_SPECS = {
    name: module.SPEC
    for name, module in GRAPH_MODULES.items()
}

GRAPH_ALIASES = {}
for name, module in GRAPH_MODULES.items():
    for alias in module.ALIASES:
        GRAPH_ALIASES[alias] = name


def canonical_graph_name(name):
    key = str(name).strip().lower()
    canonical = GRAPH_ALIASES.get(key)
    if canonical is None:
        valid = ", ".join(sorted(GRAPH_SPECS))
        raise ValueError(f"Unknown graph algorithm '{name}'. Valid options: {valid}.")
    return canonical


def generate_random_graph(A_size, I_size, edge_prob):
    return random_graph.generate(A_size, I_size, edge_prob)


def generate_k_regular_graph(A_size, I_size, degree):
    return k_regular.generate(A_size, I_size, degree)


def generate_erdos_renyi_graph(A_size, I_size, k=None, edge_prob=None):
    return erdos_renyi.generate(A_size, I_size, k=k, edge_prob=edge_prob)


def generate_graph(name, A_size, I_size, **params):
    """
    Unified graph-generation interface.

    All graph algorithms return neighbors: list[list[int]].
    """
    canonical = canonical_graph_name(name)
    if canonical == "random":
        return random_graph.generate(
            A_size=A_size,
            I_size=I_size,
            edge_prob=require_param(params, "edge_prob", canonical),
        )
    if canonical == "k_regular":
        return k_regular.generate(
            A_size=A_size,
            I_size=I_size,
            degree=require_param(params, "degree", canonical),
        )
    if canonical == "erdos_renyi":
        return erdos_renyi.generate(
            A_size=A_size,
            I_size=I_size,
            k=params.get("k"),
            edge_prob=params.get("edge_prob"),
        )
    raise AssertionError(f"Unhandled graph algorithm: {canonical}")
