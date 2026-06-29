import random

from .common import validate_edge_prob


SPEC = {
    "csv": "random",
    "label": "Random",
    "required": ("edge_prob",),
}

ALIASES = ("random", "random_graph")


def generate(A_size, I_size, edge_prob):
    """
    Independent-edge bipartite graph.

    Returns:
        neighbors[i] = list of advertisers connected to impression type i.

    Empty neighbor lists are allowed.
    """
    edge_prob = validate_edge_prob(edge_prob)
    return [
        [a for a in range(A_size) if random.random() < edge_prob]
        for _ in range(I_size)
    ]
