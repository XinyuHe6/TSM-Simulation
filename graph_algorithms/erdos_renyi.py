from .random_graph import generate as generate_random


SPEC = {
    "csv": "erdos_renyi",
    "label": "Erdos-Renyi",
    "required": ("k or edge_prob",),
}

ALIASES = ("erdos_renyi", "erdos-renyi", "erdos", "er")


def generate(A_size, I_size, k=None, edge_prob=None):
    """
    Bipartite Erdos-Renyi graph.

    Parameters:
        edge_prob: direct edge probability.
        k: expected impression-side degree. If provided, edge_prob = k / A_size.

    For A_size == I_size == n, k recovers the old G(n,n,p=k/n) experiment.
    Empty neighbor lists are allowed.
    """
    if edge_prob is None:
        if k is None:
            raise ValueError("Graph 'erdos_renyi' requires 'k' or 'edge_prob'.")
        edge_prob = float(k) / float(A_size)

    return generate_random(A_size, I_size, edge_prob)
