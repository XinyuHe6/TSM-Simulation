def validate_edge_prob(edge_prob):
    edge_prob = float(edge_prob)
    if edge_prob < -1e-12 or edge_prob > 1.0 + 1e-12:
        raise ValueError(f"edge_prob must be in [0, 1], got {edge_prob}.")
    return min(1.0, max(0.0, edge_prob))


def require_param(params, key, graph_name):
    if key not in params:
        raise ValueError(f"Graph '{graph_name}' requires parameter '{key}'.")
    return params[key]
