"""Brubach et al.'s 0.7299 algorithm, specialized to unweighted matching.

This implements Algorithm 8 (VW) from "Online Stochastic Matching: New
Algorithms and Bounds" (arXiv:1606.06395v4).  The paper analyzes both
vertex-weighted and unweighted instances; this module intentionally exposes
only the unweighted case used by this simulation project.

The offline phase solves LP (1)--(5), applies DR[f, 3], breaks C2/C3 four
cycles, and applies the balancing rules in Figure 4.  The online phase runs
the Randomized List Algorithm (Algorithm 6).
"""

import math
import random
from collections import defaultdict
from itertools import combinations

from .common import require_state


try:
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
except ImportError as exc:  # pragma: no cover - depends on the environment.
    linprog = None
    csr_matrix = None
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None


SPEC = {
    "csv": "brubach_vw",
    "label": "Brubach VW (Unweighted)",
    "marker": "h",
}

ALIASES = (
    "brubach",
    "brubach_vw",
    "unweighted_vw",
    "vw",
)


EDGE_CAP = 1.0 - math.exp(-1.0)
PAIR_CAP = 1.0 - math.exp(-2.0)
X1 = 0.2744
X2 = 0.15877
TOL = 1e-9


def _require_scipy():
    if linprog is None or csr_matrix is None:
        raise ImportError(
            "The brubach_vw algorithm requires scipy to solve its benchmark LP. "
            "Install scipy in the active virtual environment."
        ) from SCIPY_IMPORT_ERROR


def _expand_types(A_size, I_size, neighbors, e):
    if len(neighbors) != I_size:
        raise ValueError("neighbors must contain one list for each impression type.")
    if len(e) != I_size:
        raise ValueError("e must contain one arrival count for each impression type.")

    expanded_neighbors = []
    copies_of_type = [[] for _ in range(I_size)]
    for i in range(I_size):
        clean_neighbors = sorted(set(neighbors[i]))
        if any(a < 0 or a >= A_size for a in clean_neighbors):
            raise ValueError(f"neighbors[{i}] contains an invalid advertiser id.")
        for _ in range(e[i]):
            copy = len(expanded_neighbors)
            expanded_neighbors.append(clean_neighbors)
            copies_of_type[i].append(copy)
    return expanded_neighbors, copies_of_type


def _build_sparse_constraint_matrix(constraints, n_vars):
    data = []
    rows = []
    cols = []
    for row, (indices, _rhs) in enumerate(constraints):
        for col in indices:
            rows.append(row)
            cols.append(col)
            data.append(1.0)
    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(constraints), n_vars),
    )


def _solve_benchmark_lp(A_size, expanded_neighbors):
    """Solve the unweighted benchmark LP (paper constraints (1)--(5))."""
    _require_scipy()

    edge_list = []
    edges_by_copy = [[] for _ in expanded_neighbors]
    edges_by_advertiser = [[] for _ in range(A_size)]
    for copy, neigh_copy in enumerate(expanded_neighbors):
        for a in neigh_copy:
            edge_index = len(edge_list)
            edge_list.append((copy, a))
            edges_by_copy[copy].append(edge_index)
            edges_by_advertiser[a].append(edge_index)

    if not edge_list:
        return edge_list, [], 0.0

    constraints = []
    for indices in edges_by_copy:
        if indices:
            constraints.append((tuple(indices), 1.0))

    for incident in edges_by_advertiser:
        if incident:
            constraints.append((tuple(incident), 1.0))
        for left, right in combinations(incident, 2):
            constraints.append(((left, right), PAIR_CAP))

    matrix = _build_sparse_constraint_matrix(constraints, len(edge_list))
    result = linprog(
        c=[-1.0] * len(edge_list),
        A_ub=matrix,
        b_ub=[rhs for _indices, rhs in constraints],
        bounds=[(0.0, EDGE_CAP)] * len(edge_list),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Brubach benchmark LP solve failed: {result.message}")

    solution = [
        min(EDGE_CAP, max(0.0, float(value)))
        for value in result.x
    ]
    return edge_list, solution, -float(result.fun)


def _fractional_chain(edge_list, residual, active):
    """Return an even cycle, or a leaf-to-leaf path, of fractional edges."""
    adjacency = defaultdict(list)
    for edge_index in active:
        copy, a = edge_list[edge_index]
        left = (0, a)
        right = (1, copy)
        adjacency[left].append(edge_index)
        adjacency[right].append(edge_index)

    for incident in adjacency.values():
        incident.sort()

    color = {}
    parent_node = {}
    parent_edge = {}
    depth = {}

    for root in sorted(adjacency):
        if root in color:
            continue
        color[root] = 1
        depth[root] = 0
        stack = [(root, iter(adjacency[root]))]

        while stack:
            node, incident_iter = stack[-1]
            try:
                edge_index = next(incident_iter)
            except StopIteration:
                color[node] = 2
                stack.pop()
                continue

            if parent_edge.get(node) == edge_index:
                continue
            copy, a = edge_list[edge_index]
            left = (0, a)
            right = (1, copy)
            other = right if node == left else left

            if other not in color:
                color[other] = 1
                parent_node[other] = node
                parent_edge[other] = edge_index
                depth[other] = depth[node] + 1
                stack.append((other, iter(adjacency[other])))
                continue

            if color[other] == 1 and depth[other] < depth[node]:
                cycle = [edge_index]
                cursor = node
                while cursor != other:
                    cycle.append(parent_edge[cursor])
                    cursor = parent_node[cursor]
                if len(cycle) % 2 != 0:
                    raise AssertionError("Fractional bipartite graph has an odd cycle.")
                return cycle

    leaves = sorted(node for node, incident in adjacency.items() if len(incident) == 1)
    if not leaves:
        raise AssertionError("Could not find a cycle or path in the fractional graph.")

    path = []
    node = leaves[0]
    previous_edge = None
    while True:
        choices = [
            edge_index
            for edge_index in adjacency[node]
            if edge_index != previous_edge
        ]
        if not choices:
            break
        edge_index = choices[0]
        path.append(edge_index)
        copy, a = edge_list[edge_index]
        left = (0, a)
        right = (1, copy)
        node = right if node == left else left
        previous_edge = edge_index

    return path


def _dependent_round(edge_list, fractional_values, rng):
    """Apply GKPS dependent rounding to 3f and return integral F."""
    base = []
    residual = []
    for value in fractional_values:
        scaled = 3.0 * float(value)
        nearest = round(scaled)
        if abs(scaled - nearest) <= TOL:
            base.append(int(nearest))
            residual.append(0.0)
        else:
            integer = math.floor(scaled)
            base.append(int(integer))
            residual.append(scaled - integer)

    active = {
        edge_index
        for edge_index, value in enumerate(residual)
        if TOL < value < 1.0 - TOL
    }

    while active:
        chain = _fractional_chain(edge_list, residual, active)
        plus = chain[0::2]
        minus = chain[1::2]

        alpha = min(
            [1.0 - residual[edge_index] for edge_index in plus]
            + [residual[edge_index] for edge_index in minus]
        )
        beta = min(
            [residual[edge_index] for edge_index in plus]
            + [1.0 - residual[edge_index] for edge_index in minus]
        )
        if alpha + beta <= TOL:
            raise AssertionError("Dependent rounding made no progress.")

        if rng.random() < beta / (alpha + beta):
            plus_delta = alpha
            minus_delta = -alpha
        else:
            plus_delta = -beta
            minus_delta = beta

        for edge_index in plus:
            residual[edge_index] += plus_delta
        for edge_index in minus:
            residual[edge_index] += minus_delta

        for edge_index in chain:
            value = residual[edge_index]
            if value <= TOL:
                residual[edge_index] = 0.0
                active.discard(edge_index)
            elif value >= 1.0 - TOL:
                residual[edge_index] = 1.0
                active.discard(edge_index)

    rounded = [
        integer + int(round(value))
        for integer, value in zip(base, residual)
    ]
    if any(value not in (0, 1, 2) for value in rounded):
        raise AssertionError("DR[f, 3] produced an edge outside {0, 1, 2}.")
    return rounded


def _find_four_cycle(weights, cycle_type):
    support_by_copy = defaultdict(set)
    for (copy, a), value in weights.items():
        if value > 0:
            support_by_copy[copy].add(a)

    copies = sorted(support_by_copy)
    for left_pos, copy1 in enumerate(copies):
        for copy2 in copies[left_pos + 1:]:
            common = sorted(support_by_copy[copy1] & support_by_copy[copy2])
            for a1, a2 in combinations(common, 2):
                keys = (
                    (copy1, a1),
                    (copy2, a1),
                    (copy1, a2),
                    (copy2, a2),
                )
                values = [weights[key] for key in keys]
                if cycle_type == "c2" and sorted(values) == [1, 1, 1, 2]:
                    return keys
                if cycle_type == "c3" and values == [1, 1, 1, 1]:
                    return keys
    return None


def _set_multiplicity(weights, key, value):
    if value == 0:
        weights.pop(key, None)
    else:
        weights[key] = value


def _break_c2(weights, keys):
    thick = next(key for key in keys if weights[key] == 2)
    thick_copy, thick_a = thick
    other_copy = next(copy for copy, _a in keys if copy != thick_copy)
    other_a = next(a for _copy, a in keys if a != thick_a)
    opposite = (other_copy, other_a)
    cross1 = (thick_copy, other_a)
    cross2 = (other_copy, thick_a)

    _set_multiplicity(weights, thick, weights[thick] - 1)
    _set_multiplicity(weights, opposite, weights[opposite] - 1)
    _set_multiplicity(weights, cross1, weights[cross1] + 1)
    _set_multiplicity(weights, cross2, weights[cross2] + 1)


def _break_c3(weights, keys):
    copies = sorted({copy for copy, _a in keys})
    advertisers = sorted({a for _copy, a in keys})
    keep = {
        (copies[0], advertisers[0]),
        (copies[1], advertisers[1]),
    }
    for key in keys:
        _set_multiplicity(weights, key, 2 if key in keep else 0)


def _degrees(weights):
    degree_by_advertiser = defaultdict(int)
    degree_by_copy = defaultdict(int)
    for (copy, a), value in weights.items():
        degree_by_advertiser[a] += value
        degree_by_copy[copy] += value
    return degree_by_advertiser, degree_by_copy


def _break_short_cycles(weights):
    """Apply Algorithm 7 while preserving all vertex degrees."""
    weights = dict(weights)
    before = _degrees(weights)
    max_steps = 10 * (len(weights) + 1) ** 2

    for _step in range(max_steps):
        c2 = _find_four_cycle(weights, "c2")
        if c2 is not None:
            _break_c2(weights, c2)
            continue

        c3 = _find_four_cycle(weights, "c3")
        if c3 is not None:
            _break_c3(weights, c3)
            continue
        break
    else:
        raise RuntimeError("Cycle breaking did not converge.")

    if _degrees(weights) != before:
        raise AssertionError("Cycle breaking changed a vertex degree.")
    return weights


def _balanced_two_edge_values(small_degree, large_degree, small_other_edges):
    if large_degree == 2:
        small_value = {1: 0.25, 2: 0.30, 3: 0.40}[small_degree]
        return small_value, 1.0 - small_value

    if large_degree == 3 and small_degree in (1, 2):
        small_value = {1: 0.10, 2: 0.15}[small_degree]
        return small_value, 1.0 - small_value

    if large_degree == 3 and small_degree == 3:
        small_value = X1 if sorted(small_other_edges) == [2] else X2
        return small_value, 1.0 - small_value

    raise AssertionError("Unexpected two-edge configuration after DR[f, 3].")


def _balance_h(weights, expanded_count):
    """Apply every Figure 4 balancing rule, returning H-prime."""
    degree_by_advertiser, _degree_by_copy = _degrees(weights)
    incident_by_advertiser = defaultdict(list)
    incident_by_copy = defaultdict(list)
    for (copy, a), value in weights.items():
        incident_by_advertiser[a].append((copy, value))
        incident_by_copy[copy].append((a, value))

    h_prime = {}
    three_edge_rules = {
        (1, 1, 3): {1: 0.10, 3: 0.80},
        (1, 2, 3): {1: 0.15, 2: 0.20, 3: 0.65},
        (1, 3, 3): {1: 0.10, 3: 0.45},
        (2, 2, 3): {2: 0.25, 3: 0.50},
        (2, 3, 3): {2: 0.20, 3: 0.40},
    }

    for copy in range(expanded_count):
        incident = sorted(incident_by_copy.get(copy, []))
        total = sum(value for _a, value in incident)

        if total == 3 and sorted(value for _a, value in incident) == [1, 2]:
            small_a = next(a for a, value in incident if value == 1)
            large_a = next(a for a, value in incident if value == 2)
            small_degree = degree_by_advertiser[small_a]
            large_degree = degree_by_advertiser[large_a]
            small_other_edges = [
                value
                for other_copy, value in incident_by_advertiser[small_a]
                if other_copy != copy
            ]
            small_value, large_value = _balanced_two_edge_values(
                small_degree,
                large_degree,
                small_other_edges,
            )
            h_prime[(copy, small_a)] = small_value
            h_prime[(copy, large_a)] = large_value
            continue

        if total == 3 and len(incident) == 3:
            degree_pattern = tuple(
                sorted(degree_by_advertiser[a] for a, _value in incident)
            )
            rule = three_edge_rules.get(degree_pattern)
            if rule is not None:
                for a, _value in incident:
                    h_prime[(copy, a)] = rule[degree_by_advertiser[a]]
                continue

        for a, value in incident:
            h_prime[(copy, a)] = value / 3.0

    for copy in range(expanded_count):
        old_sum = sum(
            value / 3.0
            for (edge_copy, _a), value in weights.items()
            if edge_copy == copy
        )
        new_sum = sum(
            value
            for (edge_copy, _a), value in h_prime.items()
            if edge_copy == copy
        )
        if abs(old_sum - new_sum) > TOL:
            raise AssertionError("Figure 4 balancing changed an online degree.")
    return h_prime


def _build_list_distributions(h_prime, expanded_count):
    by_copy = defaultdict(list)
    for (copy, a), value in h_prime.items():
        if value > TOL:
            by_copy[copy].append((a, value))

    distributions = []
    for copy in range(expanded_count):
        candidates = sorted(by_copy.get(copy, []))
        real_mass = sum(value for _a, value in candidates)
        if real_mass > 1.0 + TOL:
            raise AssertionError("RLA probabilities exceed one.")
        dummy_mass = max(0.0, 1.0 - real_mass)
        if dummy_mass > TOL:
            candidates.append((None, dummy_mass))
        distributions.append(tuple(candidates))
    return distributions


def _weighted_random_order(distribution):
    remaining = list(distribution)
    ordered = []
    while remaining:
        total = sum(weight for _candidate, weight in remaining)
        draw = random.random() * total
        cumulative = 0.0
        chosen_index = len(remaining) - 1
        for index, (_candidate, weight) in enumerate(remaining):
            cumulative += weight
            if draw <= cumulative:
                chosen_index = index
                break
        candidate, _weight = remaining.pop(chosen_index)
        ordered.append(candidate)
    return ordered


def prepare_state(
    A_size,
    I_size,
    neighbors,
    p=None,
    T=None,
    e=None,
    seed=0,
    **kwargs,
):
    del kwargs
    from arrival_algorithms import integerize_probability_vector, validate_arrival_counts

    if e is None:
        if p is None or T is None:
            raise ValueError("brubach_vw prepare_state requires e or both p and T.")
        e = integerize_probability_vector(T, p)
    else:
        e = validate_arrival_counts(e)

    expanded_neighbors, copies_of_type = _expand_types(
        A_size,
        I_size,
        neighbors,
        e,
    )
    edge_list, f, lp_value = _solve_benchmark_lp(A_size, expanded_neighbors)
    rounded = _dependent_round(edge_list, f, random.Random(seed))
    multiplicities = {
        edge: value
        for edge, value in zip(edge_list, rounded)
        if value > 0
    }
    multiplicities = _break_short_cycles(multiplicities)
    h_prime = _balance_h(multiplicities, len(expanded_neighbors))
    distributions = _build_list_distributions(h_prime, len(expanded_neighbors))

    return {
        "e": e,
        "copies_of_type": copies_of_type,
        "expanded_neighbors": expanded_neighbors,
        "lp_value": lp_value,
        "multiplicities": multiplicities,
        "h_prime": h_prime,
        "list_distributions": distributions,
    }


def run(A_size, I_size, neighbors, arrivals, state=None):
    del neighbors
    require_state("brubach_vw", state)
    if len(state["copies_of_type"]) != I_size:
        raise ValueError("Prepared brubach_vw state does not match I_size.")

    matched = [False] * A_size
    alg = 0
    copies_of_type = state["copies_of_type"]
    distributions = state["list_distributions"]

    for i in arrivals:
        if i < 0 or i >= I_size:
            raise ValueError(f"Arrival type {i} is outside [0, {I_size}).")
        if not copies_of_type[i]:
            continue

        copy = random.choice(copies_of_type[i])
        for candidate in _weighted_random_order(distributions[copy]):
            if candidate is None:
                break
            if not matched[candidate]:
                matched[candidate] = True
                alg += 1
                break

    return alg
