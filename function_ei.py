import random
import math
import networkx as nx
import matplotlib.pyplot as plt

# =========================
# Helper: compute integer e_i from p, T
# =========================

def compute_integer_eis(T, p):
    """
    Given horizon T and probability vector p over impression types,
    compute integer e_i:
        e_i = floor(T * p_i)
    then distribute the remaining
        r = T - sum_i floor(T * p_i)
    by giving +1 to the r largest p_i.
    """
    I_size = len(p)
    # floor part
    e = [math.floor(T * pi) for pi in p]
    total = sum(e)
    remaining = T - total

    if remaining > 0:
        # indices sorted by p_i descending
        order = sorted(range(I_size), key=lambda i: p[i], reverse=True)
        for k in range(remaining):
            e[order[k]] += 1

    return e  # list of length I_size, sum(e) == T (unless T=0)


# =========================
# TSM flow construction (on expanded types)
# =========================

def compute_flow_edges(A_size, I_size, neighbors):
    """
    Build boosted flow network (TSM) and return edges (a, i) with unit flow (Ef)
    on the *expanded* type index 0..I_size-1.
    """
    Gf = nx.DiGraph()
    source = "s"
    sink = "t"

    # Advertiser nodes
    Gf.add_node(source)
    Gf.add_node(sink)

    for a in range(A_size):
        Gf.add_node(("a", a))
        Gf.add_edge(source, ("a", a), capacity=2)

    # Impression-type nodes (expanded index)
    for i in range(I_size):
        Gf.add_node(("i", i))
        Gf.add_edge(("i", i), sink, capacity=2)

    # Edges between advertisers and types
    for i in range(I_size):
        for a in neighbors[i]:
            Gf.add_edge(("a", a), ("i", i), capacity=1)

    flow_value, flow_dict = nx.maximum_flow(Gf, source, sink)

    Ef = []
    for a in range(A_size):
        a_node = ("a", a)
        if a_node not in flow_dict:
            continue
        for i in range(I_size):
            i_node = ("i", i)
            if i_node in flow_dict[a_node] and flow_dict[a_node][i_node] == 1:
                Ef.append((a, i))

    return Ef


def build_subgraph_and_components(A_size, I_size, Ef):
    """
    Build undirected subgraph H from Ef and return connected components as edge lists.
    Nodes are 0..A_size-1 for advertisers, A_size..A_size+I_size-1 for types.
    """
    H_adj = {u: set() for u in range(A_size + I_size)}
    edges_set = set()

    for (a, i) in Ef:
        u = a
        v = A_size + i
        H_adj[u].add(v)
        H_adj[v].add(u)
        edges_set.add((min(u, v), max(u, v)))

    visited = set()
    components = []  # list of (comp_nodes, comp_edges)

    for start in range(A_size + I_size):
        if start in visited or len(H_adj[start]) == 0:
            continue

        stack = [start]
        comp_nodes = set()
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp_nodes.add(u)
            for v in H_adj[u]:
                if v not in visited:
                    stack.append(v)

        comp_edges = []
        for (u, v) in edges_set:
            if u in comp_nodes and v in comp_nodes:
                comp_edges.append((u, v))
        components.append((comp_nodes, comp_edges))

    return H_adj, components


def order_edges_in_component(comp_nodes, comp_edges, H_adj):
    """
    Given one connected component (path or cycle), return edges in walk order.
    """
    adj = {u: set() for u in comp_nodes}
    for (u, v) in comp_edges:
        adj[u].add(v)
        adj[v].add(u)

    degree = {u: len(adj[u]) for u in comp_nodes}
    endpoints = [u for u in comp_nodes if degree[u] == 1]

    if len(endpoints) == 0:
        # cycle
        start = next(iter(comp_nodes))
    else:
        # path
        start = endpoints[0]

    ordered_edges = []
    visited_edges = set()
    current = start

    total_edges = len(comp_edges)
    while len(ordered_edges) < total_edges:
        for nxt in adj[current]:
            e = (min(current, nxt), max(current, nxt))
            if e in visited_edges:
                continue
            visited_edges.add(e)
            ordered_edges.append((current, nxt))
            current = nxt
            break

    return ordered_edges, (len(endpoints) == 0), endpoints


def color_component(comp_nodes, comp_edges, H_adj, A_size):
    """
    Color one component's edges blue/red according to TSM rules.
    Return dict: key = (a, i_expanded), value = "blue"/"red".
    """
    if not comp_edges:
        return {}

    ordered_edges, is_cycle, endpoints = order_edges_in_component(
        comp_nodes, comp_edges, H_adj
    )
    m = len(ordered_edges)

    def node_type(u):
        return "A" if u < A_size else "I"

    colors = [""] * m

    if is_cycle:
        # cycle: alternate blue/red
        for idx in range(m):
            colors[idx] = "blue" if idx % 2 == 0 else "red"
    else:
        start = endpoints[0]
        end = endpoints[1]
        start_type = node_type(start)
        end_type = node_type(end)

        if m % 2 == 1:
            # odd-length path, start blue
            for idx in range(m):
                colors[idx] = "blue" if idx % 2 == 0 else "red"
        else:
            if start_type == "A" and end_type == "A":
                # A-A path: alternate
                for idx in range(m):
                    colors[idx] = "blue" if idx % 2 == 0 else "red"
            elif start_type == "I" and end_type == "I":
                # I-I path: first two blue, then red/blue/...
                if m >= 1:
                    colors[0] = "blue"
                if m >= 2:
                    colors[1] = "blue"
                for idx in range(2, m):
                    colors[idx] = "red" if (idx % 2 == 0) else "blue"
            else:
                # mixed A-I path (shouldn't appear in ideal TSM structure)
                for idx in range(m):
                    colors[idx] = "blue" if idx % 2 == 0 else "red"

    edge_color_ai = {}
    for idx, (u, v) in enumerate(ordered_edges):
        col = colors[idx]
        if u < A_size:
            a = u
            i_exp = v - A_size
        else:
            a = v
            i_exp = u - A_size
        edge_color_ai[(a, i_exp)] = col

    return edge_color_ai


def compute_blue_red_matchings(A_size, I_size, neighbors, e):
    """
    Build expanded impression types according to integer e[i]:
        - For each original type i, create e[i] copies with same neighbors.
    Run TSM flow on the expanded graph.
    Return:
        blue_for_copy[j], red_for_copy[j] for expanded types j,
        copies_of_type[i] = list of expanded indices for original type i.
    """
    # Expand types
    neighbors_expanded = []
    copies_of_type = [[] for _ in range(I_size)]

    for i in range(I_size):
        for _ in range(e[i]):
            j = len(neighbors_expanded)
            neighbors_expanded.append(neighbors[i])
            copies_of_type[i].append(j)

    I_exp = len(neighbors_expanded)

    if I_exp == 0:
        # No expanded types; return empty
        return [], [], copies_of_type

    Ef = compute_flow_edges(A_size, I_exp, neighbors_expanded)
    H_adj, components = build_subgraph_and_components(A_size, I_exp, Ef)

    blue_for_copy = [None] * I_exp
    red_for_copy = [None] * I_exp

    for comp_nodes, comp_edges in components:
        edge_color_ai = color_component(comp_nodes, comp_edges, H_adj, A_size)
        for (a, i_exp), col in edge_color_ai.items():
            if col == "blue":
                blue_for_copy[i_exp] = a
            elif col == "red":
                red_for_copy[i_exp] = a

    return blue_for_copy, red_for_copy, copies_of_type


# =========================
# TSM simulation (using expanded copies)
# =========================

def simulate_tsm_once(A_size, I_size, neighbors, p, T,
                      blue_for_copy, red_for_copy, copies_of_type, e):
    """
    Simulate one run of TSM for horizon T with integer e[i] and expanded types.
    Returns:
        ALG: matched advertisers count
        hat_I: list of impression ids
        edges_real: list of (a, imp_id) edges in realization graph
        imp_type: list mapping imp_id -> original type i
    """
    matched_A = [False] * A_size
    ALG = 0

    # Counters per *expanded* type copy
    I_exp = len(blue_for_copy)
    x_copy = [0] * I_exp

    hat_I = []
    imp_type = []
    edges_real = []

    for _ in range(T):
        # sample original type i according to p
        i = random.choices(range(I_size), weights=p, k=1)[0]

        imp_id = len(hat_I)
        hat_I.append(imp_id)
        imp_type.append(i)

        # build realization edges for OPT
        for a in neighbors[i]:
            edges_real.append((a, imp_id))

        # TSM online decision using expanded copies
        if e[i] > 0 and copies_of_type[i]:
            # randomly pick a copy of type i
            j = random.choice(copies_of_type[i])  # expanded index

            if x_copy[j] == 0:
                a = blue_for_copy[j]
                if a is not None and not matched_A[a]:
                    matched_A[a] = True
                    ALG += 1
            elif x_copy[j] == 1:
                a = red_for_copy[j]
                if a is not None and not matched_A[a]:
                    matched_A[a] = True
                    ALG += 1
            # else x_copy[j] >= 2: ignore further arrivals for this copy

            x_copy[j] += 1
        # if e[i] == 0: TSM has no offline guidance for this type; do nothing

    return ALG, hat_I, edges_real, imp_type


def compute_opt_from_realization(A_size, hat_I, edges_real):
    """
    Compute offline OPT as maximum bipartite matching on realization graph.
    """
    G = nx.Graph()
    left_nodes = []

    for a in range(A_size):
        node = f"a_{a}"
        G.add_node(node, bipartite=0)
        left_nodes.append(node)

    for imp_id in hat_I:
        node = f"i_{imp_id}"
        G.add_node(node, bipartite=1)

    for (a, imp_id) in edges_real:
        G.add_edge(f"a_{a}", f"i_{imp_id}")

    matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=left_nodes)

    matched_count = 0
    for a in range(A_size):
        node = f"a_{a}"
        if node in matching:
            matched_count += 1

    return matched_count


def simulate_many_runs(A_size, I_size, neighbors, p, T, num_runs=1):
    """
    Main TSM wrapper:
        - compute integer e_i from p, T
        - build expanded types and blue/red matchings
        - run multiple simulations, compute ALG/OPT ratio
    """
    # 1) compute e_i from p, T
    e = compute_integer_eis(T, p)

    # 2) offline TSM guidance on expanded graph
    blue_for_copy, red_for_copy, copies_of_type = compute_blue_red_matchings(
        A_size, I_size, neighbors, e
    )

    ratios = []
    for _ in range(num_runs):
        ALG, hat_I, edges_real, imp_type = simulate_tsm_once(
            A_size, I_size, neighbors, p, T,
            blue_for_copy, red_for_copy, copies_of_type, e
        )
        OPT = compute_opt_from_realization(A_size, hat_I, edges_real)
        if OPT > 0:
            ratios.append(ALG / OPT)

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return avg_ratio, ratios


# =========================
# Random graph + p generator
# =========================

def generate_random_graph(A_size, I_size, edge_prob):
    """
    neighbors[i] = list of advertisers a that impression type i can connect to.
    Each edge (a, i) is included independently with probability edge_prob.
    """
    neighbors = []
    for i in range(I_size):
        neigh_i = [a for a in range(A_size) if random.random() < edge_prob]
        if not neigh_i:  # avoid isolated types if you want
            neigh_i = [random.randrange(A_size)]
        neighbors.append(neigh_i)
    return neighbors


def random_probability_vector(I_size):
    """Random probability vector over impression types 0..I_size-1."""
    raw = [random.random() for _ in range(I_size)]
    s = sum(raw)
    return [x / s for x in raw]


# =========================
# Degree-based greedy (unchanged)
# =========================

def simulate_degree_greedy_once(A_size, I_size, neighbors, p, T, adv_degrees):
    matched_A = [False] * A_size
    ALG = 0

    hat_I = []
    imp_type = []
    edges_real = []

    for _ in range(T):
        i = random.choices(range(I_size), weights=p, k=1)[0]

        imp_id = len(hat_I)
        hat_I.append(imp_id)
        imp_type.append(i)

        current_neighbors = neighbors[i]
        for a in current_neighbors:
            edges_real.append((a, imp_id))

        candidates = [a for a in current_neighbors if not matched_A[a]]

        if candidates:
            best_a = min(candidates, key=lambda x: adv_degrees[x])
            matched_A[best_a] = True
            ALG += 1

    return ALG, hat_I, edges_real, imp_type


def simulate_many_runs_degree(A_size, I_size, neighbors, p, T, num_runs=1):
    adv_degrees = [0] * A_size
    for i in range(I_size):
        for a in neighbors[i]:
            adv_degrees[a] += 1

    ratios = []
    for _ in range(num_runs):
        ALG, hat_I, edges_real, imp_type = simulate_degree_greedy_once(
            A_size, I_size, neighbors, p, T, adv_degrees
        )

        OPT = compute_opt_from_realization(A_size, hat_I, edges_real)

        if OPT > 0:
            ratios.append(ALG / OPT)
        else:
            if T > 0:
                ratios.append(1.0)

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return avg_ratio, ratios


# =========================
# simulate_random_graphs helper (TSM-based)
# =========================

def simulate_random_graphs(
    num_graphs,
    A_size,
    I_size,
    edge_prob,
    T,
    num_runs_per_graph=20,
):
    """
    For num_graphs random graphs, compute the average ALG/OPT ratio
    using simulate_many_runs() (TSM).
    """
    avg_ratios = []
    for _ in range(num_graphs):
        neighbors = generate_random_graph(A_size, I_size, edge_prob)
        p = random_probability_vector(I_size)

        avg_ratio, _ = simulate_many_runs(
            A_size, I_size, neighbors, p, T, num_runs=num_runs_per_graph
        )

        avg_ratios.append(avg_ratio)

    return avg_ratios
