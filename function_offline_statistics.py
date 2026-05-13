import random
import math
import bisect
import networkx as nx

# ============================================================
# Graph generators (match function_ei style)
# ============================================================

def generate_random_graph(A_size, I_size, edge_prob):
    """
    neighbors[i] = list of advertisers a that impression type i can connect to.
    Each edge (a, i) is included independently with probability edge_prob.
    """
    neighbors = []
    for i in range(I_size):
        neigh_i = [a for a in range(A_size) if random.random() < edge_prob]
        if not neigh_i:  # avoid isolated types
            neigh_i = [random.randrange(A_size)]
        neighbors.append(neigh_i)
    return neighbors


def generate_k_regular_graph(A_size, I_size, degree):
    """
    Generate a graph where each impression type has exactly `degree`
    advertiser neighbors.

    - If A_size == I_size, build a simple bipartite k-regular graph via a
      random union of disjoint cyclic perfect matchings, so both sides have
      degree `degree`.
    - If A_size != I_size, a true bipartite k-regular graph on both sides is
      impossible in general. In that case we use the natural one-sided version
      for this codebase: each type i has exactly `degree` distinct advertisers.

    neighbors[i] remains the list of advertisers a connected to type i.
    """
    if degree < 0 or degree > A_size:
        raise ValueError(f"regular degree must satisfy 0 <= k <= {A_size}, got {degree}.")

    if A_size == I_size:
        shifts = random.sample(range(A_size), degree)
        neighbors = []
        for i in range(I_size):
            neigh_i = sorted((i + shift) % A_size for shift in shifts)
            neighbors.append(neigh_i)
        return neighbors

    advertisers = list(range(A_size))
    neighbors = []
    for _ in range(I_size):
        neigh_i = sorted(random.sample(advertisers, degree))
        neighbors.append(neigh_i)
    return neighbors


def generate_graph(A_size, I_size, graph_mode, graph_param):
    """
    Dispatch graph generation by mode.
    - random: graph_param is edge probability in [0, 1]
    - k_regular: graph_param is integer degree k
    """
    if graph_mode == "random":
        return generate_random_graph(A_size, I_size, float(graph_param))
    if graph_mode == "k_regular":
        return generate_k_regular_graph(A_size, I_size, int(graph_param))
    raise ValueError(f"Unsupported graph_mode: {graph_mode}")


def random_probability_vector(I_size):
    """Random probability vector over impression types 0..I_size-1."""
    raw = [random.random() for _ in range(I_size)]
    s = sum(raw)
    return [x / s for x in raw]


# ============================================================
# OPT (match function_ei: realization graph + bipartite matching)
# ============================================================

def compute_opt_from_realization(A_size, hat_I, edges_real):
    """
    Compute offline OPT as maximum bipartite matching on realization graph.
    Same style as function_ei.py.
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


# ============================================================
# Paper Algorithm 2 helpers: split rates to <= 1 (copies)
# ============================================================

def build_virtual_copies(T, p):
    """
    Paper assumes each type y has rate r_y <= 1.
    We set r_i = T * p_i, and split each original type i into m_i = ceil(r_i) copies.
    Each copy has rate r_i / m_i <= 1.

    Returns:
        r_copy_of_exp[j] : rate of expanded copy j
        orig_type_of_exp[j] : original type index for expanded copy j
        copies_of_type[i] : list of expanded indices for original type i
    """
    I_size = len(p)
    r_copy_of_exp = []
    orig_type_of_exp = []
    copies_of_type = [[] for _ in range(I_size)]

    for i in range(I_size):
        r_i = T * p[i]
        if r_i <= 0:
            continue
        m = max(1, int(math.ceil(r_i)))
        r_copy = r_i / m
        for _ in range(m):
            j = len(r_copy_of_exp)
            r_copy_of_exp.append(r_copy)
            orig_type_of_exp.append(i)
            copies_of_type[i].append(j)

    return r_copy_of_exp, orig_type_of_exp, copies_of_type


def poisson_sample(lam):
    """
    Knuth Poisson sampler.
    """
    L = math.exp(-lam)
    k = 0
    prod = 1.0
    while prod > L:
        k += 1
        prod *= random.random()
    return max(0, k - 1)


def sample_arrival_sequence(I_size, p, T, use_poisson_len=False):
    """
    Generate arrival sequence of original types.
    - if use_poisson_len=False: length = T (matches function_ei simulate loops)
    - if True: length ~ Poisson(T)
    """
    if use_poisson_len:
        B = poisson_sample(float(T))
    else:
        B = int(T)

    seq = [random.choices(range(I_size), weights=p, k=1)[0] for _ in range(B)]
    return seq


# ============================================================
# Monte Carlo to estimate f (offline statistics)
# ============================================================

def estimate_f_monte_carlo(A_size, I_size, neighbors, p, T, mc_trials=200, seed=0, use_poisson_len=False):
    """
    Monte Carlo estimate of f for expanded copies:
        f[j, a] ~ expected number of times OPT matches (copy j) to advertiser a per trial.

    Trial logic:
      1) sample arrival sequence of original types (length random or fixed)
      2) assign each arrival of type i to a random expanded copy j in copies_of_type[i]
      3) build realization graph edges_real using original neighbors[i]
      4) compute OPT matching on realization graph
      5) count matched edges (copy j -> advertiser a) using the matched impression ids

    Returns:
        r_copy_of_exp, copies_of_type, f_dict
        where f_dict[(j,a)] is float
    """
    random.seed(seed)

    # expanded copies for rate <= 1
    r_copy_of_exp, orig_type_of_exp, copies_of_type = build_virtual_copies(T, p)

    # store totals across trials
    total_f = {}  # (j, a) -> count

    for _ in range(mc_trials):
        arrivals = sample_arrival_sequence(I_size, p, T, use_poisson_len=use_poisson_len)

        # Build realization graph
        hat_I = []
        edges_real = []

        # For mapping each imp_id to its assigned expanded copy j (not just original type)
        imp_copy = []

        for _t, i in enumerate(arrivals):
            imp_id = len(hat_I)
            hat_I.append(imp_id)

            # assign to expanded copy j
            if copies_of_type[i]:
                j = random.choice(copies_of_type[i])
            else:
                # if r_i == 0 (shouldn't happen if p valid), just mark as -1
                j = -1
            imp_copy.append(j)

            # edges for OPT use original neighbors[i]
            for a in neighbors[i]:
                edges_real.append((a, imp_id))

        # compute OPT matching and also retrieve matched pairs
        # We'll re-run matching but keep mapping from impression node -> advertiser node.
        G = nx.Graph()
        left_nodes = []
        for a in range(A_size):
            node = f"a_{a}"
            G.add_node(node, bipartite=0)
            left_nodes.append(node)
        for imp_id in hat_I:
            G.add_node(f"i_{imp_id}", bipartite=1)
        for (a, imp_id) in edges_real:
            G.add_edge(f"a_{a}", f"i_{imp_id}")

        matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=left_nodes)

        # matching is dict: node -> matched node
        # Count only impression-side matched edges, map to (j,a)
        for imp_id in hat_I:
            imp_node = f"i_{imp_id}"
            if imp_node in matching:
                a_node = matching[imp_node]  # "a_k"
                if a_node.startswith("a_"):
                    a = int(a_node.split("_")[1])
                else:
                    continue

                j = imp_copy[imp_id]
                if j >= 0:
                    total_f[(j, a)] = total_f.get((j, a), 0) + 1

    # convert to expectation per trial
    f_dict = {}
    denom = float(mc_trials) if mc_trials > 0 else 1.0
    for k, v in total_f.items():
        f_dict[k] = v / denom

    return r_copy_of_exp, copies_of_type, f_dict


# ============================================================
# Build Iy / Jy partitions for each expanded copy j
# ============================================================

def build_partitions_for_copy(j, A_size, neighbors, orig_type_of_copy, r_copy, f_dict):
    """
    For expanded copy j:
      - original type i = orig_type_of_copy[j]
      - rate r = r_copy[j] <= 1
      - f(j,a) is expected matches per trial (Monte Carlo)
    Build Iy in descending f; add dummy with remaining mass r - sum f.
    Build Jy by shifting by f_max (paper Figure 1 / Algorithm 2 construction).
    """
    i = orig_type_of_copy[j]
    r = r_copy[j]
    nbrs = neighbors[i]

    # collect (a, f) for neighbors
    items = [(a, f_dict.get((j, a), 0.0)) for a in nbrs]
    items.sort(key=lambda t: t[1], reverse=True)

    sum_f = sum(val for _, val in items)
    if sum_f > r and sum_f > 0:
        scale = r / sum_f
        items = [(a, val * scale) for a, val in items]
        sum_f = r

    f_dummy = max(0.0, r - sum_f)

    # Iy
    Iy_bins = []
    Iy_ends = []
    cur = 0.0
    for a, val in items:
        cur += val
        Iy_bins.append(a)
        Iy_ends.append(cur)
    # dummy at end
    cur += f_dummy
    Iy_bins.append(None)
    Iy_ends.append(cur)

    # max interval
    if items and items[0][1] > 0:
        a_max, f_max = items[0]
    else:
        a_max, f_max = None, 0.0

    # Jy: shift by f_max
    if f_max <= 0:
        Jy_bins = Iy_bins[:]
        Jy_ends = Iy_ends[:]
    else:
        Jy_bins = []
        Jy_ends = []
        cur = 0.0

        # rest (excluding max)
        rest = items[1:]
        for a, val in rest:
            cur += val
            Jy_bins.append(a)
            Jy_ends.append(cur)

        # dummy chunk before the max chunk
        cur += f_dummy
        Jy_bins.append(None)
        Jy_ends.append(cur)

        # max chunk at end: [r - f_max, r]
        Jy_bins.append(a_max)
        Jy_ends.append(r)

    # numerical cleanup
    if Iy_ends:
        Iy_ends[-1] = r
    if Jy_ends:
        Jy_ends[-1] = r

    return (Iy_bins, Iy_ends, Jy_bins, Jy_ends, r)


def pick_from_partition(bins, ends, x):
    idx = bisect.bisect_left(ends, x)
    if idx >= len(bins):
        return bins[-1]
    return bins[idx]


# ============================================================
# Algorithm 2 simulation: one run + wrapper many runs (ratio)
# ============================================================

def simulate_offline_statistics_once(
    A_size, I_size, neighbors, p, T,
    r_copy_of_exp, orig_type_of_exp, copies_of_type,
    Iy_bins_list, Iy_ends_list, Jy_bins_list, Jy_ends_list,
    use_poisson_len=False
):
    """
    One run:
      - generate arrival sequence
      - online Algorithm 2 using (Iy/Jy) for expanded copies
      - build realization for OPT and compute OPT
    Returns: ALG, OPT
    """
    matched_A = [False] * A_size
    ALG = 0

    hat_I = []
    edges_real = []

    arrivals = sample_arrival_sequence(I_size, p, T, use_poisson_len=use_poisson_len)

    # each arrival assigned to random expanded copy j
    for _t, i in enumerate(arrivals):
        imp_id = len(hat_I)
        hat_I.append(imp_id)

        # build edges for OPT
        for a in neighbors[i]:
            edges_real.append((a, imp_id))

        # online decision: pick expanded copy j
        if copies_of_type[i]:
            j = random.choice(copies_of_type[i])
        else:
            continue

        r = r_copy_of_exp[j]
        if r <= 0:
            continue

        x = random.random() * r

        # first choice from Iy
        a1 = pick_from_partition(Iy_bins_list[j], Iy_ends_list[j], x)
        if a1 is not None and (not matched_A[a1]):
            matched_A[a1] = True
            ALG += 1
            continue

        # second choice from Jy
        a2 = pick_from_partition(Jy_bins_list[j], Jy_ends_list[j], x)
        if a2 is not None and (not matched_A[a2]):
            matched_A[a2] = True
            ALG += 1

    OPT = compute_opt_from_realization(A_size, hat_I, edges_real)
    return ALG, OPT


def simulate_many_runs_offline_statistics(
    A_size, I_size, neighbors, p, T,
    mc_trials=200,
    num_runs=20,
    seed=0,
    use_poisson_len=False
):
    """
    Main wrapper (same style as function_ei simulate_many_runs):
      1) Monte Carlo estimate f
      2) build partitions
      3) run num_runs simulations and compute ALG/OPT ratios
    Returns:
      avg_ratio, ratios
    """
    random.seed(seed)

    # ===== phase 1: estimate f =====
    r_copy_of_exp, orig_type_of_exp, copies_of_type = None, None, None
    r_copy_of_exp, copies_of_type, f_dict = estimate_f_monte_carlo(
        A_size, I_size, neighbors, p, T,
        mc_trials=mc_trials,
        seed=seed,
        use_poisson_len=use_poisson_len
    )

    # need orig_type_of_exp for partitions
    # rebuild to get orig_type_of_exp consistently
    r_copy_of_exp2, orig_type_of_exp, copies_of_type2 = build_virtual_copies(T, p)
    # sanity: sizes should match
    if len(r_copy_of_exp2) != len(r_copy_of_exp):
        r_copy_of_exp = r_copy_of_exp2
    if copies_of_type2 is not None:
        copies_of_type = copies_of_type2

    I_exp = len(r_copy_of_exp)

    # ===== phase 2: build partitions for each expanded copy =====
    Iy_bins_list = [None] * I_exp
    Iy_ends_list = [None] * I_exp
    Jy_bins_list = [None] * I_exp
    Jy_ends_list = [None] * I_exp

    for j in range(I_exp):
        Iy_bins, Iy_ends, Jy_bins, Jy_ends, r = build_partitions_for_copy(
            j=j,
            A_size=A_size,
            neighbors=neighbors,
            orig_type_of_copy=orig_type_of_exp,
            r_copy=r_copy_of_exp,
            f_dict=f_dict
        )
        Iy_bins_list[j] = Iy_bins
        Iy_ends_list[j] = Iy_ends
        Jy_bins_list[j] = Jy_bins
        Jy_ends_list[j] = Jy_ends

    # ===== simulate =====
    ratios = []
    for _ in range(num_runs):
        ALG, OPT = simulate_offline_statistics_once(
            A_size, I_size, neighbors, p, T,
            r_copy_of_exp, orig_type_of_exp, copies_of_type,
            Iy_bins_list, Iy_ends_list, Jy_bins_list, Jy_ends_list,
            use_poisson_len=use_poisson_len
        )
        if OPT > 0:
            ratios.append(ALG / OPT)

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return avg_ratio, ratios
