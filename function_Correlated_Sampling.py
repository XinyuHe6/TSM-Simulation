import bisect
import math
import random
import networkx as nx

try:
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
except ImportError as exc:  # pragma: no cover - exercised only in missing-dependency environments.
    linprog = None
    csr_matrix = None
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None


# ============================================================
# Graph generators
# ============================================================

def generate_random_graph(A_size, I_size, edge_prob):
    """
    neighbors[i] = list of advertisers a that impression type i can connect to.
    Each edge (a, i) is included independently with probability edge_prob.
    """
    neighbors = []
    for _ in range(I_size):
        neigh_i = [a for a in range(A_size) if random.random() < edge_prob]
        if not neigh_i:
            neigh_i = [random.randrange(A_size)]
        neighbors.append(neigh_i)
    return neighbors


def generate_k_regular_graph(A_size, I_size, degree):
    """
    Generate a graph where each impression type has exactly `degree`
    advertiser neighbors.
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
    if graph_mode == "random":
        return generate_random_graph(A_size, I_size, float(graph_param))
    if graph_mode == "k_regular":
        return generate_k_regular_graph(A_size, I_size, int(graph_param))
    raise ValueError(f"Unsupported graph_mode: {graph_mode}")


def random_probability_vector(I_size):
    raw = [random.random() for _ in range(I_size)]
    s = sum(raw)
    return [x / s for x in raw]


# ============================================================
# Realization + OPT helpers
# ============================================================

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


def poisson_sample(lam):
    L = math.exp(-lam)
    k = 0
    prod = 1.0
    while prod > L:
        k += 1
        prod *= random.random()
    return max(0, k - 1)


def sample_arrival_sequence(I_size, p, T, use_poisson_len=False):
    """
    Generate an arrival sequence of original types.
    - if use_poisson_len=False: exactly T arrivals
    - if True: total arrivals ~ Poisson(T)
    """
    if use_poisson_len:
        B = poisson_sample(float(T))
    else:
        B = int(T)

    return [random.choices(range(I_size), weights=p, k=1)[0] for _ in range(B)]


# ============================================================
# Natural LP for the unweighted paper algorithm
# ============================================================

def _require_scipy():
    if linprog is None or csr_matrix is None:
        raise ImportError(
            "function_Correlated_Sampling.py requires scipy to solve the LP. "
            "Install it in the active environment before running this algorithm."
        ) from SCIPY_IMPORT_ERROR


def _one_minus_exp_neg(x):
    return -math.expm1(-max(0.0, float(x)))


def _build_sparse_matrix(constraints, n_vars):
    data = []
    row_ind = []
    col_ind = []
    for row_idx, indices in enumerate(constraints):
        for col_idx in indices:
            row_ind.append(row_idx)
            col_ind.append(col_idx)
            data.append(1.0)
    return csr_matrix((data, (row_ind, col_ind)), shape=(len(constraints), n_vars))


def solve_natural_lp_unweighted(
    A_size,
    I_size,
    neighbors,
    lambdas,
    max_rounds=200,
    separation_tol=1e-9,
):
    """
    Solve the paper's natural LP for the unweighted case via cutting planes.

    Variables:
        x[i, a] for every edge (i, a).

    Constraints:
        sum_a x[i, a] <= lambda_i
        sum_{i in S} x[i, a] <= 1 - exp(-sum_{i in S} lambda_i)  for all a and S
        x[i, a] >= 0
    """
    _require_scipy()

    edge_list = []
    edge_to_var = {}
    incident_types = [[] for _ in range(A_size)]
    vars_for_type = [[] for _ in range(I_size)]

    for i in range(I_size):
        for a in neighbors[i]:
            var_idx = len(edge_list)
            edge_list.append((i, a))
            edge_to_var[(i, a)] = var_idx
            incident_types[a].append(i)
            vars_for_type[i].append(var_idx)

    n_vars = len(edge_list)
    if n_vars == 0:
        return [[0.0] * A_size for _ in range(I_size)], 0.0

    constraints = []
    bounds_rhs = []
    seen_keys = set()

    def add_constraint(indices, rhs, key):
        if key in seen_keys:
            return False
        seen_keys.add(key)
        constraints.append(tuple(sorted(set(indices))))
        bounds_rhs.append(float(rhs))
        return True

    for i in range(I_size):
        if vars_for_type[i]:
            add_constraint(vars_for_type[i], max(0.0, lambdas[i]), ("row", i))

    for a in range(A_size):
        types = sorted({i for i in incident_types[a] if lambdas[i] > separation_tol})
        if not types:
            continue

        for i in types:
            add_constraint(
                [edge_to_var[(i, a)]],
                _one_minus_exp_neg(lambdas[i]),
                ("adv_subset", a, (i,)),
            )

        add_constraint(
            [edge_to_var[(i, a)] for i in types],
            _one_minus_exp_neg(sum(lambdas[i] for i in types)),
            ("adv_subset", a, tuple(types)),
        )

    objective = [-1.0] * n_vars
    bounds = [(0.0, None)] * n_vars

    latest_solution = None
    for _round in range(max_rounds):
        A_ub = _build_sparse_matrix(constraints, n_vars)
        result = linprog(
            c=objective,
            A_ub=A_ub,
            b_ub=bounds_rhs,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"Natural LP solve failed: {result.message}")

        latest_solution = result.x
        new_cuts = []

        for a in range(A_size):
            ranked = []
            for i in incident_types[a]:
                lam_i = float(lambdas[i])
                if lam_i <= separation_tol:
                    continue
                var_idx = edge_to_var[(i, a)]
                x_ia = max(0.0, float(latest_solution[var_idx]))
                ranked.append((x_ia / lam_i, i, x_ia, lam_i))

            if not ranked:
                continue

            ranked.sort(key=lambda item: (-item[0], item[1]))

            subset = []
            sum_x = 0.0
            sum_lambda = 0.0
            best_subset = None
            best_violation = separation_tol

            for _ratio, i, x_ia, lam_i in ranked:
                subset.append(i)
                sum_x += x_ia
                sum_lambda += lam_i

                violation = sum_x - _one_minus_exp_neg(sum_lambda)
                key = ("adv_subset", a, tuple(sorted(subset)))
                if violation > best_violation and key not in seen_keys:
                    best_violation = violation
                    best_subset = tuple(sorted(subset))

            if best_subset is not None:
                new_cuts.append((a, best_subset))

        if not new_cuts:
            break

        for a, subset in new_cuts:
            add_constraint(
                [edge_to_var[(i, a)] for i in subset],
                _one_minus_exp_neg(sum(lambdas[i] for i in subset)),
                ("adv_subset", a, subset),
            )
    else:
        raise RuntimeError(
            f"Natural LP cutting-plane did not converge within {max_rounds} rounds."
        )

    x_by_type = [[0.0] * A_size for _ in range(I_size)]
    for var_idx, (i, a) in enumerate(edge_list):
        x_by_type[i][a] = max(0.0, float(latest_solution[var_idx]))

    opt_value = -float(result.fun)
    return x_by_type, opt_value


def solve_pair_approx_lp_unweighted(
    A_size,
    I_size,
    neighbors,
    lambdas,
    pair_cap=None,
):
    """
    Solve a fast approximate LP that keeps low-order advertiser-subset
    constraints from the natural LP:

        x[i, a] <= 1 - exp(-lambda_i)
        x[i, a] + x[j, a] <= 1 - exp(-(lambda_i + lambda_j))
        sum_i x[i, a] <= 1 - exp(-sum_i lambda_i)

    This is an experimental relaxation of the paper's natural LP: it avoids all
    exponential-size subsets, but preserves pairwise competition for the same
    advertiser. If pair_cap is provided, it overrides only the pairwise RHS.
    """
    _require_scipy()

    if pair_cap is not None:
        pair_cap = float(pair_cap)

    edge_list = []
    vars_for_type = [[] for _ in range(I_size)]
    incident_by_adv = [[] for _ in range(A_size)]

    for i in range(I_size):
        for a in neighbors[i]:
            var_idx = len(edge_list)
            edge_list.append((i, a))
            vars_for_type[i].append(var_idx)
            incident_by_adv[a].append((i, var_idx))

    n_vars = len(edge_list)
    if n_vars == 0:
        return [[0.0] * A_size for _ in range(I_size)], 0.0

    constraints = []
    bounds_rhs = []

    for i in range(I_size):
        if vars_for_type[i]:
            constraints.append(tuple(vars_for_type[i]))
            bounds_rhs.append(max(0.0, float(lambdas[i])))

    for a in range(A_size):
        incident = [
            (i, var_idx)
            for i, var_idx in incident_by_adv[a]
            if float(lambdas[i]) > 0.0
        ]
        if not incident:
            continue

        for i, var_idx in incident:
            constraints.append((var_idx,))
            bounds_rhs.append(_one_minus_exp_neg(lambdas[i]))

        for left_pos in range(len(incident)):
            i, var_i = incident[left_pos]
            for right_pos in range(left_pos + 1, len(incident)):
                j, var_j = incident[right_pos]
                rhs = (
                    pair_cap
                    if pair_cap is not None
                    else _one_minus_exp_neg(float(lambdas[i]) + float(lambdas[j]))
                )
                constraints.append((var_i, var_j))
                bounds_rhs.append(rhs)

        constraints.append(tuple(var_idx for _i, var_idx in incident))
        bounds_rhs.append(
            _one_minus_exp_neg(sum(float(lambdas[i]) for i, _var_idx in incident))
        )

    objective = [-1.0] * n_vars
    bounds = [(0.0, None)] * n_vars
    A_ub = _build_sparse_matrix(constraints, n_vars)
    result = linprog(
        c=objective,
        A_ub=A_ub,
        b_ub=bounds_rhs,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Pair-approx LP solve failed: {result.message}")

    x_by_type = [[0.0] * A_size for _ in range(I_size)]
    for var_idx, (i, a) in enumerate(edge_list):
        x_by_type[i][a] = max(0.0, float(result.x[var_idx]))

    opt_value = -float(result.fun)
    return x_by_type, opt_value


def solve_correlated_sampling_lp(
    A_size,
    I_size,
    neighbors,
    lambdas,
    constraint_mode="natural",
    max_rounds=200,
    separation_tol=1e-9,
    pair_cap=None,
):
    if constraint_mode == "natural":
        return solve_natural_lp_unweighted(
            A_size=A_size,
            I_size=I_size,
            neighbors=neighbors,
            lambdas=lambdas,
            max_rounds=max_rounds,
            separation_tol=separation_tol,
        )
    if constraint_mode == "pair_approx":
        return solve_pair_approx_lp_unweighted(
            A_size=A_size,
            I_size=I_size,
            neighbors=neighbors,
            lambdas=lambdas,
            pair_cap=pair_cap,
        )
    raise ValueError(f"Unsupported LP constraint mode: {constraint_mode}")


# ============================================================
# Correlated sampling distributions
# ============================================================

def _pick_from_weighted_intervals(bins, ends, total_mass, x=None):
    if total_mass <= 0.0 or not bins:
        return None
    if x is None:
        x = random.random() * total_mass
    idx = bisect.bisect_left(ends, x)
    if idx >= len(bins):
        return bins[-1]
    return bins[idx]


def _build_weighted_interval(items):
    bins = []
    ends = []
    cur = 0.0
    for item, mass in items:
        if mass <= 0.0:
            continue
        cur += mass
        bins.append(item)
        ends.append(cur)
    return bins, ends, cur


def build_correlated_sampling_distributions(
    A_size,
    I_size,
    neighbors,
    lambdas,
    x_by_type,
    tol=1e-10,
):
    """
    Build the paper's unweighted Correlated Sampling distributions D_i.

    We use the simpler heavy-neighbor description from Section 5.2 whenever
    some real advertiser has x_ij > lambda_i / 2. Otherwise we fall back to the
    wasteful half-interval construction from Section 5.1.
    """
    del A_size  # kept for symmetry with other helper signatures.

    samplers = []
    for i in range(I_size):
        lam_i = float(lambdas[i])
        masses = [(a, max(0.0, float(x_by_type[i][a]))) for a in sorted(neighbors[i])]
        total_real_mass = sum(mass for _, mass in masses)
        dummy_mass = max(0.0, lam_i - total_real_mass)

        if lam_i <= tol:
            samplers.append({"mode": "empty"})
            continue

        heavy_adv = None
        heavy_mass = -1.0
        for a, mass in masses:
            if mass > heavy_mass:
                heavy_adv = a
                heavy_mass = mass

        if heavy_adv is not None and heavy_mass > 0.5 * lam_i + tol:
            first_items = list(masses)
            if dummy_mass > tol:
                first_items.append((None, dummy_mass))
            first_bins, first_ends, first_total = _build_weighted_interval(first_items)

            second_items = [(a, mass) for a, mass in masses if a != heavy_adv and mass > tol]
            if dummy_mass > tol:
                second_items.append((None, dummy_mass))
            second_bins, second_ends, second_total = _build_weighted_interval(second_items)

            samplers.append(
                {
                    "mode": "correlated",
                    "heavy_adv": heavy_adv,
                    "lambda": lam_i,
                    "first_bins": first_bins,
                    "first_ends": first_ends,
                    "first_total": first_total,
                    "second_bins": second_bins,
                    "second_ends": second_ends,
                    "second_total": second_total,
                }
            )
            continue

        interval_items = list(masses)
        if dummy_mass > tol:
            interval_items.append((None, dummy_mass))
        bins, ends, total = _build_weighted_interval(interval_items)
        samplers.append(
            {
                "mode": "wasteful",
                "lambda": lam_i,
                "bins": bins,
                "ends": ends,
                "total": total,
            }
        )

    return samplers


def sample_pair_from_distribution(sampler):
    mode = sampler["mode"]
    if mode == "empty":
        return None, None

    if mode == "wasteful":
        lam_i = sampler["lambda"]
        nu = random.random() * lam_i
        nu_prime = nu + 0.5 * lam_i
        if nu_prime >= lam_i:
            nu_prime -= lam_i
        first = _pick_from_weighted_intervals(
            sampler["bins"],
            sampler["ends"],
            sampler["total"],
            x=nu,
        )
        second = _pick_from_weighted_intervals(
            sampler["bins"],
            sampler["ends"],
            sampler["total"],
            x=nu_prime,
        )
        return first, second

    if mode == "correlated":
        first = _pick_from_weighted_intervals(
            sampler["first_bins"],
            sampler["first_ends"],
            sampler["first_total"],
        )
        if first != sampler["heavy_adv"]:
            return first, sampler["heavy_adv"]

        second = _pick_from_weighted_intervals(
            sampler["second_bins"],
            sampler["second_ends"],
            sampler["second_total"],
        )
        return first, second

    raise ValueError(f"Unsupported sampler mode: {mode}")


# ============================================================
# Simulation wrappers
# ============================================================

def simulate_correlated_sampling_once(
    A_size,
    I_size,
    neighbors,
    p,
    T,
    samplers,
    use_poisson_len=False,
):
    matched_A = [False] * A_size
    ALG = 0

    hat_I = []
    edges_real = []

    arrivals = sample_arrival_sequence(I_size, p, T, use_poisson_len=use_poisson_len)

    for i in arrivals:
        imp_id = len(hat_I)
        hat_I.append(imp_id)

        for a in neighbors[i]:
            edges_real.append((a, imp_id))

        a1, a2 = sample_pair_from_distribution(samplers[i])

        if a1 is not None and not matched_A[a1]:
            matched_A[a1] = True
            ALG += 1
            continue

        if a2 is not None and not matched_A[a2]:
            matched_A[a2] = True
            ALG += 1

    OPT = compute_opt_from_realization(A_size, hat_I, edges_real)
    return ALG, OPT


def prepare_correlated_sampling_state(
    A_size,
    I_size,
    neighbors,
    p,
    T,
    lp_max_rounds=200,
    lp_separation_tol=1e-9,
    lp_constraint_mode="natural",
    lp_pair_cap=None,
):
    lambdas = [float(T) * float(pi) for pi in p]
    x_by_type, _ = solve_correlated_sampling_lp(
        A_size=A_size,
        I_size=I_size,
        neighbors=neighbors,
        lambdas=lambdas,
        constraint_mode=lp_constraint_mode,
        max_rounds=lp_max_rounds,
        separation_tol=lp_separation_tol,
        pair_cap=lp_pair_cap,
    )
    samplers = build_correlated_sampling_distributions(
        A_size=A_size,
        I_size=I_size,
        neighbors=neighbors,
        lambdas=lambdas,
        x_by_type=x_by_type,
    )
    return {"samplers": samplers}


def simulate_correlated_sampling_on_arrivals(A_size, neighbors, arrivals, state):
    matched_A = [False] * A_size
    ALG = 0
    samplers = state["samplers"]

    for i in arrivals:
        a1, a2 = sample_pair_from_distribution(samplers[i])

        if a1 is not None and not matched_A[a1]:
            matched_A[a1] = True
            ALG += 1
            continue

        if a2 is not None and not matched_A[a2]:
            matched_A[a2] = True
            ALG += 1

    return ALG


def simulate_many_runs_correlated_sampling(
    A_size,
    I_size,
    neighbors,
    p,
    T,
    num_runs=20,
    seed=0,
    use_poisson_len=False,
    lp_max_rounds=200,
    lp_separation_tol=1e-9,
    lp_constraint_mode="natural",
    lp_pair_cap=None,
):
    """
    Main wrapper for the unweighted Correlated Sampling algorithm.

    We set the arrival rates to lambda_i = T * p_i, solve either the paper's
    natural LP or the experimental pair-approx LP, construct the pair-sampling
    distributions D_i, and then simulate the online policy against offline OPT.
    """
    random.seed(seed)

    state = prepare_correlated_sampling_state(
        A_size=A_size,
        I_size=I_size,
        neighbors=neighbors,
        p=p,
        T=T,
        lp_max_rounds=lp_max_rounds,
        lp_separation_tol=lp_separation_tol,
        lp_constraint_mode=lp_constraint_mode,
        lp_pair_cap=lp_pair_cap,
    )

    ratios = []
    for _ in range(num_runs):
        ALG, OPT = simulate_correlated_sampling_once(
            A_size=A_size,
            I_size=I_size,
            neighbors=neighbors,
            p=p,
            T=T,
            samplers=state["samplers"],
            use_poisson_len=use_poisson_len,
        )
        if OPT > 0:
            ratios.append(ALG / OPT)

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return avg_ratio, ratios
