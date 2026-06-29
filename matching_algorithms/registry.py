from . import (
    brubach_vw,
    correlated_sampling,
    degree_matching,
    fluid_lp,
    manshadi2,
    manshadi3,
    manshadi4,
    random_matching,
    tsm,
)
from arrival_algorithms import arrival_probabilities_from_counts, integerize_probability_vector
from .common import MatchingResult, advertiser_degrees, compute_offline_opt
from .manshadi_common import prepare_group_state


MATCHING_MODULES = {
    "random_matching": random_matching,
    "degree_matching": degree_matching,
    "tsm": tsm,
    "brubach_vw": brubach_vw,
    "manshadi2": manshadi2,
    "manshadi3": manshadi3,
    "manshadi4": manshadi4,
    "correlated_sampling": correlated_sampling,
    "fluid_lp": fluid_lp,
}

MATCHING_SPECS = {
    name: module.SPEC
    for name, module in MATCHING_MODULES.items()
}

MATCHING_ALIASES = {}
for name, module in MATCHING_MODULES.items():
    for alias in module.ALIASES:
        MATCHING_ALIASES[alias] = name


def canonical_matching_name(name):
    key = str(name).strip().lower()
    canonical = MATCHING_ALIASES.get(key)
    if canonical is None:
        valid = ", ".join(sorted(MATCHING_SPECS))
        raise ValueError(f"Unknown matching algorithm '{name}'. Valid options: {valid}.")
    return canonical


def canonicalize_matching_algorithms(raw_algorithms):
    if isinstance(raw_algorithms, str):
        parts = raw_algorithms.split(",")
    else:
        parts = list(raw_algorithms)

    algorithms = []
    seen = set()
    for part in parts:
        name = str(part).strip()
        if not name:
            continue
        canonical = canonical_matching_name(name)
        if canonical not in seen:
            algorithms.append(canonical)
            seen.add(canonical)

    if not algorithms:
        raise ValueError("At least one matching algorithm is required.")
    return algorithms


def manshadi_try_count(algorithm):
    canonical = canonical_matching_name(algorithm)
    if not canonical.startswith("manshadi"):
        raise ValueError(f"Algorithm '{algorithm}' is not a Manshadi variant.")
    return int(canonical[len("manshadi"):])


def prepare_matching_state(
    algorithm,
    A_size,
    I_size,
    neighbors,
    p=None,
    T=None,
    e=None,
    seed=0,
    arrival_algorithm="fixed",
    mc_trials=200,
    corr_lp_max_rounds=200,
    corr_lp_separation_tol=1e-9,
    corr_lp_constraint_mode="natural",
    corr_lp_pair_cap=None,
):
    states = prepare_matching_states(
        algorithms=[algorithm],
        A_size=A_size,
        I_size=I_size,
        neighbors=neighbors,
        p=p,
        T=T,
        e=e,
        seed=seed,
        arrival_algorithm=arrival_algorithm,
        mc_trials=mc_trials,
        corr_lp_max_rounds=corr_lp_max_rounds,
        corr_lp_separation_tol=corr_lp_separation_tol,
        corr_lp_constraint_mode=corr_lp_constraint_mode,
        corr_lp_pair_cap=corr_lp_pair_cap,
    )
    return states[canonical_matching_name(algorithm)]


def prepare_matching_states(
    algorithms,
    A_size,
    I_size,
    neighbors,
    p=None,
    T=None,
    e=None,
    seed=0,
    arrival_algorithm="fixed",
    mc_trials=200,
    corr_lp_max_rounds=200,
    corr_lp_separation_tol=1e-9,
    corr_lp_constraint_mode="natural",
    corr_lp_pair_cap=None,
):
    """
    Precompute reusable per-graph state for selected matching algorithms.
    """
    algorithms = canonicalize_matching_algorithms(algorithms)
    states = {algorithm: None for algorithm in algorithms}

    if e is None:
        if p is None or T is None:
            raise ValueError("prepare_matching_states requires e or both p and T.")
        e = integerize_probability_vector(T, p)
    else:
        e = [int(value) for value in e]

    T = sum(e) if T is None else int(T)
    if T != sum(e):
        raise ValueError("T must equal sum(e) when e is provided.")
    p = arrival_probabilities_from_counts(e)

    prepare_kwargs = {
        "A_size": A_size,
        "I_size": I_size,
        "neighbors": neighbors,
        "p": p,
        "T": T,
        "e": e,
        "seed": seed,
        "arrival_algorithm": arrival_algorithm,
        "mc_trials": mc_trials,
        "corr_lp_max_rounds": corr_lp_max_rounds,
        "corr_lp_separation_tol": corr_lp_separation_tol,
        "corr_lp_constraint_mode": corr_lp_constraint_mode,
        "corr_lp_pair_cap": corr_lp_pair_cap,
    }

    manshadi_algorithms = [
        algorithm for algorithm in algorithms if algorithm.startswith("manshadi")
    ]
    if manshadi_algorithms:
        max_tries = max(manshadi_try_count(algorithm) for algorithm in manshadi_algorithms)
        manshadi_state = prepare_group_state(
            max_tries=max_tries,
            **prepare_kwargs,
        )
        for algorithm in manshadi_algorithms:
            states[algorithm] = manshadi_state

    for algorithm in algorithms:
        if algorithm.startswith("manshadi"):
            continue
        module = MATCHING_MODULES[algorithm]
        prepare_state = getattr(module, "prepare_state", None)
        if prepare_state is not None:
            states[algorithm] = prepare_state(**prepare_kwargs)

    return states


def run_matching(algorithm, A_size, I_size, neighbors, arrivals, state=None):
    """
    Unified matching interface.

    Inputs:
        algorithm: matching algorithm name or alias.
        neighbors: graph as list[list[int]].
        arrivals: realized arrival sequence as list[int].
        state: optional precomputed state from prepare_matching_states.

    Returns:
        MatchingResult with the matched advertiser count in result.alg.
    """
    canonical = canonical_matching_name(algorithm)
    module = MATCHING_MODULES[canonical]
    alg = module.run(
        A_size=A_size,
        I_size=I_size,
        neighbors=neighbors,
        arrivals=arrivals,
        state=state,
    )
    return MatchingResult(algorithm=canonical, alg=int(alg))
