from arrival_algorithms import canonical_arrival_name

from .common import require_state


def prepare_group_state(
    A_size,
    I_size,
    neighbors,
    p,
    T,
    max_tries,
    seed=0,
    arrival_algorithm="fixed",
    mc_trials=200,
    **kwargs,
):
    del kwargs
    from function_offline_statistics_multiple import (
        prepare_offline_statistics_multi_state,
    )

    use_poisson_len = canonical_arrival_name(arrival_algorithm) == "poisson"
    return prepare_offline_statistics_multi_state(
        A_size=A_size,
        I_size=I_size,
        neighbors=neighbors,
        p=p,
        T=T,
        max_tries=max_tries,
        mc_trials=mc_trials,
        seed=seed,
        use_poisson_len=use_poisson_len,
    )


def prepare_state_for_try_count(
    A_size,
    I_size,
    neighbors,
    p,
    T,
    try_count,
    seed=0,
    arrival_algorithm="fixed",
    mc_trials=200,
    **kwargs,
):
    return prepare_group_state(
        A_size=A_size,
        I_size=I_size,
        neighbors=neighbors,
        p=p,
        T=T,
        max_tries=try_count,
        seed=seed,
        arrival_algorithm=arrival_algorithm,
        mc_trials=mc_trials,
        **kwargs,
    )


def run_with_try_count(A_size, arrivals, state, try_count):
    require_state(f"manshadi{try_count}", state)
    from function_offline_statistics_multiple import simulate_offline_statistics_k_on_arrivals

    return simulate_offline_statistics_k_on_arrivals(
        A_size,
        arrivals,
        state,
        try_count,
    )
