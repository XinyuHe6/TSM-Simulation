from .common import require_state


SPEC = {
    "csv": "correlated_sampling",
    "label": "Correlated Sampling",
    "marker": "P",
}

ALIASES = ("correlated", "correlated_sampling", "corr")


def prepare_state(
    A_size,
    I_size,
    neighbors,
    p,
    T,
    corr_lp_max_rounds=200,
    corr_lp_separation_tol=1e-9,
    corr_lp_constraint_mode="natural",
    corr_lp_pair_cap=None,
    **kwargs,
):
    del kwargs
    from .correlated_sampling_core import prepare_correlated_sampling_state

    return prepare_correlated_sampling_state(
        A_size=A_size,
        I_size=I_size,
        neighbors=neighbors,
        p=p,
        T=T,
        lp_max_rounds=corr_lp_max_rounds,
        lp_separation_tol=corr_lp_separation_tol,
        lp_constraint_mode=corr_lp_constraint_mode,
        lp_pair_cap=corr_lp_pair_cap,
    )


def run(A_size, I_size, neighbors, arrivals, state=None):
    del I_size
    require_state("correlated_sampling", state)
    from .correlated_sampling_core import simulate_correlated_sampling_on_arrivals

    return simulate_correlated_sampling_on_arrivals(
        A_size,
        neighbors,
        arrivals,
        state,
    )
