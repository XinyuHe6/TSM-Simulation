from .manshadi_common import prepare_state_for_try_count, run_with_try_count


TRY_COUNT = 3

SPEC = {
    "csv": "manshadi_3try",
    "label": "Manshadi (3 tries)",
    "marker": "D",
}

ALIASES = ("manshadi3", "manshadi_3try")


def prepare_state(A_size, I_size, neighbors, p, T, **kwargs):
    return prepare_state_for_try_count(
        A_size,
        I_size,
        neighbors,
        p,
        T,
        TRY_COUNT,
        **kwargs,
    )


def run(A_size, I_size, neighbors, arrivals, state=None):
    del I_size, neighbors
    return run_with_try_count(A_size, arrivals, state, TRY_COUNT)
