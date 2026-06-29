from .manshadi_common import prepare_state_for_try_count, run_with_try_count


TRY_COUNT = 2

SPEC = {
    "csv": "manshadi_2try",
    "label": "Manshadi (2 tries)",
    "marker": "^",
}

ALIASES = (
    "offline_statistics",
    "offline_statistics_2try",
    "manshadi",
    "manshadi2",
    "manshadi_2try",
)


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
