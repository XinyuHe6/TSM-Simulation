import random

from .common import require_state


SPEC = {
    "csv": "tsm",
    "label": "TSM",
    "marker": "s",
}

ALIASES = ("tsm",)


def prepare_state(A_size, I_size, neighbors, p=None, T=None, e=None, **kwargs):
    del kwargs
    from arrival_algorithms import integerize_probability_vector, validate_arrival_counts
    from .tsm_core import compute_blue_red_matchings

    if e is None:
        if p is None or T is None:
            raise ValueError("TSM prepare_state requires e or both p and T.")
        e = integerize_probability_vector(T, p)
    else:
        e = validate_arrival_counts(e)

    blue_for_copy, red_for_copy, copies_of_type = compute_blue_red_matchings(
        A_size,
        I_size,
        neighbors,
        e,
    )
    return {
        "e": e,
        "blue_for_copy": blue_for_copy,
        "red_for_copy": red_for_copy,
        "copies_of_type": copies_of_type,
    }


def run(A_size, I_size, neighbors, arrivals, state=None):
    del I_size, neighbors
    require_state("tsm", state)

    matched_A = [False] * A_size
    alg = 0

    e = state["e"]
    blue_for_copy = state["blue_for_copy"]
    red_for_copy = state["red_for_copy"]
    copies_of_type = state["copies_of_type"]
    x_copy = [0] * len(blue_for_copy)

    for i in arrivals:
        if e[i] <= 0 or not copies_of_type[i]:
            continue

        j = random.choice(copies_of_type[i])
        if x_copy[j] == 0:
            a = blue_for_copy[j]
        elif x_copy[j] == 1:
            a = red_for_copy[j]
        else:
            a = None

        if a is not None and not matched_A[a]:
            matched_A[a] = True
            alg += 1

        x_copy[j] += 1

    return alg
