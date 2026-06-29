from .common import advertiser_degrees


SPEC = {
    "csv": "degree_matching",
    "label": "Degree Matching",
    "marker": "o",
}

ALIASES = ("degree", "degree_matching")


def prepare_state(A_size, I_size, neighbors, p, T, **kwargs):
    del I_size, p, T, kwargs
    return {"adv_degrees": advertiser_degrees(A_size, neighbors)}


def run(A_size, I_size, neighbors, arrivals, state=None):
    del I_size
    if state is None:
        state = {"adv_degrees": advertiser_degrees(A_size, neighbors)}

    adv_degrees = state["adv_degrees"]
    matched_A = [False] * A_size
    alg = 0

    for i in arrivals:
        candidates = [a for a in neighbors[i] if not matched_A[a]]
        if candidates:
            a = min(candidates, key=lambda value: adv_degrees[value])
            matched_A[a] = True
            alg += 1

    return alg
