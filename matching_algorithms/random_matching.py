import random


SPEC = {
    "csv": "random_matching",
    "label": "Random Matching",
    "marker": "x",
}

ALIASES = ("random", "random_greedy", "random_matching")


def run(A_size, I_size, neighbors, arrivals, state=None):
    del I_size, state
    matched_A = [False] * A_size
    alg = 0

    for i in arrivals:
        candidates = [a for a in neighbors[i] if not matched_A[a]]
        if candidates:
            a = random.choice(candidates)
            matched_A[a] = True
            alg += 1

    return alg
