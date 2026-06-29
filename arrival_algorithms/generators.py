import math
import random


ARRIVAL_SPECS = {
    "fixed": {
        "label": "IID Fixed Length",
    },
    "poisson": {
        "label": "IID Poisson Length",
    },
}

ARRIVAL_ALIASES = {
    "fixed": "fixed",
    "fixed_length": "fixed",
    "exact": "fixed",
    "iid": "fixed",
    "iid_fixed": "fixed",
    "poisson": "poisson",
    "poisson_len": "poisson",
    "poisson_length": "poisson",
    "iid_poisson": "poisson",
}


def canonical_arrival_name(name):
    key = str(name).strip().lower()
    canonical = ARRIVAL_ALIASES.get(key)
    if canonical is None:
        valid = ", ".join(sorted(ARRIVAL_SPECS))
        raise ValueError(f"Unknown arrival algorithm '{name}'. Valid options: {valid}.")
    return canonical


def poisson_sample(lam):
    """
    Knuth sampler for Poisson(lam).
    """
    L = math.exp(-float(lam))
    k = 0
    prod = 1.0
    while prod > L:
        k += 1
        prod *= random.random()
    return max(0, k - 1)


def random_probability_vector(I_size):
    """
    Generate a random probability vector over impression types 0..I_size-1.
    """
    raw = [random.random() for _ in range(I_size)]
    total = sum(raw)
    if total <= 0.0:
        return [1.0 / I_size for _ in range(I_size)]
    return [value / total for value in raw]


def validate_arrival_counts(e):
    """
    Validate and return integer expected arrival counts e_i.
    """
    counts = [int(value) for value in e]
    if any(value < 0 for value in counts):
        raise ValueError("arrival counts e_i must be non-negative.")
    return counts


def arrival_probabilities_from_counts(e):
    """
    Convert integer expected arrival counts e_i into probabilities p_i=e_i/sum(e).
    """
    counts = validate_arrival_counts(e)
    total = sum(counts)
    if total <= 0:
        return [0.0 for _ in counts]
    return [float(value) / float(total) for value in counts]


def integerize_probability_vector(T, p):
    """
    Convert probabilities p_i into integer expected counts e_i summing to T.

    This is only a compatibility bridge for old callers. New experiment flow
    should generate or pass e_i first, then derive p_i=e_i/sum(e).
    """
    total_arrivals = int(T)
    if total_arrivals < 0:
        raise ValueError("T must be non-negative.")
    if not p:
        return []

    total_prob = float(sum(p))
    if total_prob <= 0.0:
        return [0 for _ in p]

    normalized = [float(value) / total_prob for value in p]
    expected = [total_arrivals * value for value in normalized]
    counts = [math.floor(value) for value in expected]
    remaining = total_arrivals - sum(counts)

    order = sorted(
        range(len(p)),
        key=lambda i: (expected[i] - counts[i], normalized[i]),
        reverse=True,
    )
    for i in order[:remaining]:
        counts[i] += 1

    return counts


def random_arrival_counts(I_size, T):
    """
    Generate a random integer arrival distribution e_i with sum_i e_i = T.
    """
    return integerize_probability_vector(T, random_probability_vector(I_size))


def sample_arrival_sequence(name, I_size=None, p=None, T=None, e=None, **params):
    """
    Unified arrival-generation interface.

    Returns:
        arrivals: list[int], where each value is an impression type index.
    """
    del params
    if e is not None:
        counts = validate_arrival_counts(e)
        weights = counts
        expected_length = sum(counts)
        if I_size is None:
            I_size = len(counts)
    else:
        if p is None or T is None:
            raise ValueError("sample_arrival_sequence requires either e or both p and T.")
        counts = integerize_probability_vector(T, p)
        weights = counts
        expected_length = sum(counts)

    if I_size is None:
        raise ValueError("I_size is required when p is used.")

    canonical = canonical_arrival_name(name)
    if canonical == "fixed":
        length = int(expected_length)
    elif canonical == "poisson":
        length = poisson_sample(float(expected_length))
    else:
        raise AssertionError(f"Unhandled arrival algorithm: {canonical}")

    if length <= 0:
        return []

    if sum(weights) <= 0:
        return []

    return [
        random.choices(range(I_size), weights=weights, k=1)[0]
        for _ in range(length)
    ]
