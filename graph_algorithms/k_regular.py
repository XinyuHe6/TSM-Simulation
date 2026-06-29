import random


SPEC = {
    "csv": "k_regular",
    "label": "k-Regular",
    "required": ("degree",),
}

ALIASES = ("k_regular", "k-regular", "regular")


def generate(A_size, I_size, degree):
    """
    Graph where every impression type has exactly `degree` neighbors.

    If A_size == I_size, this uses a random union of cyclic perfect matchings,
    so both sides have degree `degree`. Otherwise it keeps the one-sided
    convention used in the original codebase: every impression type has
    `degree` distinct advertiser neighbors.
    """
    degree = int(degree)
    if degree < 0 or degree > A_size:
        raise ValueError(f"regular degree must satisfy 0 <= k <= {A_size}, got {degree}.")

    if A_size == I_size:
        shifts = random.sample(range(A_size), degree)
        return [
            sorted((i + shift) % A_size for shift in shifts)
            for i in range(I_size)
        ]

    advertisers = list(range(A_size))
    return [
        sorted(random.sample(advertisers, degree))
        for _ in range(I_size)
    ]
