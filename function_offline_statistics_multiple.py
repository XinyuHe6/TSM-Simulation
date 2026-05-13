import random

from function_offline_statistics import (
    build_virtual_copies,
    compute_opt_from_realization,
    estimate_f_monte_carlo,
    pick_from_partition,
    sample_arrival_sequence,
)


def _build_shifted_partition(items, f_dummy, r, shift):
    """
    Build one shifted partition for the multiple-try Manshadi extension.

    shift = 0 gives Iy:
      a1, a2, ..., dummy
    shift = 1 gives Jy:
      a2, a3, ..., dummy, a1
    shift = 2 gives:
      a3, a4, ..., dummy, a1, a2
    """
    if not items:
        return [None], [r]

    n = len(items)
    shift = max(0, min(int(shift), n))

    prefix = items[shift:]
    suffix = items[:shift]

    bins = []
    ends = []
    cur = 0.0

    for a, val in prefix:
        cur += val
        bins.append(a)
        ends.append(cur)

    cur += f_dummy
    bins.append(None)
    ends.append(cur)

    for a, val in suffix:
        cur += val
        bins.append(a)
        ends.append(cur)

    if ends:
        ends[-1] = r
    return bins, ends


def _build_shifted_partitions_for_copy(
    j,
    neighbors,
    orig_type_of_copy,
    r_copy,
    f_dict,
    max_tries,
):
    i = orig_type_of_copy[j]
    r = r_copy[j]
    nbrs = neighbors[i]

    items = [(a, f_dict.get((j, a), 0.0)) for a in nbrs]
    items.sort(key=lambda item: item[1], reverse=True)

    sum_f = sum(val for _, val in items)
    if sum_f > r and sum_f > 0:
        scale = r / sum_f
        items = [(a, val * scale) for a, val in items]
        sum_f = r

    f_dummy = max(0.0, r - sum_f)
    base_bins, base_ends = _build_shifted_partition(items, f_dummy, r, shift=0)

    if not items or items[0][1] <= 0:
        return [(base_bins[:], base_ends[:]) for _ in range(max_tries)]

    return [
        _build_shifted_partition(items, f_dummy, r, shift=shift)
        for shift in range(max_tries)
    ]


def _build_all_shifted_partitions(
    neighbors,
    r_copy_of_exp,
    orig_type_of_exp,
    f_dict,
    max_tries,
):
    bins_by_try = [[] for _ in range(max_tries)]
    ends_by_try = [[] for _ in range(max_tries)]

    for j in range(len(r_copy_of_exp)):
        partitions = _build_shifted_partitions_for_copy(
            j=j,
            neighbors=neighbors,
            orig_type_of_copy=orig_type_of_exp,
            r_copy=r_copy_of_exp,
            f_dict=f_dict,
            max_tries=max_tries,
        )
        for try_index, (bins, ends) in enumerate(partitions):
            bins_by_try[try_index].append(bins)
            ends_by_try[try_index].append(ends)

    return bins_by_try, ends_by_try


def simulate_offline_statistics_k_once(
    A_size,
    I_size,
    neighbors,
    p,
    T,
    r_copy_of_exp,
    copies_of_type,
    bins_by_try,
    ends_by_try,
    try_count,
    use_poisson_len=False,
):
    matched_A = [False] * A_size
    ALG = 0

    hat_I = []
    edges_real = []

    arrivals = sample_arrival_sequence(I_size, p, T, use_poisson_len=use_poisson_len)

    for i in arrivals:
        imp_id = len(hat_I)
        hat_I.append(imp_id)

        for a in neighbors[i]:
            edges_real.append((a, imp_id))

        if copies_of_type[i]:
            j = random.choice(copies_of_type[i])
        else:
            continue

        r = r_copy_of_exp[j]
        if r <= 0:
            continue

        x = random.random() * r

        for try_index in range(try_count):
            a = pick_from_partition(
                bins_by_try[try_index][j],
                ends_by_try[try_index][j],
                x,
            )
            if a is not None and (not matched_A[a]):
                matched_A[a] = True
                ALG += 1
                break

    OPT = compute_opt_from_realization(A_size, hat_I, edges_real)
    return ALG, OPT


def prepare_offline_statistics_multi_state(
    A_size,
    I_size,
    neighbors,
    p,
    T,
    max_tries,
    mc_trials=200,
    seed=0,
    use_poisson_len=False,
):
    r_copy_of_exp, copies_of_type, f_dict = estimate_f_monte_carlo(
        A_size,
        I_size,
        neighbors,
        p,
        T,
        mc_trials=mc_trials,
        seed=seed,
        use_poisson_len=use_poisson_len,
    )

    r_copy_of_exp2, orig_type_of_exp, copies_of_type2 = build_virtual_copies(T, p)
    if len(r_copy_of_exp2) != len(r_copy_of_exp):
        r_copy_of_exp = r_copy_of_exp2
    if copies_of_type2 is not None:
        copies_of_type = copies_of_type2

    bins_by_try, ends_by_try = _build_all_shifted_partitions(
        neighbors=neighbors,
        r_copy_of_exp=r_copy_of_exp,
        orig_type_of_exp=orig_type_of_exp,
        f_dict=f_dict,
        max_tries=max_tries,
    )

    return {
        "r_copy_of_exp": r_copy_of_exp,
        "copies_of_type": copies_of_type,
        "bins_by_try": bins_by_try,
        "ends_by_try": ends_by_try,
    }


def simulate_offline_statistics_k_on_arrivals(
    A_size,
    arrivals,
    state,
    try_count,
):
    matched_A = [False] * A_size
    ALG = 0

    r_copy_of_exp = state["r_copy_of_exp"]
    copies_of_type = state["copies_of_type"]
    bins_by_try = state["bins_by_try"]
    ends_by_try = state["ends_by_try"]

    for i in arrivals:
        if copies_of_type[i]:
            j = random.choice(copies_of_type[i])
        else:
            continue

        r = r_copy_of_exp[j]
        if r <= 0:
            continue

        x = random.random() * r

        for try_index in range(try_count):
            a = pick_from_partition(
                bins_by_try[try_index][j],
                ends_by_try[try_index][j],
                x,
            )
            if a is not None and (not matched_A[a]):
                matched_A[a] = True
                ALG += 1
                break

    return ALG


def simulate_many_runs_offline_statistics_k(
    A_size,
    I_size,
    neighbors,
    p,
    T,
    try_count,
    mc_trials=200,
    num_runs=20,
    seed=0,
    use_poisson_len=False,
):
    values = simulate_many_runs_offline_statistics_multi(
        A_size=A_size,
        I_size=I_size,
        neighbors=neighbors,
        p=p,
        T=T,
        try_counts=[try_count],
        mc_trials=mc_trials,
        num_runs=num_runs,
        seed=seed,
        use_poisson_len=use_poisson_len,
    )
    return values[try_count]


def simulate_many_runs_offline_statistics_multi(
    A_size,
    I_size,
    neighbors,
    p,
    T,
    try_counts,
    mc_trials=200,
    num_runs=20,
    seed=0,
    use_poisson_len=False,
):
    try_counts = sorted(set(int(k) for k in try_counts))
    if not try_counts:
        return {}
    if try_counts[0] <= 0:
        raise ValueError("try counts must be positive integers.")

    max_tries = max(try_counts)

    r_copy_of_exp, copies_of_type, f_dict = estimate_f_monte_carlo(
        A_size,
        I_size,
        neighbors,
        p,
        T,
        mc_trials=mc_trials,
        seed=seed,
        use_poisson_len=use_poisson_len,
    )

    r_copy_of_exp2, orig_type_of_exp, copies_of_type2 = build_virtual_copies(T, p)
    if len(r_copy_of_exp2) != len(r_copy_of_exp):
        r_copy_of_exp = r_copy_of_exp2
    if copies_of_type2 is not None:
        copies_of_type = copies_of_type2

    bins_by_try, ends_by_try = _build_all_shifted_partitions(
        neighbors=neighbors,
        r_copy_of_exp=r_copy_of_exp,
        orig_type_of_exp=orig_type_of_exp,
        f_dict=f_dict,
        max_tries=max_tries,
    )

    results = {}
    for try_count in try_counts:
        ratios = []
        random.seed(seed + 10000)
        for _ in range(num_runs):
            ALG, OPT = simulate_offline_statistics_k_once(
                A_size=A_size,
                I_size=I_size,
                neighbors=neighbors,
                p=p,
                T=T,
                r_copy_of_exp=r_copy_of_exp,
                copies_of_type=copies_of_type,
                bins_by_try=bins_by_try,
                ends_by_try=ends_by_try,
                try_count=try_count,
                use_poisson_len=use_poisson_len,
            )
            if OPT > 0:
                ratios.append(ALG / OPT)

        avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        results[try_count] = (avg_ratio, ratios)

    return results
