import random

from function_offline_statistics import (
    generate_random_graph,
    generate_k_regular_graph,
    generate_graph,
    random_probability_vector,
    compute_opt_from_realization,
    build_virtual_copies,
    sample_arrival_sequence,
    estimate_f_monte_carlo,
    pick_from_partition,
)


def _build_shifted_partition(items, f_dummy, r, shift):
    """
    Build a shifted partition of advertiser chunks plus the dummy chunk.

    items are sorted in descending f-value order and contain only real advertisers.
    shift = 0 recovers Iy.
    shift = 1 recovers Jy from function_offline_statistics.py.
    shift = 2 is the natural third-attempt extension used here.
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


def build_partitions_for_copy_three(j, A_size, neighbors, orig_type_of_copy, r_copy, f_dict):
    """
    Same as the 2-try offline statistics partition builder, but also creates
    a third shifted partition for a third matching attempt.
    """
    del A_size  # kept for signature symmetry with the original helper.

    i = orig_type_of_copy[j]
    r = r_copy[j]
    nbrs = neighbors[i]

    items = [(a, f_dict.get((j, a), 0.0)) for a in nbrs]
    items.sort(key=lambda t: t[1], reverse=True)

    sum_f = sum(val for _, val in items)
    if sum_f > r and sum_f > 0:
        scale = r / sum_f
        items = [(a, val * scale) for a, val in items]
        sum_f = r

    f_dummy = max(0.0, r - sum_f)

    Iy_bins, Iy_ends = _build_shifted_partition(items, f_dummy, r, shift=0)
    if items and items[0][1] > 0:
        Jy_bins, Jy_ends = _build_shifted_partition(items, f_dummy, r, shift=1)
        Ky_bins, Ky_ends = _build_shifted_partition(items, f_dummy, r, shift=2)
    else:
        Jy_bins, Jy_ends = Iy_bins[:], Iy_ends[:]
        Ky_bins, Ky_ends = Iy_bins[:], Iy_ends[:]

    return Iy_bins, Iy_ends, Jy_bins, Jy_ends, Ky_bins, Ky_ends, r


def simulate_offline_statistics3_once(
    A_size,
    I_size,
    neighbors,
    p,
    T,
    r_copy_of_exp,
    orig_type_of_exp,
    copies_of_type,
    Iy_bins_list,
    Iy_ends_list,
    Jy_bins_list,
    Jy_ends_list,
    Ky_bins_list,
    Ky_ends_list,
    use_poisson_len=False,
):
    """
    One run of the 3-try extension:
      - same realization model as function_offline_statistics
      - same first two attempts
      - one extra third attempt from a third shifted partition
    Returns: ALG, OPT
    """
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

        a1 = pick_from_partition(Iy_bins_list[j], Iy_ends_list[j], x)
        if a1 is not None and (not matched_A[a1]):
            matched_A[a1] = True
            ALG += 1
            continue

        a2 = pick_from_partition(Jy_bins_list[j], Jy_ends_list[j], x)
        if a2 is not None and (not matched_A[a2]):
            matched_A[a2] = True
            ALG += 1
            continue

        a3 = pick_from_partition(Ky_bins_list[j], Ky_ends_list[j], x)
        if a3 is not None and (not matched_A[a3]):
            matched_A[a3] = True
            ALG += 1

    OPT = compute_opt_from_realization(A_size, hat_I, edges_real)
    return ALG, OPT


def simulate_many_runs_offline_statistics3(
    A_size,
    I_size,
    neighbors,
    p,
    T,
    mc_trials=200,
    num_runs=20,
    seed=0,
    use_poisson_len=False,
):
    """
    3-try extension of simulate_many_runs_offline_statistics().

    Assumption for the extension:
      - keep the same f-estimation and copy construction
      - keep the original first two shifted partitions
      - add a third partition by shifting one more advertiser chunk to the end
    """
    random.seed(seed)

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

    I_exp = len(r_copy_of_exp)

    Iy_bins_list = [None] * I_exp
    Iy_ends_list = [None] * I_exp
    Jy_bins_list = [None] * I_exp
    Jy_ends_list = [None] * I_exp
    Ky_bins_list = [None] * I_exp
    Ky_ends_list = [None] * I_exp

    for j in range(I_exp):
        Iy_bins, Iy_ends, Jy_bins, Jy_ends, Ky_bins, Ky_ends, _ = build_partitions_for_copy_three(
            j=j,
            A_size=A_size,
            neighbors=neighbors,
            orig_type_of_copy=orig_type_of_exp,
            r_copy=r_copy_of_exp,
            f_dict=f_dict,
        )
        Iy_bins_list[j] = Iy_bins
        Iy_ends_list[j] = Iy_ends
        Jy_bins_list[j] = Jy_bins
        Jy_ends_list[j] = Jy_ends
        Ky_bins_list[j] = Ky_bins
        Ky_ends_list[j] = Ky_ends

    ratios = []
    for _ in range(num_runs):
        ALG, OPT = simulate_offline_statistics3_once(
            A_size,
            I_size,
            neighbors,
            p,
            T,
            r_copy_of_exp,
            orig_type_of_exp,
            copies_of_type,
            Iy_bins_list,
            Iy_ends_list,
            Jy_bins_list,
            Jy_ends_list,
            Ky_bins_list,
            Ky_ends_list,
            use_poisson_len=use_poisson_len,
        )
        if OPT > 0:
            ratios.append(ALG / OPT)

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return avg_ratio, ratios
