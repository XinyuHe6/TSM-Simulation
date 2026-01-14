import random

def simulate_degree_greedy_once(A_size, I_size, neighbors, p, T, adv_degrees):

    # Simulates a single run using the Minimum Degree


    matched_A = [False] * A_size
    ALG = 0

    hat_I = []
    imp_type = []
    edges_real = []

    for _ in range(T):
        # Sample an impression type based on probability p
        i = random.choices(range(I_size), weights=p, k=1)[0]

        imp_id = len(hat_I)
        hat_I.append(imp_id)
        imp_type.append(i)

        # Build realization edges (used for OPT calculation later)
        current_neighbors = neighbors[i]
        for a in current_neighbors:
            edges_real.append((a, imp_id))



        # Find available candidates (advertisers not yet matched)
        candidates = []
        for a in current_neighbors:
            if not matched_A[a]:
                candidates.append(a)

        #If there are candidates, pick the one with the lowest degree
        if candidates:
            # Find the candidate with the minimum degree value in adv_degrees
            best_a = min(candidates, key=lambda x: adv_degrees[x])

            # Assign the match
            matched_A[best_a] = True
            ALG += 1

    return ALG, hat_I, edges_real, imp_type

def simulate_many_runs_degree(A_size, I_size, neighbors, p, T, num_runs=1):

    # Main function that executes multiple simulations

    # Calculate the degree of each advertiser
    # Count how many edges each advertiser has in the original graph
    adv_degrees = [0] * A_size
    for i in range(I_size):
        for a in neighbors[i]:
            adv_degrees[a] += 1

    # 2. Simulation Loop
    ratios = []
    for _ in range(num_runs):
        # Run one 'online' simulation
        ALG, hat_I, edges_real, imp_type = simulate_degree_greedy_once(
            A_size, I_size, neighbors, p, T, adv_degrees
        )

        # Calculate the Hindsight Optimum
        # Note: Assumes 'compute_opt_from_realization' is already defined in your environment
        OPT = compute_opt_from_realization(A_size, hat_I, edges_real)

        if OPT > 0:
            ratios.append(ALG / OPT)
        else:
            # If OPT is 0 (no valid edges), the ratio is technically 1.0
            if T > 0: ratios.append(1.0)

    # Calculate Average
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return avg_ratio, ratios