from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchingResult:
    algorithm: str
    alg: int


def advertiser_degrees(A_size, neighbors):
    degrees = [0] * A_size
    for neigh_i in neighbors:
        for a in neigh_i:
            degrees[a] += 1
    return degrees


def compute_offline_opt(A_size, neighbors, arrivals):
    """
    Compute offline OPT for a realized arrival sequence.

    This is maximum bipartite matching between advertisers and realized
    impression copies. It is implemented directly so the shared benchmark does
    not require importing the older NetworkX helper modules.
    """
    B_size = len(arrivals)
    if A_size <= 0 or B_size <= 0:
        return 0

    adj = [[] for _ in range(A_size)]
    for imp_id, i in enumerate(arrivals):
        for a in neighbors[i]:
            if 0 <= a < A_size:
                adj[a].append(imp_id)

    pair_u = [-1] * A_size
    pair_v = [-1] * B_size
    dist = [0] * A_size

    def bfs():
        queue = deque()
        found_free = False
        for u in range(A_size):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                next_u = pair_v[v]
                if next_u == -1:
                    found_free = True
                elif dist[next_u] == -1:
                    dist[next_u] = dist[u] + 1
                    queue.append(next_u)
        return found_free

    def dfs(u):
        for v in adj[u]:
            next_u = pair_v[v]
            if next_u == -1 or (
                dist[next_u] == dist[u] + 1 and dfs(next_u)
            ):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(A_size):
            if pair_u[u] == -1 and dfs(u):
                matching += 1
    return matching


def require_state(algorithm, state):
    if state is None:
        raise ValueError(
            f"Algorithm '{algorithm}' requires precomputed state. "
            "Call prepare_matching_states first."
        )
