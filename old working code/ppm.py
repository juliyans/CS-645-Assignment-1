import random
import networkx as nx
from dataclasses import dataclass
from src.topology import leaves, branch_root_of, path_leaf_to_victim
from collections import defaultdict
from collections import Counter

@dataclass(frozen=True)
class Hosts:
    attackers: list[int]
    normal_users: list[int]

# Choose attackers and normal users
def choose_hosts(G: nx.DiGraph, num_attackers: int, seed: int = 0) -> Hosts:
    # Attackers must be at leaf routers
    # At most one attacker per branch
    # All other leaves are normal users
    rng = random.Random(seed)

    leafs = leaves(G)
    if len(leafs) < num_attackers + 1:
        raise ValueError("Not enough leaf nodes for attackers + at least one normal user")

    # Group leaves by branch
    by_branch: dict[int, list[int]] = {}
    for lf in leafs:
        br = branch_root_of(G, lf)
        by_branch.setdefault(br, []).append(lf)

    # Pick at most one attacker per branch
    branch_keys = list(by_branch.keys())
    rng.shuffle(branch_keys)

    attackers: list[int] = []
    used_branches: set[int] = set()

    for br in branch_keys:
        if len(attackers) >= num_attackers:
            break
        cand = by_branch[br]
        if cand:
            attackers.append(rng.choice(cand))
            used_branches.add(br)

    # If there's not enough branches/leaves
    if len(attackers) < num_attackers:
        remaining = [lf for lf in leafs if lf not in attackers]
        rng.shuffle(remaining)
        attackers.extend(remaining[: (num_attackers - len(attackers))])

    normal_users = [lf for lf in leafs if lf not in attackers]
    return Hosts(attackers=attackers, normal_users=normal_users)

@dataclass
class NodePacket:
    # Packet header for node sampling
    # Stores a single router ID and overwritten as it travels
    node: int | None = None

class NodeSampler:
    def __init__(self, p: float, seed: int = 0):
        self.p = p
        self.rng = random.Random(seed)

    def forward(self, G: nx.DiGraph, source_leaf: int) -> NodePacket:
        # Simulate one packet traveling from source_leaf to victim
        # At each router on the path and mark with probability p
        pkt = NodePacket(node=None)

        for router in path_leaf_to_victim(G, source_leaf):
            if self.rng.random() < self.p:
                pkt.node = router  # Overwrite marking

        return pkt
    
@dataclass
class EdgePacket:
    # Packet header for edge sampling:
    # Start: router that last marked the packet
    # End: the next router closer to victim after start (captured when not re-marked)
    # Distance: number of hops since last marking
    start: int | None = None
    end: int | None = None
    distance: int = 0

@dataclass
class EdgeSampler:
    def __init__(self, p: float, seed: int = 0):
        self.p = p
        self.rng = random.Random(seed)

    def forward(self, G: nx.DiGraph, source_leaf: int) -> EdgePacket:
        # Simulate one packet traveling from source_leaf to victim
        # At each router on path (leaf to victim)
        '''
          with prob p:
             start = router
             distance = 0
             end = None
          else:
             if distance == 0: end = router
             distance += 1
        '''
        pkt = EdgePacket()

        for router in path_leaf_to_victim(G, source_leaf):
            if self.rng.random() < self.p:
                pkt.start = router
                pkt.end = None
                pkt.distance = 0
            else:
                if pkt.distance == 0:
                    pkt.end = router
                pkt.distance += 1

        return pkt
    
def collect_node_samples(G: nx.DiGraph, sampler: NodeSampler, sources: list[int], packets_per_source: list[int]) -> list[int]:
    # Collect node sampling marks observed at victim from multiple sources.
    observed = []
    for src, n_packets in zip(sources, packets_per_source):
        for _ in range(n_packets):
            pkt = sampler.forward(G, src)
            if pkt.node is not None:
                observed.append(pkt.node)
    return observed # Returns a list of observed node IDs

def collect_edge_samples(G: nx.DiGraph, sampler: EdgeSampler, sources: list[int], packets_per_source: list[int]) -> list[tuple[int | None, int | None, int]]:
    # Collect raw edge sampling tuples (start, end, distance) observed at victim
    observed = []
    for src, n_packets in zip(sources, packets_per_source):
        for _ in range(n_packets):
            pkt = sampler.forward(G, src)
            observed.append((pkt.start, pkt.end, pkt.distance))
    return observed

def edges_by_distance(samples: list[tuple[int | None, int | None, int]], victim: int = 0) -> dict[int, set[tuple[int, int]]]:
    # Convert raw edge samples into a mapping:
    # Distance d to set of edges (start to end
    by_d = defaultdict(set)
    for s, e, d in samples:
        if s is None: # If start is None, skip since no marking happened
            continue
        if d == 0: # If distance == 0, treat edge as (start to victim)
            by_d[0].add((s, victim))
        else: # Else require end to exist, store (start to end)
            if e is None:
                continue
            by_d[d].add((s, e))
    return dict(by_d)

def node_reconstruct_order(node_obs: list[int]) -> list[int]:
    # Return routers sorted by frequency descending.
    # Closer-to-victim routers tend to be sampled more often.
    counts = Counter(node_obs)
    return [n for n, _ in counts.most_common()]

def node_guess_attacker_leaf(G: nx.DiGraph, ordered_nodes: list[int]) -> int | None:
    # Choose the farthest router as the least frequent router in ordered list
    # Return a leaf in that router's subtree
    if not ordered_nodes:
        return None

    farthest = ordered_nodes[-1]  # Least frequent
    # choose a leaf in farthest's subtree
    stack = [farthest]
    leafs = []
    while stack:
        u = stack.pop()
        kids = list(G.successors(u))
        if not kids:
            leafs.append(u)
        else:
            stack.extend(kids)
    return min(leafs) if leafs else farthest

def edge_reconstruct_path(by_d: dict[int, set[tuple[int, int]]], victim: int = 0) -> list[int]:
    # Reconstruct a single path back to attacker using the distance-indexed edge sets.
    # Returns routers from attacker-side toward victim
    if 0 not in by_d or not by_d[0]:
        return []

    # Choose one edge at d=0 (R0 to victim)
    r0 = sorted(by_d[0])[0][0]
    path = [r0]  # Closest router to victim

    cur = r0
    d = 1
    while d in by_d:
        # Find an edge at distance d whose end matches current router
        candidates = [s for (s, e) in by_d[d] if e == cur]
        if not candidates:
            break
        nxt = sorted(candidates)[0]
        path.append(nxt)
        cur = nxt
        d += 1

    # Path is currently (closest to farthest) so reverse to (farthest to closest)
    path.reverse()
    return path

def node_guess_two_attackers(G: nx.DiGraph, node_obs: list[int], max_attackers: int = 2) -> list[int]:
    """
    Heuristic for multi-attacker node sampling:
    - group observed nodes by branch (child of victim)
    - reconstruct per-branch
    - pick up to 2 branches with the most observations (strongest signals)
    Returns a list of guessed attacker leaves (unique).
    """
    # Group observations by branch root (child of victim)
    by_branch: dict[int, list[int]] = {}
    for n in node_obs:
        br = branch_root_of(G, n)
        by_branch.setdefault(br, []).append(n)

    # Rank branches by number of observations (more packets -> more marks)
    branches_sorted = sorted(by_branch.items(), key=lambda kv: len(kv[1]), reverse=True)

    guesses: list[int] = []
    for br, obs in branches_sorted:
        ordered = node_reconstruct_order(obs)
        guess = node_guess_attacker_leaf(G, ordered)
        if guess is None:
            continue
        if guess not in guesses:
            guesses.append(guess)
        if len(guesses) >= max_attackers:
            break

    return guesses


def edge_build_graph(samples: list[tuple[int | None, int | None, int]], victim: int = 0) -> nx.DiGraph:
    """
    Build a directed graph of reconstructed edges from edge-sampling tuples.
    Edge direction is from farther -> closer (toward victim).
    """
    H = nx.DiGraph()
    H.add_node(victim)

    by_d = edges_by_distance(samples, victim=victim)
    for d, edges in by_d.items():
        for u, v in edges:
            H.add_edge(u, v)
    return H


def _edge_distance_index_to_victim(H: nx.DiGraph, victim: int = 0) -> dict[int, int]:
    """Return distance *index* to victim for each node in H.

    Our code stores edge tuples under key d where:
      d == 0 means the router is directly adjacent to the victim.

    In a directed graph where edges point toward the victim (u -> v),
    the shortest path length L(u,victim) satisfies:
      L == 1 for an adjacent router.

    So we define the distance *index* as: idx(u) = L(u,victim) - 1.
    """
    # Compute shortest path length to victim for all nodes that can reach victim.
    # Do BFS from victim on the *reversed* graph.
    rev = H.reverse(copy=False)
    lengths = nx.single_source_shortest_path_length(rev, victim)
    idx: dict[int, int] = {}
    for n, L in lengths.items():
        if n == victim:
            continue
        idx[n] = L - 1
    return idx


def edge_build_pruned_graph(samples: list[tuple[int | None, int | None, int]], victim: int = 0) -> nx.DiGraph:
    """Build and prune the edge-reconstruction graph.

    This implements the key step in the edge-sampling traceback algorithm:
    remove any edge (x,y,d) whose recorded distance d does not match the
    distance from x to the victim in the reconstructed graph.

    We also enforce that edges go one step closer to the victim.
    """
    # First build a graph that retains the sampled distance d per edge.
    by_d = edges_by_distance(samples, victim=victim)
    H = nx.DiGraph()
    H.add_node(victim)

    # Store all candidate directed edges with their claimed distance index.
    # d == 0 corresponds to edges (u -> victim).
    edges_with_d: list[tuple[int, int, int]] = []
    for d, edges in by_d.items():
        for u, v in edges:
            edges_with_d.append((u, v, d))
            H.add_edge(u, v)

    # Iteratively prune until stable.
    changed = True
    while changed:
        changed = False
        idx = _edge_distance_index_to_victim(H, victim=victim)
        to_remove: list[tuple[int, int]] = []

        for u, v, d in edges_with_d:
            if not H.has_edge(u, v):
                continue
            # If u cannot reach victim in current graph, prune.
            if u not in idx:
                to_remove.append((u, v))
                continue

            # Distance index must match exactly.
            if idx[u] != d:
                to_remove.append((u, v))
                continue

            # Edge must move one step closer toward victim.
            if v == victim:
                if d != 0:
                    to_remove.append((u, v))
                continue
            if v not in idx:
                to_remove.append((u, v))
                continue
            if idx[v] != d - 1:
                to_remove.append((u, v))

        if to_remove:
            H.remove_edges_from(to_remove)
            changed = True

    return H


def edge_guess_attackers_from_graph(H: nx.DiGraph, victim: int = 0) -> list[int]:
    """
    Attackers appear as *sources* in the reconstructed graph.
    Since edges point toward the victim (u -> v), attackers have in_degree == 0.
    Returns candidate routers (excluding victim).
    """
    candidates = [n for n in H.nodes if n != victim and H.in_degree(n) == 0]
    return sorted(candidates)


def edge_guess_single_attacker_from_graph(H: nx.DiGraph, victim: int = 0) -> int | None:
    """Guess a single attacker from a pruned reconstruction graph.

    Strategy:
      1) take all source candidates (in_degree == 0)
      2) choose the farthest (largest distance index to victim)
    """
    sources = edge_guess_attackers_from_graph(H, victim=victim)
    if not sources:
        return None
    idx = _edge_distance_index_to_victim(H, victim=victim)
    return max(sources, key=lambda n: idx.get(n, -1))