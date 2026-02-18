import random
import networkx as nx
from dataclasses import dataclass
from src.topology import leaves, branch_root_of, path_leaf_to_victim
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# Host selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Hosts:
    attackers: list[int]
    normal_users: list[int]


def choose_hosts(G: nx.DiGraph, num_attackers: int, num_normal: int = 1, seed: int = 0) -> Hosts:
    """
    Select attacker and normal-user leaves.

    Rules (per assignment):
      - Attackers must be leaf routers.
      - At most one attacker per branch.
      - Exactly num_normal normal users (default = 1 per assignment wording).
    """
    rng = random.Random(seed)

    leafs = leaves(G)
    if len(leafs) < num_attackers + num_normal:
        raise ValueError(
            f"Not enough leaf nodes for {num_attackers} attackers + "
            f"{num_normal} normal user(s). Found {len(leafs)} leaves."
        )

    # Group leaves by branch (direct child of victim)
    by_branch: dict[int, list[int]] = {}
    for lf in leafs:
        br = branch_root_of(G, lf)
        by_branch.setdefault(br, []).append(lf)

    # Pick at most one attacker per branch
    branch_keys = list(by_branch.keys())
    rng.shuffle(branch_keys)

    attackers: list[int] = []
    for br in branch_keys:
        if len(attackers) >= num_attackers:
            break
        cand = by_branch[br]
        if cand:
            attackers.append(rng.choice(cand))

    if len(attackers) != num_attackers:
        raise ValueError(
            f"Could not place {num_attackers} attackers with at most one per branch. "
            f"Only {len(attackers)} branches had available leaves."
        )

    # Pick exactly num_normal normal users from the remaining leaves
    remaining = [lf for lf in leafs if lf not in attackers]
    if len(remaining) < num_normal:
        raise ValueError(
            f"Not enough remaining leaves for {num_normal} normal user(s)."
        )
    rng.shuffle(remaining)
    normal_users = remaining[:num_normal]

    return Hosts(attackers=attackers, normal_users=normal_users)


# ---------------------------------------------------------------------------
# Node sampling
# ---------------------------------------------------------------------------

@dataclass
class NodePacket:
    """Packet header field for node sampling: a single overwritten router ID."""
    node: int | None = None


class NodeSampler:
    def __init__(self, p: float, seed: int = 0):
        self.p = p
        self.rng = random.Random(seed)

    def forward(self, G: nx.DiGraph, source_leaf: int) -> NodePacket:
        """
        Simulate one packet from source_leaf to victim.
        Each router on the path overwrites w.node with probability p.
        (Marking procedure from slides.)
        """
        pkt = NodePacket(node=None)
        for router in path_leaf_to_victim(G, source_leaf):
            if self.rng.random() < self.p:
                pkt.node = router   # overwrite
        return pkt


# ---------------------------------------------------------------------------
# Edge sampling
# ---------------------------------------------------------------------------

@dataclass
class EdgePacket:
    """
    Packet header fields for edge sampling.
      start    : router that last marked the packet
      end      : next router closer to victim (filled in by downstream router)
      distance : hops since last marking event
    """
    start: int | None = None
    end: int | None = None
    distance: int = 0


class EdgeSampler:
    def __init__(self, p: float, seed: int = 0):
        self.p = p
        self.rng = random.Random(seed)

    def forward(self, G: nx.DiGraph, source_leaf: int) -> EdgePacket:
        """
        Simulate one packet from source_leaf to victim.

        Marking procedure (from slides):
          if x < p:
              write R into w.start, 0 into w.distance
              (w.end is NOT explicitly cleared per pseudocode)
          else:
              if w.distance == 0: write R into w.end
              increment w.distance
        """
        pkt = EdgePacket()
        for router in path_leaf_to_victim(G, source_leaf):
            if self.rng.random() < self.p:
                pkt.start = router
                pkt.distance = 0
                # Per slide pseudocode: w.end is NOT reset here
            else:
                if pkt.distance == 0:
                    pkt.end = router
                pkt.distance += 1
        return pkt


# ---------------------------------------------------------------------------
# Helper: collect samples (utility functions, not used in experiment loop)
# ---------------------------------------------------------------------------

def collect_node_samples(
    G: nx.DiGraph,
    sampler: NodeSampler,
    sources: list[int],
    packets_per_source: list[int],
) -> list[int]:
    observed = []
    for src, n in zip(sources, packets_per_source):
        for _ in range(n):
            pkt = sampler.forward(G, src)
            if pkt.node is not None:
                observed.append(pkt.node)
    return observed


def collect_edge_samples(
    G: nx.DiGraph,
    sampler: EdgeSampler,
    sources: list[int],
    packets_per_source: list[int],
) -> list[tuple[int | None, int | None, int]]:
    observed = []
    for src, n in zip(sources, packets_per_source):
        for _ in range(n):
            pkt = sampler.forward(G, src)
            observed.append((pkt.start, pkt.end, pkt.distance))
    return observed


# ---------------------------------------------------------------------------
# Node sampling reconstruction
# ---------------------------------------------------------------------------

def node_reconstruct_order(node_obs: list[int]) -> list[int]:
    """
    Return routers sorted by frequency descending.
    Routers closer to the victim appear more often because downstream
    routers cannot overwrite marks that were already written by them.
    Most frequent = closest to victim, least frequent = farthest (attacker side).
    """
    counts = Counter(node_obs)
    return [n for n, _ in counts.most_common()]


def node_guess_attacker_leaf(
    G: nx.DiGraph,
    ordered_nodes: list[int],
    node_obs: list[int] | None = None,
) -> int | None:
    """
    Guess the attacker leaf from node-sampling observations.

    Strategy:
      1. Take the least-frequent router (farthest from victim).
      2. Find all leaves in its subtree.
      3. Prefer a leaf that actually appeared in node_obs (was sampled).
      4. Fall back to first leaf if none were directly observed.
    """
    if not ordered_nodes:
        return None

    farthest = ordered_nodes[-1]   # least frequent = farthest from victim

    # Collect all leaves under farthest
    stack, leafs = [farthest], []
    while stack:
        u = stack.pop()
        kids = list(G.successors(u))
        if not kids:
            leafs.append(u)
        else:
            stack.extend(kids)

    if not leafs:
        return farthest

    # Prefer a leaf that was directly observed in node_obs
    if node_obs:
        obs_set = set(node_obs)
        observed_leaves = [lf for lf in leafs if lf in obs_set]
        if observed_leaves:
            return observed_leaves[0]

    return leafs[0]


def node_guess_two_attackers(
    G: nx.DiGraph,
    node_obs: list[int],
    max_attackers: int = 2,
) -> list[int]:
    """
    Multi-attacker heuristic for node sampling:
      - Group observed node marks by branch.
      - Rank branches by observation count (stronger signal = more packets).
      - For each top branch, reconstruct the farthest router and return its leaf.
    """
    by_branch: dict[int, list[int]] = {}
    for n in node_obs:
        br = branch_root_of(G, n)
        by_branch.setdefault(br, []).append(n)

    branches_sorted = sorted(by_branch.items(), key=lambda kv: len(kv[1]), reverse=True)

    guesses: list[int] = []
    for br, obs in branches_sorted:
        ordered = node_reconstruct_order(obs)
        # Pass obs so leaf selection prefers observed leaves
        guess = node_guess_attacker_leaf(G, ordered, node_obs=obs)
        if guess is None:
            continue
        if guess not in guesses:
            guesses.append(guess)
        if len(guesses) >= max_attackers:
            break

    return guesses


# ---------------------------------------------------------------------------
# Edge sampling reconstruction
# ---------------------------------------------------------------------------

def edges_by_distance(
    samples: list[tuple[int | None, int | None, int]],
    victim: int = 0,
) -> dict[int, set[tuple[int, int]]]:
    """
    Convert raw edge-sample tuples into distance-indexed edge sets.

      d == 0  ->  edge (start, victim)   [router directly adjacent to victim]
      d  > 0  ->  edge (start, end)      [start is d hops from victim]
    """
    by_d: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for s, e, d in samples:
        if s is None:
            continue
        if d == 0:
            by_d[0].add((s, victim))
        else:
            if e is None:
                continue
            by_d[d].add((s, e))
    return dict(by_d)


def edge_reconstruct_path(
    by_d: dict[int, set[tuple[int, int]]],
    victim: int = 0,
) -> list[int]:
    """
    Reconstruct a single attack path from distance-indexed edge sets.

    Returns routers ordered [farthest (attacker side) ... closest to victim].
    Use path[0] to get the attacker-side router.
    """
    if 0 not in by_d or not by_d[0]:
        return []

    # Start from the router directly adjacent to victim (d=0)
    r0 = sorted(by_d[0])[0][0]
    path = [r0]   # built closest -> farthest, then reversed

    cur = r0
    d = 1
    while d in by_d:
        candidates = [s for (s, e) in by_d[d] if e == cur]
        if not candidates:
            break
        nxt = sorted(candidates)[0]
        path.append(nxt)
        cur = nxt
        d += 1

    # Reverse so path[0] = farthest (attacker side), path[-1] = closest to victim
    path.reverse()
    return path


def edge_build_graph(
    samples: list[tuple[int | None, int | None, int]],
    victim: int = 0,
) -> nx.DiGraph:
    """
    Build a directed graph from edge-sampling tuples.
    Edge direction: farther router -> closer router (toward victim).
    Attacker leaves are SOURCES (in_degree == 0).
    Victim is the SINK (out_degree == 0 within this subgraph).
    """
    H = nx.DiGraph()
    H.add_node(victim)

    by_d = edges_by_distance(samples, victim=victim)
    for d, edges in by_d.items():
        for u, v in edges:
            H.add_edge(u, v)   # u is farther, v is closer
    return H


def edge_guess_attackers_from_graph(H: nx.DiGraph, victim: int = 0) -> list[int]:
    """
    In the reconstructed graph edges point toward the victim.
    Attackers are SOURCES: nodes with in_degree == 0 (nothing points to them).
    Returns sorted list of candidate attacker nodes (victim excluded).
    """
    candidates = [
        n for n in H.nodes
        if n != victim and H.in_degree(n) == 0   # FIX: was out_degree (wrong)
    ]
    return sorted(candidates)
