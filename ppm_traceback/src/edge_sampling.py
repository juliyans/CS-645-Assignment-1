import random
import networkx as nx
from dataclasses import dataclass
from collections import defaultdict
from src.topology import path_leaf_to_victim

'''
Edge Sampling:
- Each packet can store ONE edge (two adjacent routers), not the whole path
- Packet stores: (start, end, distance)
    start    = router that started the mark
    end      = next router toward victim (filled in by the next router)
    distance = hops since the last mark started

Marking procedure (from slides):
for each packet w at router R:
    let x = random in [0, 1)
    if x < p:
        w.start = R
        w.distance = 0
    else:
        if w.distance == 0:
            w.end = R
        w.distance += 1

Victim reconstruction idea:
- Collect many (start, end, distance) samples
- Convert them into edges grouped by distance from the victim
- Chain edges together (or build a graph) to infer attacker location(s)
'''

@dataclass
class EdgePacket:
    # Header fields used by edge sampling
    start: int | None = None # Router that last started a mark
    end: int | None = None # Next router toward victim (filled downstream)
    distance: int = 0 # Hops since last mark started

class EdgeSampler:
    def __init__(self, p: float, seed: int = 0):
        self.p = p
        self.rng = random.Random(seed)

    def forward(self, G: nx.DiGraph, source_leaf: int) -> EdgePacket:
        # Simulate one packet from source_leaf to victim
        # Routers either:
        # - Start a new mark (set start, reset distance)
        # - Or finish the edge (set end when distance==0) and increase distance
        pkt = EdgePacket()
        for router in path_leaf_to_victim(G, source_leaf):
            if self.rng.random() < self.p:
                pkt.start = router
                pkt.distance = 0
            else:
                if pkt.distance == 0:
                    pkt.end = router
                pkt.distance += 1
        return pkt

def edges_by_distance(samples, victim=0):
    # Convert raw samples into edges grouped by distance from victim
    # distance == 0  -> edge is (start, victim)
    # distance  > 0  -> edge is (start, end) (end must exist)
    by_d = defaultdict(set)

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

def edge_reconstruct_path(by_d, victim=0) -> list[int]:
    # Reconstruct ONE path by chaining distance-indexed edges
    # Start at distance 0 (router directly connected to victim)
    # Then walk outward by matching (start, end) where end == current node
    # Returned list is [farthest (attacker side), ..., closest to victim]
    if 0 not in by_d or not by_d[0]:
        return []

    # Pick a starting neighbor of victim (if multiple exist, choose smallest ID)
    cur = sorted(by_d[0])[0][0]
    path = [cur]
    d = 1

    while d in by_d:
        # Find the router at distance d that points into the current router
        candidates = [s for (s, e) in by_d[d] if e == cur]
        if not candidates:
            break
        cur = sorted(candidates)[0]
        path.append(cur)
        d += 1

    # Reverse so index 0 is the farthest router (attacker side)
    return list(reversed(path))

def edge_build_graph(samples, victim=0) -> nx.DiGraph:
    # Build a directed graph from all observed edges
    # Edge direction: farther -> closer (toward victim)
    # In this graph, attackers appear as source nodes (in_degree == 0)
    H = nx.DiGraph()
    H.add_node(victim)

    by_d = edges_by_distance(samples, victim=victim)
    for edges in by_d.values():
        H.add_edges_from(edges)
    return H

def edge_guess_attackers_from_graph(H, victim=0) -> list[int]:
    # Attackers are sources in the reconstructed graph:
    # - Victim is excluded
    # - Attacker candidates have in_degree == 0
    return sorted(n for n in H.nodes if n != victim and H.in_degree(n) == 0)