import random
import networkx as nx
from dataclasses import dataclass
from collections import Counter
from src.topology import branch_root_of, path_leaf_to_victim

'''
Node Sampling:
- Each packet stores ONE router ID (not the whole path)
- As the packet moves toward the victim, routers may overwrite that field

Marking procedure (from slides):
for each packet w at router R:
    let x = random number in [0, 1)
    if x < p: write R into w.node

Reconstruction idea:
- The victim counts how often each router ID appears in received marks
- Routers closer to the victim overwrite more often, so they show up more
- Sorting by frequency gives an ordering from victim side -> attacker side
'''

@dataclass
class NodePacket:
    # One header field: last router that marked the packet
    # None means the packet was never marked
    node: int | None = None

class NodeSampler:
    def __init__(self, p: float, seed: int = 0):
        self.p = p
        self.rng = random.Random(seed)

    def forward(self, G: nx.DiGraph, source_leaf: int) -> NodePacket:
        # Simulate one packet from source_leaf to victim
        # Each router overwrites pkt.node with probability p (last write wins)
        pkt = NodePacket()
        for router in path_leaf_to_victim(G, source_leaf):
            if self.rng.random() < self.p:
                pkt.node = router
        return pkt

def node_reconstruct_order(node_obs: list[int]) -> list[int]:
    # Victim reconstruction (slides):
    # - Count marks per router ID
    # - Sort by count (high -> low)
    # - High count = close to victim, low count = farther (attacker side)
    return [n for n, _ in Counter(node_obs).most_common()]

def node_guess_attacker_leaf(G, ordered_nodes, node_obs=None) -> int | None:
    # Guess attacker as:
    # 1) farthest observed router = least frequent router
    # 2) attacker must be in that router's subtree
    # 3) return a leaf in that subtree (prefer a leaf seen in node_obs if possible)
    if not ordered_nodes:
        return None

    farthest = ordered_nodes[-1]

    # Collect all leaves in farthest's subtree
    stack = [farthest]
    leafs = []
    while stack:
        u = stack.pop()
        kids = list(G.successors(u))
        if not kids:
            leafs.append(u)
        else:
            stack.extend(kids)

    if not leafs:
        return farthest

    # If any leaf itself appeared as a mark, prefer it (strong signal)
    if node_obs:
        obs = set(node_obs)
        seen = [lf for lf in leafs if lf in obs]
        if seen:
            return seen[0]

    return leafs[0]

def node_guess_two_attackers(G, node_obs, max_attackers=2) -> list[int]:
    # Two-attacker heuristic:
    # - Group marks by branch (child of victim)
    # - Run single-attacker guess per branch
    # - Pick top branches with the most observations
    by_branch: dict[int, list[int]] = {}
    for n in node_obs:
        by_branch.setdefault(branch_root_of(G, n), []).append(n)

    guesses = []
    for _, obs in sorted(by_branch.items(), key=lambda kv: len(kv[1]), reverse=True):
        g = node_guess_attacker_leaf(G, node_reconstruct_order(obs), node_obs=obs)
        if g and g not in guesses:
            guesses.append(g)
        if len(guesses) >= max_attackers:
            break

    return guesses