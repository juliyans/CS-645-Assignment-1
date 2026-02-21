import random
import networkx as nx
from dataclasses import dataclass
from src.topology import leaves, branch_root_of

# Re-export both algorithm modules so experiment.py can import everything
# From src.ppm without needing to know about the individual files
from src.node_sampling import * 
from src.edge_sampling import *

@dataclass(frozen=True)
class Hosts:
    attackers: list[int]    # Leaf node IDs chosen as attackers
    normal_users: list[int] # Leaf node IDs chosen as normal users

def choose_hosts(G, num_attackers, num_normal=1, seed=0) -> Hosts:
    # Select attacker and normal user leaves following the requirements:
    # - Attackers must be leaf nodes
    # - At most one attacker per branch, by picking one per branch key
    # - Exactly num_normal normal users from the remaining leaves
    rng = random.Random(seed)

    # Group all leaves by their branch so we can enforce one attacker per branch
    leafs = leaves(G)
    by_branch: dict[int, list[int]] = {}
    for lf in leafs:
        by_branch.setdefault(branch_root_of(G, lf), []).append(lf)

    # Shuffle branch order so attacker placement varies across trials
    branch_keys = list(by_branch.keys())
    rng.shuffle(branch_keys)

    # Pick one random leaf from each of the first num_attackers branches
    attackers = [rng.choice(by_branch[br]) for br in branch_keys[:num_attackers]]
    if len(attackers) != num_attackers:
        raise ValueError("Not enough branches for requested attackers")

    # Normal users are drawn from leaves not already chosen as attackers
    remaining = [lf for lf in leafs if lf not in attackers]
    rng.shuffle(remaining)
    return Hosts(attackers=attackers, normal_users=remaining[:num_normal])