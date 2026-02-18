import random
import networkx as nx
from dataclasses import dataclass
from topology import leaves, branch_root_of

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
        raise ValueError("Not enough leaf nodes for attackers + at least one benign user")

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