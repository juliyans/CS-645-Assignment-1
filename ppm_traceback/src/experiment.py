import random
import networkx as nx
from dataclasses import dataclass
from src.ppm import (
    choose_hosts, NodeSampler, EdgeSampler,
    node_reconstruct_order, node_guess_attacker_leaf, node_guess_two_attackers,
    edges_by_distance, edge_reconstruct_path, edge_build_graph, edge_guess_attackers_from_graph,
)

NORMAL_RATE = 1 # Normal user sends 1 packet per tick
MAX_TICKS   = 500  # Time limit per trial since attack packets available = x * MAX_TICKS
# x controls how many marked samples the victim collects

@dataclass
class Stats:
    node_acc: float # Fraction of trials where node sampling found the attacker
    edge_acc: float  # Fraction of trials where edge sampling found the attacker
    node_mean_conv: float | None # Mean ticks to converge (successful trials only)
    edge_mean_conv: float | None

def _run_trial(G, p, x, seed, num_attackers):
    # Run one simulation trial for either Q1 (num_attackers=1) or Q2 (num_attackers=2)
    rng = random.Random(seed)

    # Select attacker and normal user leaves (one attacker per branch, one normal user)
    hosts     = choose_hosts(G, num_attackers=num_attackers, num_normal=1, seed=seed + 1)
    attackers = hosts.attackers
    normal    = hosts.normal_users

    # Independent samplers per trial to avoid RNG state sharing between trials
    NS = NodeSampler(p=p, seed=rng.randint(0, 10**9))
    ES = EdgeSampler(p=p, seed=rng.randint(0, 10**9))

    node_obs, edge_obs = [], [] # Observations collected at the victim
    node_conv = edge_conv = None # Tick at which each algorithm converges (None = not yet)
    true_set = set(attackers) # Set truth for exact-match checking in Q2

    for tick in range(1, MAX_TICKS + 1):

        # Normal user sends background traffic each tick
        # Marks are not recorded so victim only processes attack packets
        for b in normal:
            for _ in range(NORMAL_RATE):
                NS.forward(G, b)
                ES.forward(G, b)

        # Each attacker sends x packets per tick (x times faster than normal user)
        # Higher x = more marked samples = easier/faster to reconstruct path
        for a in attackers:
            for _ in range(x * NORMAL_RATE):
                pN = NS.forward(G, a)
                if pN.node is not None:
                    node_obs.append(pN.node) # Record marked router ID
                pE = ES.forward(G, a)
                edge_obs.append((pE.start, pE.end, pE.distance)) # Record edge tuple

        # Check node sampling convergence once per tick
        if node_conv is None:
            if num_attackers == 1:
                # Rank routers by frequency, least frequent = farthest = attacker side
                g = node_guess_attacker_leaf(G, node_reconstruct_order(node_obs), node_obs)
                if g == attackers[0]:
                    node_conv = tick
            else:
                # Q2: group by branch so reconstruct farthest leaf per branch
                if set(node_guess_two_attackers(G, node_obs, 2)) == true_set:
                    node_conv = tick

        # Check edge sampling convergence once per tick
        if edge_conv is None:
            if num_attackers == 1:
                # Chain distance-indexed edges from victim outward so path[0] = attacker side
                path = edge_reconstruct_path(edges_by_distance(edge_obs), victim=0)
                if path and path[0] == attackers[0]:
                    edge_conv = tick
            else:
                # Q2: build graph and find source nodes (in_degree==0) as attackers
                # Exact match required where there's no missing attackers & no false positives
                H = edge_build_graph(edge_obs, victim=0)
                if set(edge_guess_attackers_from_graph(H)) == true_set:
                    edge_conv = tick

        if node_conv and edge_conv:
            break  # Both converged, so there's no need to continue

    return (1 if node_conv else 0), (1 if edge_conv else 0), node_conv, edge_conv

def _run_grid(G, p_values, x_values, trials, seed, num_attackers) -> dict:
    # Go through all (x, p) combinations, running 'trials'
    # Returns a Stats object per (x, p) pair
    rng = random.Random(seed)
    results = {}
    for x in x_values:
        for p in p_values:
            ns, es, nc, ec = 0, 0, [], []
            for _ in range(trials):
                n_ok, e_ok, n_c, e_c = _run_trial(G, p, x, rng.randint(0, 10**9), num_attackers)
                ns += n_ok; es += e_ok
                if n_c: nc.append(n_c)
                if e_c: ec.append(e_c)
            results[(x, p)] = Stats(
                node_acc=ns/trials, edge_acc=es/trials,
                node_mean_conv=sum(nc)/len(nc) if nc else None,
                edge_mean_conv=sum(ec)/len(ec) if ec else None,
            )
    return results

# Q1 and Q2 are the same runs, just different attacker counts
def run_grid_one_attacker(G, p_values, x_values, trials, seed=0):
    return _run_grid(G, p_values, x_values, trials, seed, num_attackers=1)

def run_grid_two_attackers(G, p_values, x_values, trials, seed=0):
    return _run_grid(G, p_values, x_values, trials, seed, num_attackers=2)