import random
import networkx as nx
from dataclasses import dataclass
from src.ppm import (
    choose_hosts,
    NodeSampler, EdgeSampler,
    node_reconstruct_order, node_guess_attacker_leaf,
    edges_by_distance, edge_reconstruct_path
)

NORMAL_RATE = 1  # normal leaves send 1 packet per tick

def run_trial_one_attacker(G: nx.DiGraph, p: float, x: int, seed: int, max_attack_packets: int = 5000):
    """
    One trial
      - 1 attacker at a leaf
      - normal leaves are normal users at slow rate
      - attacker sends x times faster

    Returns:
      (node_success, edge_success, node_packets_to_converge, edge_packets_to_converge)
    """
    rng = random.Random(seed)
    hosts = choose_hosts(G, num_attackers=1, seed=seed + 1)
    attacker = hosts.attackers[0]
    normal = hosts.normal_users

    node_sampler = NodeSampler(p=p, seed=rng.randint(0, 10**9))
    edge_sampler = EdgeSampler(p=p, seed=rng.randint(0, 10**9))

    node_obs = []
    edge_obs = []

    node_conv = None
    edge_conv = None

    attack_packets_seen = 0

    while attack_packets_seen < max_attack_packets:
        # Normal traffic, slow
        for b in normal:
            for _ in range(NORMAL_RATE):
                node_sampler.forward(G, b)
                edge_sampler.forward(G, b)

        # Attacker traffic, x times faster
        for _ in range(x * NORMAL_RATE):
            # Node sampling packet
            pktN = node_sampler.forward(G, attacker)
            if pktN.node is not None:
                node_obs.append(pktN.node)

            # Edge sampling packet
            pktE = edge_sampler.forward(G, attacker)
            edge_obs.append((pktE.start, pktE.end, pktE.distance))

            attack_packets_seen += 1

            # Check node convergence
            if node_conv is None:
                ordered = node_reconstruct_order(node_obs)
                guess = node_guess_attacker_leaf(G, ordered)
                if guess == attacker:
                    node_conv = attack_packets_seen

            # Check edge convergence
            if edge_conv is None:
                by_d = edges_by_distance(edge_obs, victim=0)
                path = edge_reconstruct_path(by_d, victim=0)
                guess_edge = path[0] if path else None
                if guess_edge == attacker:
                    edge_conv = attack_packets_seen

            if node_conv is not None and edge_conv is not None:
                break

        if node_conv is not None and edge_conv is not None:
            break

    node_success = 1 if node_conv is not None else 0
    edge_success = 1 if edge_conv is not None else 0
    return node_success, edge_success, node_conv, edge_conv

@dataclass
class Stats:
    node_acc: float
    edge_acc: float
    node_mean_conv: float | None
    edge_mean_conv: float | None


def run_grid_one_attacker(G: nx.DiGraph, p_values: list[float], x_values: list[int], trials: int, seed: int = 0):
    """
    Runs for 1 attacker:
      p in (0.2, 0.4, 0.5, 0.6, 0.8)
      x in (10, 100, 1000)
    Returns dict keyed by (x,p)
    """
    rng = random.Random(seed)
    results: dict[tuple[int, float], Stats] = {}

    for x in x_values:
        for p in p_values:
            node_succ = 0
            edge_succ = 0
            node_convs = []
            edge_convs = []

            for t in range(trials):
                s = rng.randint(0, 10**9)
                n_ok, e_ok, n_conv, e_conv = run_trial_one_attacker(G, p=p, x=x, seed=s)

                node_succ += n_ok
                edge_succ += e_ok
                if n_conv is not None:
                    node_convs.append(n_conv)
                if e_conv is not None:
                    edge_convs.append(e_conv)

            results[(x, p)] = Stats(
                node_acc=node_succ / trials,
                edge_acc=edge_succ / trials,
                node_mean_conv=(sum(node_convs) / len(node_convs)) if node_convs else None,
                edge_mean_conv=(sum(edge_convs) / len(edge_convs)) if edge_convs else None,
            )

    return results