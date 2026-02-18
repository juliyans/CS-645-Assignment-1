import random
import networkx as nx

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