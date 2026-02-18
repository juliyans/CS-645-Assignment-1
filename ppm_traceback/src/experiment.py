import random
import networkx as nx
from dataclasses import dataclass
from src.ppm import (
    choose_hosts,
    NodeSampler, EdgeSampler,
    node_reconstruct_order, node_guess_attacker_leaf,
    edges_by_distance, edge_reconstruct_path,
    node_guess_two_attackers,
    edge_build_graph, edge_guess_attackers_from_graph,
)

NORMAL_RATE = 1    # normal user sends 1 packet per simulation tick
MAX_TICKS   = 500  # simulation time limit in ticks
               # attack packets available = x * MAX_TICKS per trial
               # x=10  -> up to 5,000 attack packets
               # x=100 -> up to 50,000 attack packets
               # x=1000-> up to 500,000 attack packets
               # This makes x affect ACCURACY (not just convergence speed),
               # matching the assignment intent that higher x = stronger signal.


# ---------------------------------------------------------------------------
# Q1: Single attacker + one normal user
# ---------------------------------------------------------------------------

def run_trial_one_attacker(
    G: nx.DiGraph,
    p: float,
    x: int,
    seed: int,
    max_ticks: int = MAX_TICKS,
) -> tuple[int, int, int | None, int | None]:
    """
    One simulation trial: 1 attacker, 1 normal user.

    Each tick:
      - Normal user sends NORMAL_RATE packet(s)  [slow background traffic]
      - Attacker sends x * NORMAL_RATE packets    [x times faster]

    Runs for at most max_ticks ticks. Larger x gives the victim more marked
    attack packets per tick, so accuracy increases with x.

    Returns (node_success, edge_success, node_conv_ticks, edge_conv_ticks).
    """
    rng = random.Random(seed)
    hosts = choose_hosts(G, num_attackers=1, num_normal=1, seed=seed + 1)
    attacker = hosts.attackers[0]
    normal   = hosts.normal_users

    node_sampler = NodeSampler(p=p, seed=rng.randint(0, 10**9))
    edge_sampler = EdgeSampler(p=p, seed=rng.randint(0, 10**9))

    node_obs: list[int]   = []
    edge_obs: list[tuple] = []
    node_conv: int | None = None
    edge_conv: int | None = None

    for tick in range(1, max_ticks + 1):
        # Normal user background traffic
        for b in normal:
            for _ in range(NORMAL_RATE):
                node_sampler.forward(G, b)
                edge_sampler.forward(G, b)

        # Attacker traffic: x packets per tick
        for _ in range(x * NORMAL_RATE):
            pktN = node_sampler.forward(G, attacker)
            if pktN.node is not None:
                node_obs.append(pktN.node)

            pktE = edge_sampler.forward(G, attacker)
            edge_obs.append((pktE.start, pktE.end, pktE.distance))

        # Check convergence once per tick (after all packets this tick)
        if node_conv is None:
            ordered = node_reconstruct_order(node_obs)
            guess = node_guess_attacker_leaf(G, ordered, node_obs=node_obs)
            if guess == attacker:
                node_conv = tick

        if edge_conv is None:
            by_d = edges_by_distance(edge_obs, victim=0)
            path = edge_reconstruct_path(by_d, victim=0)
            # path[0] = farthest router (attacker side) after reverse()
            guess_edge = path[0] if path else None
            if guess_edge == attacker:
                edge_conv = tick

        if node_conv is not None and edge_conv is not None:
            break

    node_success = 1 if node_conv is not None else 0
    edge_success = 1 if edge_conv is not None else 0
    return node_success, edge_success, node_conv, edge_conv


@dataclass
class Stats:
    node_acc: float
    edge_acc: float
    node_mean_conv: float | None   # mean ticks to convergence (successful trials only)
    edge_mean_conv: float | None


def run_grid_one_attacker(
    G: nx.DiGraph,
    p_values: list[float],
    x_values: list[int],
    trials: int,
    seed: int = 0,
) -> dict[tuple[int, float], Stats]:
    """Run Q1 grid over all (x, p) combinations. Returns dict keyed by (x, p)."""
    rng = random.Random(seed)
    results: dict[tuple[int, float], Stats] = {}

    for x in x_values:
        for p in p_values:
            node_succ = 0
            edge_succ = 0
            node_convs: list[int] = []
            edge_convs: list[int] = []

            for _ in range(trials):
                s = rng.randint(0, 10**9)
                n_ok, e_ok, n_conv, e_conv = run_trial_one_attacker(G, p=p, x=x, seed=s)
                node_succ += n_ok
                edge_succ += e_ok
                if n_conv is not None: node_convs.append(n_conv)
                if e_conv is not None: edge_convs.append(e_conv)

            results[(x, p)] = Stats(
                node_acc      = node_succ / trials,
                edge_acc      = edge_succ / trials,
                node_mean_conv= sum(node_convs) / len(node_convs) if node_convs else None,
                edge_mean_conv= sum(edge_convs) / len(edge_convs) if edge_convs else None,
            )

    return results


# ---------------------------------------------------------------------------
# Q2: Two attackers + one normal user
# ---------------------------------------------------------------------------

def run_trial_two_attackers(
    G: nx.DiGraph,
    p: float,
    x: int,
    seed: int,
    max_ticks: int = MAX_TICKS,
) -> tuple[int, int, int | None, int | None]:
    """
    One simulation trial: 2 attackers (different branches), 1 normal user.

    Convergence = victim identifies EXACTLY both true attacker leaves
    (exact set match: no missing attackers, no false positives).

    Returns (node_success, edge_success, node_conv_ticks, edge_conv_ticks).
    """
    rng = random.Random(seed)
    hosts     = choose_hosts(G, num_attackers=2, num_normal=1, seed=seed + 1)
    attackers = hosts.attackers[:]
    normal    = hosts.normal_users

    node_sampler = NodeSampler(p=p, seed=rng.randint(0, 10**9))
    edge_sampler = EdgeSampler(p=p, seed=rng.randint(0, 10**9))

    node_obs: list[int]   = []
    edge_obs: list[tuple] = []
    node_conv: int | None = None
    edge_conv: int | None = None
    true_set = set(attackers)

    for tick in range(1, max_ticks + 1):
        # Normal user background traffic
        for b in normal:
            for _ in range(NORMAL_RATE):
                node_sampler.forward(G, b)
                edge_sampler.forward(G, b)

        # Both attackers each send x packets per tick
        for a in attackers:
            for _ in range(x * NORMAL_RATE):
                pktN = node_sampler.forward(G, a)
                if pktN.node is not None:
                    node_obs.append(pktN.node)

                pktE = edge_sampler.forward(G, a)
                edge_obs.append((pktE.start, pktE.end, pktE.distance))

        # Check convergence once per tick
        if node_conv is None:
            guesses = node_guess_two_attackers(G, node_obs, max_attackers=2)
            if set(guesses) == true_set:
                node_conv = tick

        if edge_conv is None:
            H = edge_build_graph(edge_obs, victim=0)
            guesses_edge = edge_guess_attackers_from_graph(H, victim=0)
            if set(guesses_edge) == true_set:   # exact match required
                edge_conv = tick

        if node_conv is not None and edge_conv is not None:
            break

    node_success = 1 if node_conv is not None else 0
    edge_success = 1 if edge_conv is not None else 0
    return node_success, edge_success, node_conv, edge_conv


def run_grid_two_attackers(
    G: nx.DiGraph,
    p_values: list[float],
    x_values: list[int],
    trials: int,
    seed: int = 0,
) -> dict[tuple[int, float], Stats]:
    """Run Q2 grid over all (x, p) combinations. Returns dict keyed by (x, p)."""
    rng = random.Random(seed)
    results: dict[tuple[int, float], Stats] = {}

    for x in x_values:
        for p in p_values:
            node_succ = 0
            edge_succ = 0
            node_convs: list[int] = []
            edge_convs: list[int] = []

            for _ in range(trials):
                s = rng.randint(0, 10**9)
                n_ok, e_ok, n_conv, e_conv = run_trial_two_attackers(G, p=p, x=x, seed=s)
                node_succ += n_ok
                edge_succ += e_ok
                if n_conv is not None: node_convs.append(n_conv)
                if e_conv is not None: edge_convs.append(e_conv)

            results[(x, p)] = Stats(
                node_acc      = node_succ / trials,
                edge_acc      = edge_succ / trials,
                node_mean_conv= sum(node_convs) / len(node_convs) if node_convs else None,
                edge_mean_conv= sum(edge_convs) / len(edge_convs) if edge_convs else None,
            )

    return results
