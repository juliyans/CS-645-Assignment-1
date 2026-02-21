import networkx as nx

VICTIM = 0  # Node 0 is always the victim, which is root of the tree

def load_topology(path: str) -> nx.DiGraph:
    # Read topology file and build a directed graph
    # Each line is "u v" meaning router u forwards packets toward router v
    G = nx.DiGraph()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v = map(int, line.split())
            G.add_edge(u, v)
    return G

def validate_tree_topology(G: nx.DiGraph) -> None:
    # Must follow all constraints before running experiments
    if VICTIM not in G.nodes:
        raise ValueError("Victim node 0 must exist")
    # Must be a directed tree: no cycles, one parent per node, fully connected
    if not nx.is_arborescence(G):
        raise ValueError("Topology must be a directed tree")
    # 10-20 routers excluding the victim
    routers = [n for n in G.nodes if n != VICTIM]
    if not (10 <= len(routers) <= 20):
        raise ValueError(f"Router count must be 10-20, found {len(routers)}")
    # 3-5 direct branches from the victim
    if len(list(G.successors(VICTIM))) not in [3, 4, 5]:
        raise ValueError("Branch count must be 3-5")
    # No path longer than 15 hops
    depths = nx.single_source_shortest_path_length(G, VICTIM)
    if max(depths.values()) > 15:
        raise ValueError("Max depth must be <= 15")

def leaves(G: nx.DiGraph) -> list[int]:
    # Leaf nodes are routers with no successors or where endpoints that hosts connect at
    return [n for n in G.nodes if n != VICTIM and G.out_degree(n) == 0]

def branch_root_of(G: nx.DiGraph, node: int) -> int:
    # Go through the tree until it reaches a direct child of the victim
    # This identifies which branch a given node belongs to
    cur = node
    while list(G.predecessors(cur))[0] != VICTIM:
        cur = list(G.predecessors(cur))[0]
    return cur

def path_leaf_to_victim(G: nx.DiGraph, leaf: int) -> list[int]:
    # Return ordered list of routers from source leaf toward the victim
    # Used to simulate a packet travelling through the network
    path, cur = [], leaf
    while cur != VICTIM:
        path.append(cur)
        cur = list(G.predecessors(cur))[0]
    return path