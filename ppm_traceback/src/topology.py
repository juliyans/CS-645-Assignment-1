import networkx as nx

# Victim node ID or root of the tree
VICTIM = 0

# Router count constraints, excluding victim
ROUTER_MIN = 10
ROUTER_MAX = 20

# Max allowed hop count from attacker to victim
MAX_HOPS = 15

# Allowed # of branches from the victim, outdegree of node 0
BRANCH_CHOICES = [3, 4, 5]

# Load a network topology from a text file
def load_topology(path: str) -> nx.DiGraph:
    # Parent child
    # Node 0 is the victim
    # Edges are directed outward from the victim
    G = nx.DiGraph()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            # Each line defines a directed edge of parent to child
            u, v = map(int, line.split())
            G.add_edge(u, v)

    # Returns a directed graph of the topology.
    return G

# Check that the topology satisfies constraints
def validate_tree_topology(G: nx.DiGraph) -> None:
    # Victim check
    if VICTIM not in G.nodes:
        raise ValueError("Victim node 0 exists")

    # Topology is a directed tree 
    # No cycles
    # Exactly one parent per node, except root
    # Connected
    if not nx.is_arborescence(G):
        raise ValueError("Topology is a directed tree")

    # Victim is at the root of the tree
    if G.in_degree(VICTIM) != 0:
        raise ValueError("Victim is the root")

    # Router count is between 10 and 20 (excluding victim)
    routers = [n for n in G.nodes if n != VICTIM]
    if not (ROUTER_MIN <= len(routers) <= ROUTER_MAX):
        raise ValueError(
            f"Router count must be {ROUTER_MIN}-{ROUTER_MAX} excluding victim. "
            f"Found {len(routers)}."
        )

    # Branch count constraint/number of direct children of the victim
    branches = list(G.successors(VICTIM))
    if len(branches) not in BRANCH_CHOICES:
        raise ValueError(
            f"Branch count must be in {BRANCH_CHOICES}. Found {len(branches)}."
        )

    # Hop constraint that computes shortest-path distance from victim to every node
    depths = nx.single_source_shortest_path_length(G, VICTIM)
    max_depth = max(depths.values())

    if max_depth > MAX_HOPS:
        raise ValueError(
            f"Max hop depth must be <= {MAX_HOPS}. Found {max_depth}."
        )