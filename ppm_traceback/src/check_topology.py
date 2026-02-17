import networkx as nx
from topology import load_topology, validate_tree_topology, leaves, branch_root_of

def main():
    G = load_topology("data/topology1.txt")
    validate_tree_topology(G)

    leafs = leaves(G)

    # Group the leaves by branch
    by_branch = {}
    for lf in leafs:
        br = branch_root_of(G, lf)
        by_branch.setdefault(br, []).append(lf)


    print("Topology works")
    print(f"Routers: {len([n for n in G.nodes if n != 0])}")
    print(f"Branches from victim: {list(G.successors(0))}")
    print(f"Leaves: {leafs}")
    print("Leaves grouped by branch:")
    for br, lfs in sorted(by_branch.items()):
        print(f"  Branch {br}: {sorted(lfs)}")

if __name__ == "__main__":
    main()