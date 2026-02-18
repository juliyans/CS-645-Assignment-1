import networkx as nx
from topology import load_topology, validate_tree_topology, leaves, branch_root_of
from ppm import choose_hosts
from ppm import NodeSampler

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

    # After printing leaves grouped by branch:
    print("\nHost selection testing")
    h1 = choose_hosts(G, num_attackers=1, seed=42)
    print(f"1 attacker: attackers={h1.attackers}, normal user={sorted(h1.normal_users)}")

    h2 = choose_hosts(G, num_attackers=2, seed=42)
    print(f"2 attackers: attackers={h2.attackers}, normal user={sorted(h2.normal_users)}")

    print("\n--- Node sampling marking test ---")
    sampler = NodeSampler(p=0.5, seed=1)

    test_leaf = h1.attackers[0]   # Use the chosen attacker leaf
    for i in range(10):
        pkt = sampler.forward(G, test_leaf)
        print(f"packet {i}: marked_node={pkt.node}")

if __name__ == "__main__":
    main()