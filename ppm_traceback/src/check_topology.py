from topology import load_topology, validate_tree_topology

def main():
    G = load_topology("data/topology1.txt")
    validate_tree_topology(G)

    routers = [n for n in G.nodes if n != 0]
    branches = list(G.successors(0))
    depths = dict(__import__("networkx").single_source_shortest_path_length(G, 0))

    print("Topology works")
    print(f"Routers: {len(routers)}")
    print(f"Branches from victim: {len(branches)} -> {branches}")
    print(f"Max depth: {max(depths.values())}")

if __name__ == "__main__":
    main()