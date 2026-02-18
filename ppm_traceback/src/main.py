# Name: Julianne Maghirang
# Date: February 17, 2026
# CS 645 Assignment 1: Probabilistic Packet Marking with Node and Edge Sampling
# Resources: 

import matplotlib.pyplot as plt
from src.topology import load_topology, validate_tree_topology
from src.experiment import run_grid_one_attacker, run_grid_two_attackers
from pathlib import Path

PLOT_DIR = Path("data/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

P_VALUES = [0.2, 0.4, 0.5, 0.6, 0.8]
X_VALUES = [10, 100, 1000]

def plot_accuracy(results, title_prefix: str):
    for x in X_VALUES:
        node = [results[(x, p)].node_acc for p in P_VALUES]
        edge = [results[(x, p)].edge_acc for p in P_VALUES]

        plt.figure()
        plt.plot(P_VALUES, node, marker="o", label="Node sampling")
        plt.plot(P_VALUES, edge, marker="o", label="Edge sampling")
        plt.xlabel("Marking probability p")
        plt.ylabel("Accuracy (attacker found)")
        plt.ylim(-0.05, 1.05)
        plt.title(f"{title_prefix}: Accuracy vs p (x={x})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR /f"{title_prefix}_accuracy_x{x}.png", dpi=200)

def plot_convergence(results, title_prefix: str):
    for x in X_VALUES:
        node = [results[(x, p)].node_mean_conv for p in P_VALUES]
        edge = [results[(x, p)].edge_mean_conv for p in P_VALUES]

        node_y = [float("nan") if v is None else v for v in node]
        edge_y = [float("nan") if v is None else v for v in edge]

        plt.figure()
        plt.plot(P_VALUES, node_y, marker="o", label="Node sampling")
        plt.plot(P_VALUES, edge_y, marker="o", label="Edge sampling")
        plt.xlabel("Marking probability p")
        plt.ylabel("Mean attack packets to converge")
        plt.title(f"{title_prefix}: Convergence vs p (x={x})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR /f"{title_prefix}_convergence_x{x}.png", dpi=200)

def main():
    G = load_topology("data/topology1.txt")
    validate_tree_topology(G)

    results1 = run_grid_one_attacker(G, P_VALUES, X_VALUES, trials=50, seed=123)
    plot_accuracy(results1, "Q1_single")
    plot_convergence(results1, "Q1_single")

    results2 = run_grid_two_attackers(G, P_VALUES, X_VALUES, trials=50, seed=456)
    plot_accuracy(results2, "Q2_two")
    plot_convergence(results2, "Q2_two")

    print(f"Done! Plots are saved in {PLOT_DIR}")

if __name__ == "__main__":
    main()