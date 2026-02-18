# Name: Julianne Maghirang
# Date: February 17, 2026
# CS 645 Assignment 1: Probabilistic Packet Marking with Node and Edge Sampling

import matplotlib.pyplot as plt
from src.topology import load_topology, validate_tree_topology
from src.experiment import run_grid_one_attacker, run_grid_two_attackers
from pathlib import Path

PLOT_DIR = Path("data/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

P_VALUES = [0.2, 0.4, 0.5, 0.6, 0.8]
X_VALUES = [10, 100, 1000]


# ---------------------------------------------------------------------------
# Plot: Accuracy vs marking probability p  (one plot per x value)
# Covers assignment part (a): compare across p values
# ---------------------------------------------------------------------------

def plot_accuracy_vs_p(results: dict, title_prefix: str) -> None:
    """
    For each x value: plot node vs edge sampling accuracy as p varies.
    Physical significance: shows how marking probability affects traceback success.
    """
    for x in X_VALUES:
        node = [results[(x, p)].node_acc for p in P_VALUES]
        edge = [results[(x, p)].edge_acc for p in P_VALUES]

        plt.figure()
        plt.plot(P_VALUES, node, marker="o", label="Node sampling")
        plt.plot(P_VALUES, edge, marker="s", label="Edge sampling")
        plt.xlabel("Marking probability p")
        plt.ylabel("Accuracy (fraction of trials attacker found)")
        plt.ylim(-0.05, 1.05)
        plt.title(f"{title_prefix}: Accuracy vs p  (x={x})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{title_prefix}_accuracy_vs_p_x{x}.png", dpi=200)
        plt.close()


# ---------------------------------------------------------------------------
# Plot: Accuracy vs attacker rate multiplier x  (one plot per p value)
# Covers assignment part (b): compare across x values
# ---------------------------------------------------------------------------

def plot_accuracy_vs_x(results: dict, title_prefix: str) -> None:
    """
    For each p value: plot node vs edge sampling accuracy as x varies.
    Physical significance: shows how attacker traffic volume affects traceback.
    Higher x = more attack packets relative to normal traffic = easier to trace.
    """
    for p in P_VALUES:
        node = [results[(x, p)].node_acc for x in X_VALUES]
        edge = [results[(x, p)].edge_acc for x in X_VALUES]

        plt.figure()
        plt.plot(X_VALUES, node, marker="o", label="Node sampling")
        plt.plot(X_VALUES, edge, marker="s", label="Edge sampling")
        plt.xscale("log")
        plt.xlabel("Attacker rate multiplier x  (log scale)")
        plt.ylabel("Accuracy (fraction of trials attacker found)")
        plt.ylim(-0.05, 1.05)
        plt.title(f"{title_prefix}: Accuracy vs x  (p={p})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{title_prefix}_accuracy_vs_x_p{p}.png", dpi=200)
        plt.close()


# ---------------------------------------------------------------------------
# Plot: Mean convergence packets vs p  (one plot per x value)
# ---------------------------------------------------------------------------

def plot_convergence_vs_p(results: dict, title_prefix: str) -> None:
    """
    For each x value: plot mean packets to converge as p varies.
    Physical significance: shows how many attack packets the victim needs
    to receive before the attack path can be reconstructed.
    """
    for x in X_VALUES:
        node = [results[(x, p)].node_mean_conv for p in P_VALUES]
        edge = [results[(x, p)].edge_mean_conv for p in P_VALUES]

        node_y = [float("nan") if v is None else v for v in node]
        edge_y = [float("nan") if v is None else v for v in edge]

        plt.figure()
        plt.plot(P_VALUES, node_y, marker="o", label="Node sampling")
        plt.plot(P_VALUES, edge_y, marker="s", label="Edge sampling")
        plt.xlabel("Marking probability p")
        plt.ylabel("Mean attack packets to converge")
        plt.title(f"{title_prefix}: Convergence vs p  (x={x})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{title_prefix}_convergence_vs_p_x{x}.png", dpi=200)
        plt.close()


# ---------------------------------------------------------------------------
# Plot: Mean convergence packets vs x  (one plot per p value)
# ---------------------------------------------------------------------------

def plot_convergence_vs_x(results: dict, title_prefix: str) -> None:
    """
    For each p value: plot mean packets to converge as x varies.
    """
    for p in P_VALUES:
        node = [results[(x, p)].node_mean_conv for x in X_VALUES]
        edge = [results[(x, p)].edge_mean_conv for x in X_VALUES]

        node_y = [float("nan") if v is None else v for v in node]
        edge_y = [float("nan") if v is None else v for v in edge]

        plt.figure()
        plt.plot(X_VALUES, node_y, marker="o", label="Node sampling")
        plt.plot(X_VALUES, edge_y, marker="s", label="Edge sampling")
        plt.xscale("log")
        plt.xlabel("Attacker rate multiplier x  (log scale)")
        plt.ylabel("Mean attack packets to converge")
        plt.title(f"{title_prefix}: Convergence vs x  (p={p})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{title_prefix}_convergence_vs_x_p{p}.png", dpi=200)
        plt.close()


# ---------------------------------------------------------------------------
# Helper: run all plots for one result set
# ---------------------------------------------------------------------------

def run_all_plots(results: dict, title_prefix: str) -> None:
    plot_accuracy_vs_p(results, title_prefix)       # part (a)
    plot_accuracy_vs_x(results, title_prefix)       # part (b)
    plot_convergence_vs_p(results, title_prefix)
    plot_convergence_vs_x(results, title_prefix)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Run on both topology files as required by the assignment.
    # topology1: 5 leaves, varied depths (max 4 hops), richer branching.
    # topology2: 3 leaves, deep linear paths (max 7 hops).
    topologies = [
        ("data/topology1.txt", "topo1"),
        ("data/topology2.txt", "topo2"),
    ]

    for topo_path, topo_label in topologies:
        print(f"\n{'='*60}")
        print(f"Topology: {topo_path}")
        print(f"{'='*60}")

        G = load_topology(topo_path)
        validate_tree_topology(G)

        # Q1: single attacker + one normal user
        print("Running Q1 (1 attacker, 1 normal user)...")
        results1 = run_grid_one_attacker(G, P_VALUES, X_VALUES, trials=50, seed=123)
        run_all_plots(results1, f"Q1_{topo_label}")

        # Q2: two attackers + one normal user
        print("Running Q2 (2 attackers, 1 normal user)...")
        results2 = run_grid_two_attackers(G, P_VALUES, X_VALUES, trials=50, seed=456)
        run_all_plots(results2, f"Q2_{topo_label}")

        # Print a summary table to terminal
        print(f"\n--- Q1 accuracy summary ({topo_label}) ---")
        print(f"{'x':>6}  {'p':>5}  {'Node acc':>10}  {'Edge acc':>10}")
        for x in X_VALUES:
            for p in P_VALUES:
                s = results1[(x, p)]
                print(f"{x:>6}  {p:>5.2f}  {s.node_acc:>10.3f}  {s.edge_acc:>10.3f}")

        print(f"\n--- Q2 accuracy summary ({topo_label}) ---")
        print(f"{'x':>6}  {'p':>5}  {'Node acc':>10}  {'Edge acc':>10}")
        for x in X_VALUES:
            for p in P_VALUES:
                s = results2[(x, p)]
                print(f"{x:>6}  {p:>5.2f}  {s.node_acc:>10.3f}  {s.edge_acc:>10.3f}")

    print(f"\nDone! All plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
