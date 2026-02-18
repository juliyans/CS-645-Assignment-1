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
# Plot helpers
# ---------------------------------------------------------------------------

def plot_accuracy_vs_p(results: dict, title_prefix: str) -> None:
    """
    Accuracy vs p — one line per x value, node and edge side by side.
    Covers assignment part (a).
    Physical significance: shows optimal p near 1/d. Too-high p causes
    downstream routers to overwrite upstream marks, losing path information.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {10: "tab:blue", 100: "tab:orange", 1000: "tab:green"}

    for x in X_VALUES:
        node = [results[(x, p)].node_acc for p in P_VALUES]
        edge = [results[(x, p)].edge_acc for p in P_VALUES]
        axes[0].plot(P_VALUES, node, marker="o", label=f"x={x}", color=colors[x])
        axes[1].plot(P_VALUES, edge, marker="s", label=f"x={x}", color=colors[x])

    for ax, name in zip(axes, ["Node sampling", "Edge sampling"]):
        ax.set_title(name)
        ax.set_xlabel("Marking probability p")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Attacker rate")

    axes[0].set_ylabel("Accuracy (fraction of trials attacker found)")
    fig.suptitle(f"{title_prefix}: Accuracy vs p", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{title_prefix}_accuracy_vs_p.png", dpi=200)
    plt.close()


def plot_accuracy_vs_x(results: dict, title_prefix: str) -> None:
    """
    Accuracy vs x — one line per p value, node and edge side by side.
    Covers assignment part (b).
    Physical significance: higher x = more attack packets per unit time,
    so the victim accumulates enough samples to reconstruct the path within
    the time limit. Shows that low x can fail at high p or deep paths.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {0.2: "tab:blue", 0.4: "tab:orange", 0.5: "tab:green",
              0.6: "tab:red", 0.8: "tab:purple"}

    for p in P_VALUES:
        node = [results[(x, p)].node_acc for x in X_VALUES]
        edge = [results[(x, p)].edge_acc for x in X_VALUES]
        axes[0].plot(X_VALUES, node, marker="o", label=f"p={p}", color=colors[p])
        axes[1].plot(X_VALUES, edge, marker="s", label=f"p={p}", color=colors[p])

    for ax, name in zip(axes, ["Node sampling", "Edge sampling"]):
        ax.set_title(name)
        ax.set_xlabel("Attacker rate multiplier x (log scale)")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Marking prob")

    axes[0].set_ylabel("Accuracy (fraction of trials attacker found)")
    fig.suptitle(f"{title_prefix}: Accuracy vs x", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{title_prefix}_accuracy_vs_x.png", dpi=200)
    plt.close()


def plot_convergence(results: dict, title_prefix: str) -> None:
    """
    Mean ticks to converge vs p — one line per x, node and edge side by side.
    Physical significance: fewer ticks = victim identifies attacker faster,
    enabling quicker filtering/mitigation of the DoS attack.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {10: "tab:blue", 100: "tab:orange", 1000: "tab:green"}

    for x in X_VALUES:
        node_y = [results[(x, p)].node_mean_conv or float("nan") for p in P_VALUES]
        edge_y = [results[(x, p)].edge_mean_conv or float("nan") for p in P_VALUES]
        axes[0].plot(P_VALUES, node_y, marker="o", label=f"x={x}", color=colors[x])
        axes[1].plot(P_VALUES, edge_y, marker="s", label=f"x={x}", color=colors[x])

    for ax, name in zip(axes, ["Node sampling", "Edge sampling"]):
        ax.set_title(name)
        ax.set_xlabel("Marking probability p")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Attacker rate")

    axes[0].set_ylabel("Mean ticks to converge")
    fig.suptitle(f"{title_prefix}: Convergence speed vs p", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{title_prefix}_convergence.png", dpi=200)
    plt.close()


def run_all_plots(results: dict, title_prefix: str) -> None:
    plot_accuracy_vs_p(results, title_prefix)   # part (a)
    plot_accuracy_vs_x(results, title_prefix)   # part (b)
    plot_convergence(results, title_prefix)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Two topologies to show different characteristics:
    #   topology1: 4 branches, max depth 5 hops -> shows x effect on convergence speed
    #   topology2: 3 branches, max depth 7 hops -> shows p effect (p=0.8 fails at low x)
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

        print("Running Q1 (1 attacker, 1 normal user)...")
        results1 = run_grid_one_attacker(G, P_VALUES, X_VALUES, trials=50, seed=123)
        run_all_plots(results1, f"Q1_{topo_label}")

        print("Running Q2 (2 attackers, 1 normal user)...")
        results2 = run_grid_two_attackers(G, P_VALUES, X_VALUES, trials=50, seed=456)
        run_all_plots(results2, f"Q2_{topo_label}")

        # Print summary tables
        print(f"\n--- Q1 accuracy summary ({topo_label}) ---")
        print(f"{'x':>6}  {'p':>5}  {'Node acc':>10}  {'Edge acc':>10}  {'Node conv':>10}  {'Edge conv':>10}")
        for x in X_VALUES:
            for p in P_VALUES:
                s = results1[(x, p)]
                nc = f"{s.node_mean_conv:.1f}" if s.node_mean_conv else "N/A"
                ec = f"{s.edge_mean_conv:.1f}" if s.edge_mean_conv else "N/A"
                print(f"{x:>6}  {p:>5.2f}  {s.node_acc:>10.3f}  {s.edge_acc:>10.3f}  {nc:>10}  {ec:>10}")

        print(f"\n--- Q2 accuracy summary ({topo_label}) ---")
        print(f"{'x':>6}  {'p':>5}  {'Node acc':>10}  {'Edge acc':>10}  {'Node conv':>10}  {'Edge conv':>10}")
        for x in X_VALUES:
            for p in P_VALUES:
                s = results2[(x, p)]
                nc = f"{s.node_mean_conv:.1f}" if s.node_mean_conv else "N/A"
                ec = f"{s.edge_mean_conv:.1f}" if s.edge_mean_conv else "N/A"
                print(f"{x:>6}  {p:>5.2f}  {s.node_acc:>10.3f}  {s.edge_acc:>10.3f}  {nc:>10}  {ec:>10}")

    print(f"\nDone! All plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()