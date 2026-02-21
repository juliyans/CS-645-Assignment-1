# Name: Julianne Maghirang
# Date: February 17, 2026
# CS 645 Assignment 1: Probabilistic Packet Marking with Node and Edge Sampling

import matplotlib.pyplot as plt
from pathlib import Path
from src.topology import load_topology, validate_tree_topology
from src.experiment import run_grid_one_attacker, run_grid_two_attackers

# Directory to save all generated plots
PLOT_DIR = Path("data/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

P_VALUES = [0.2, 0.4, 0.5, 0.6, 0.8] # Marking probabilities to sweep
X_VALUES = [10, 100, 1000] # Attacker rate multipliers

# Helper to create a figure with two subplots (node left, edge right)
def _side_by_side(title, xlabel, xscale=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True) # sharey=True keeps the y-axis scale the same
    for ax, name in zip(axes, ["Node sampling", "Edge sampling"]):
        ax.set_title(name); ax.set_xlabel(xlabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        if xscale: ax.set_xscale(xscale)
    axes[0].set_ylabel("Accuracy (fraction of trials attacker found)")
    fig.suptitle(title, fontsize=13)
    return fig, axes

# Shows how marking probability p affects accuracy
def plot_accuracy_vs_p(results, prefix):
    # Each line is one attacker rate x with both algorithms side by side
    # Optimal p = 1/d; higher p causes downstream
    # Routers to overwrite upstream marks, degrading edge sampling accuracy
    fig, axes = _side_by_side(f"{prefix}: Accuracy vs p", "Marking probability p")
    colors = {10: "tab:blue", 100: "tab:orange", 1000: "tab:green"}
    for x in X_VALUES:
        axes[0].plot(P_VALUES, [results[(x,p)].node_acc for p in P_VALUES], "o-", label=f"x={x}", color=colors[x])
        axes[1].plot(P_VALUES, [results[(x,p)].edge_acc for p in P_VALUES], "s-", label=f"x={x}", color=colors[x])
    for ax in axes: ax.set_ylim(-0.05, 1.05); ax.legend(title="Attacker rate")
    plt.tight_layout(); plt.savefig(PLOT_DIR / f"{prefix}_accuracy_vs_p.png", dpi=200); plt.close()

# Shows how attacker rate x affects accuracy
def plot_accuracy_vs_x(results, prefix):
    # Each line is one marking probability p & log scale on x-axis
    # Higher x = more attack packets per tick =
    # Victim accumulates enough samples to reconstruct path within time limit
    fig, axes = _side_by_side(f"{prefix}: Accuracy vs x", "Attacker rate multiplier x (log scale)", xscale="log")
    colors = {0.2:"tab:blue",0.4:"tab:orange",0.5:"tab:green",0.6:"tab:red",0.8:"tab:purple"}
    for p in P_VALUES:
        axes[0].plot(X_VALUES, [results[(x,p)].node_acc for x in X_VALUES], "o-", label=f"p={p}", color=colors[p])
        axes[1].plot(X_VALUES, [results[(x,p)].edge_acc for x in X_VALUES], "s-", label=f"p={p}", color=colors[p])
    for ax in axes: ax.set_ylim(-0.05, 1.05); ax.legend(title="Marking prob")
    plt.tight_layout(); plt.savefig(PLOT_DIR / f"{prefix}_accuracy_vs_x.png", dpi=200); plt.close()

# Shows how quickly each algorithm identifies the attacker (in ticks)
def plot_convergence(results, prefix):
    # Only counts successful trials & NaN shown where algorithm never converged
    # Faster convergence means victim can filter attack traffic sooner
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {10: "tab:blue", 100: "tab:orange", 1000: "tab:green"}
    for x in X_VALUES:
        axes[0].plot(P_VALUES, [results[(x,p)].node_mean_conv or float("nan") for p in P_VALUES], "o-", label=f"x={x}", color=colors[x])
        axes[1].plot(P_VALUES, [results[(x,p)].edge_mean_conv or float("nan") for p in P_VALUES], "s-", label=f"x={x}", color=colors[x])
    for ax, name in zip(axes, ["Node sampling", "Edge sampling"]):
        ax.set_title(name); ax.set_xlabel("Marking probability p")
        ax.grid(True, linestyle="--", alpha=0.4); ax.legend(title="Attacker rate")
    axes[0].set_ylabel("Mean ticks to converge")
    fig.suptitle(f"{prefix}: Convergence speed vs p", fontsize=13)
    plt.tight_layout(); plt.savefig(PLOT_DIR / f"{prefix}_convergence.png", dpi=200); plt.close()


def main():
    for path, label in [("data/topology1.txt", "topo1"), ("data/topology2.txt", "topo2")]:
        print(f"\nTopology: {path}")
        G = load_topology(path)
        validate_tree_topology(G)

        # Run Q1 and Q2 for this topology and save all plots
        for q, run, na in [("Q1", run_grid_one_attacker, 1), ("Q2", run_grid_two_attackers, 2)]:
            print(f"Running {q} ({na} attacker{'s' if na > 1 else ''}, 1 normal user)...")
            results = run(G, P_VALUES, X_VALUES, trials=50, seed=123 if q == "Q1" else 456)
            prefix = f"{q}_{label}"
            plot_accuracy_vs_p(results, prefix)
            plot_accuracy_vs_x(results, prefix)
            plot_convergence(results, prefix)

            # Print summary table for checking
            print(f"\n{q} accuracy summary ({label})")
            print(f"{'x':>6}  {'p':>5}  {'Node acc':>10}  {'Edge acc':>10}  {'Node conv':>10}  {'Edge conv':>10}")
            for x in X_VALUES:
                for p in P_VALUES:
                    s = results[(x, p)]
                    nc = f"{s.node_mean_conv:.1f}" if s.node_mean_conv else "N/A"
                    ec = f"{s.edge_mean_conv:.1f}" if s.edge_mean_conv else "N/A"
                    print(f"{x:>6}  {p:>5.2f}  {s.node_acc:>10.3f}  {s.edge_acc:>10.3f}  {nc:>10}  {ec:>10}")

    print(f"\nDone! Plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()