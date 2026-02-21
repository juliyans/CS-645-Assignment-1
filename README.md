# maghirang_hw1_code

# CS 645 Assignment 1: Probabilistic Packet Marking with Node and Edge Sampling (Python)

## Install dependencies

```bash
pip install networkx matplotlib
```

## Usage

Run the main script under the ppm_traceback folder:
```bash
python -m src.main
```

### What this does
- Runs Question 1: one attacker and one normal user  
- Runs Question 2: two attackers and one normal user  
- Sweeps marking probabilities `p = {0.2, 0.4, 0.5, 0.6, 0.8}`
- Sweeps attacker rates `x = {10, 100, 1000}`
- Generates accuracy and convergence plots
- Prints summary tables to the console

Plots are saved to:

```
data/plots/
```

## Project Structure

### `src/main.py`
It runs all experiments, generates plots and prints results.

### `src/experiment.py`
This implements the packet-level simulations for Questions 1 and 2 and computes accuracy and convergence statistics.

### `src/ppm.py`
This script enforces assignment constraints (exactly one normal user, at most one attacker per branch) and re-exports sampling modules.

### `src/node_sampling.py`
Node sampling implementation: each packet records a single router ID and the attacker is inferred using frequency ordering

### `src/edge_sampling.py`
Edge sampling implementation: each packet records an edge and distance and the victim reconstructs paths or an attack graph.

### `src/topology.py`
This loads topology files, validates constraints and provides helper functions.

## Resources

### Python
  https://docs.python.org/3/library/random.html
  https://docs.python.org/3/library/dataclasses.html
  https://docs.python.org/3/library/collections.html

### NetworkX
  https://networkx.org/documentation/stable/
  https://networkx.org/documentation/stable/reference/classes/digraph.html
  https://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.successors.html  
  https://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.in_degree.html

### Matplotlib
  https://matplotlib.org/stable/api/pyplot_summary.html
  https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
  https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html
  https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html

Savage et al., *Practical Network Support for IP Traceback*  

https://www.makeareadme.com/