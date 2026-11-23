# Advanced Algorithms Project: Biased Sampling for Narrow Passages

## 📝 Short Description
This project focuses on motion planning in continuous spaces, specifically addressing the "narrow passage" problem. Standard uniform sampling rarely hits narrow passages because the hit probability is directly proportional to the volume of the passage. 

This repository implements the Probabilistic Roadmap (PRM) algorithm and compares standard uniform sampling against heuristic biased samplers (Gaussian, Bridge test, Obstacle-based) and samplers learned from experience (CVAE, Ichter et al. 2018). The goal is to evaluate these approaches based on success rates, runtime efficiency, and the number of collision checks required to find a feasible path.

## 🛠 Tech Stack & Libraries
*   **Language:** Python 3.12
*   **Collision Checking & Geometry:** `shapely` (Core infrastructure for map and obstacle definitions)
*   **Graph Operations:** `networkx` (For pathfinding and connected components)
*   **Data Structures:** `scipy.spatial` (KDTree for efficient nearest-neighbor searches)
*   **Visualization:** `matplotlib` / `seaborn` (For plotting roadmaps and benchmark results)

---
## Pseudocode PRM (Probabilistic Roadmap with Biased Sampling)

```
1: V <- {q_start, q_goal}, E <- empty set
2: while samples < N do
3:     q_rand <- BiasedSampler(type) // Uniform, Gaussian, or Bridge-test
4:     if q_rand is in C_free then
5:         V <- V union {q_rand}
6:         Neighbors <- KNearestNeighbors(q_rand, V, k)
7:         for q_near in Neighbors do
8:             if LocalPlanner(q_rand, q_near) is collision-free then
9:                 E <- E union {(q_rand, q_near)}
10:    if PathExists(q_start, q_goal, G(V, E)) then
11:        return ShortestPath(G)
12: return FAILURE

```

* **Invariants:** Every vertex $v \in V$ lies entirely in $\mathcal{C}_{free}$, and every edge $e \in E$ represents a continuous, collision-free local path in $\mathcal{C}_{free}$.

* **Termination:** The algorithm terminates either when a connected path between $q_{start}$ and $q_{goal}$ is found, or when a predefined budget (maximum iterations $N$ or a time limit) is exhausted.

* **Type of Guarantee:** The algorithm is **probabilistically complete**. As $N \to \infty$, the probability of finding a path, if one exists, approaches 1. However, to maintain this theoretical guarantee in practice, uniform samples must be mixed in with biased samples (Gaussian/Bridge-test) to ensure the entire free space is sufficiently covered and not just the obstacle boundaries. It is not asymptotically optimal unless modified to a PRM\* variant.



## 📋 Task Breakdown

### 1. Core Framework & PRM Infrastructure
- [x] Set up the 2D environment and obstacle representation (using `Shapely`).
- [x] Implement `CollisionChecker`: Verify if a configuration node is in the free space.
- [x] Implement `LocalPlanner(c, n)`: Check for collisions along the straight line connecting two nodes.
- [x] Implement the neighbor search algorithm (K-Nearest Neighbors or Radius Search) to define `N_c`.
- [x] Implement the main PRM loop (manage Node/Edge sets and check `same_connected_component`).
- [x] Integrate a shortest-path algorithm (A* or Dijkstra) on the generated roadmap (can use `networkx`).

### 2. Sampler Implementations
- [x] **Baseline:** Implement `UniformSampler`.
- [x] **Heuristic 1:** Implement `GaussianSampler` (samples near obstacle boundaries).
- [x] **Heuristic 2:** Implement `BridgeTestSampler` (designed for narrow passages).
- [x] **Heuristic 3:** Implement `ObstacleBasedSampler` (OBPRM).
- [x] **Learned-based:** Implement `CVAESampler` (Conditional Variational Autoencoder, per Ichter et al. 2018).

### 3. Metrics & Language-Independent Benchmarks
- [x] Inject internal counters into the PRM class: `num_collision_checks`.
- [x] Inject internal counters: `num_nodes_expanded`.
- [x] Measure runtime execution for distinct phases (Sampling, Edge Connection, Pathfinding).

### 4. Benchmarking & Visualization
- [x] Design custom 2D test maps featuring varying widths of narrow passages.
- [x] Write an automated benchmarking script: Execute 100+ runs per sampler across different maps to calculate the success rate.
- [x] Write a script to export benchmark results to CSV format.
- [x] Write a visualization script (`matplotlib`/`seaborn`) to render the 2D map and the generated roadmaps.
- [x] Generate statistical plots comparing Uniform, Heuristic, and CVAE samplers based on runtime and collision checks.