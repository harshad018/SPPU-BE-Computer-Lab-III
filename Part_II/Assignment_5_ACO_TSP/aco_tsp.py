"""
Assignment 5 – Ant Colony Optimization (ACO) for the Travelling Salesman Problem
=================================================================================
Implements ACO to find the shortest route that visits every city exactly once
and returns to the starting city.

Features
--------
- Configurable α (pheromone weight), β (heuristic weight), ρ (evaporation rate)
- Reproducible results via seed
- Optional matplotlib tour plot (use --plot flag)
- Works on random or predefined city layouts

Usage
-----
    python aco_tsp.py                          # 20 random cities, default params
    python aco_tsp.py --cities 30 --ants 50   # 30 cities, 50 ants
    python aco_tsp.py --seed 7 --plot          # reproducible + tour plot
    python aco_tsp.py --help

Requirements
------------
    pip install numpy matplotlib
"""

import argparse
import math
import random
import sys
import numpy as np


# ---------------------------------------------------------------------------
# City generation & distance matrix
# ---------------------------------------------------------------------------

def generate_cities(n: int, seed: int = 42) -> np.ndarray:
    """Return (n, 2) array of random city (x, y) coordinates in [0, 100]."""
    rng = np.random.RandomState(seed)
    return rng.rand(n, 2) * 100.0


def distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix."""
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = cities[i] - cities[j]
                dist[i][j] = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
    return dist


# ---------------------------------------------------------------------------
# ACO
# ---------------------------------------------------------------------------

class AntColonyTSP:
    """Ant Colony Optimisation solver for TSP.

    Parameters
    ----------
    dist      : square distance matrix (n x n)
    n_ants    : number of ants per iteration
    n_iter    : number of iterations
    alpha     : pheromone importance (default 1.0)
    beta      : heuristic (1/distance) importance (default 5.0)
    rho       : pheromone evaporation rate in [0, 1] (default 0.5)
    Q         : pheromone deposit constant (default 100)
    seed      : random seed
    """

    def __init__(self, dist: np.ndarray, n_ants: int = 20, n_iter: int = 100,
                 alpha: float = 1.0, beta: float = 5.0, rho: float = 0.5,
                 Q: float = 100.0, seed: int = 42):
        self.dist = dist
        self.n = dist.shape[0]
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.rng = np.random.RandomState(seed)
        # Initialise pheromone matrix
        self.pheromone = np.ones((self.n, self.n))

    def _heuristic(self) -> np.ndarray:
        """Return 1/dist (avoiding division by zero on diagonal)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(self.dist > 0, 1.0 / self.dist, 0.0)
        return h

    def _construct_tour(self, heuristic: np.ndarray) -> list:
        """Build one ant's tour using probabilistic city selection."""
        start = int(self.rng.randint(0, self.n))
        visited = [start]
        unvisited = set(range(self.n)) - {start}

        while unvisited:
            current = visited[-1]
            # Compute transition probabilities
            ph = np.array([
                (self.pheromone[current][j] ** self.alpha) *
                (heuristic[current][j] ** self.beta)
                for j in unvisited
            ])
            total = ph.sum()
            if total == 0:
                # Fallback: pick randomly
                next_city = int(self.rng.choice(list(unvisited)))
            else:
                probs = ph / total
                next_city = int(self.rng.choice(list(unvisited), p=probs))
            visited.append(next_city)
            unvisited.remove(next_city)

        return visited

    def _tour_length(self, tour: list) -> float:
        """Compute total tour distance (including return to start)."""
        total = 0.0
        for i in range(len(tour)):
            total += self.dist[tour[i]][tour[(i + 1) % len(tour)]]
        return total

    def _update_pheromone(self, all_tours: list, all_lengths: list):
        """Evaporate pheromone and deposit from each ant's tour."""
        self.pheromone *= (1.0 - self.rho)
        for tour, length in zip(all_tours, all_lengths):
            deposit = self.Q / length
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i + 1) % len(tour)]
                self.pheromone[a][b] += deposit
                self.pheromone[b][a] += deposit

    def solve(self, verbose: bool = True) -> tuple:
        """Run ACO and return (best_tour, best_length, history).

        history – list of best lengths per iteration.
        """
        heuristic = self._heuristic()
        best_tour = None
        best_length = float("inf")
        history = []

        for iteration in range(self.n_iter):
            all_tours, all_lengths = [], []
            for _ in range(self.n_ants):
                tour = self._construct_tour(heuristic)
                length = self._tour_length(tour)
                all_tours.append(tour)
                all_lengths.append(length)
                if length < best_length:
                    best_length = length
                    best_tour = tour[:]

            self._update_pheromone(all_tours, all_lengths)
            history.append(best_length)

            if verbose and (iteration % 10 == 0 or iteration == self.n_iter - 1):
                print(f"  Iteration {iteration + 1:>4}/{self.n_iter}  "
                      f"Best tour length: {best_length:.2f}")

        return best_tour, best_length, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tour(cities: np.ndarray, tour: list, length: float):
    """Plot the best tour using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    # Draw tour edges
    for i in range(len(tour)):
        a, b = tour[i], tour[(i + 1) % len(tour)]
        ax.plot([cities[a][0], cities[b][0]],
                [cities[a][1], cities[b][1]], "b-", linewidth=1)
    # Draw cities
    ax.scatter(cities[:, 0], cities[:, 1], c="red", zorder=5, s=60)
    # Label start city
    start = tour[0]
    ax.scatter([cities[start][0]], [cities[start][1]],
               c="green", zorder=6, s=100, marker="*", label="Start")
    ax.set_title(f"ACO TSP Best Tour  (length = {length:.2f})")
    ax.legend()
    plt.tight_layout()
    out_file = "aco_tsp_tour.png"
    plt.savefig(out_file, dpi=100)
    print(f"  Tour plot saved to: {out_file}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ACO for Travelling Salesman Problem."
    )
    parser.add_argument("--cities", type=int, default=20,
                        help="Number of cities (default: 20)")
    parser.add_argument("--ants", type=int, default=20,
                        help="Ants per iteration (default: 20)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations (default: 100)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Pheromone weight α (default: 1.0)")
    parser.add_argument("--beta", type=float, default=5.0,
                        help="Heuristic weight β (default: 5.0)")
    parser.add_argument("--rho", type=float, default=0.5,
                        help="Evaporation rate ρ (default: 0.5)")
    parser.add_argument("--Q", type=float, default=100.0,
                        help="Pheromone deposit constant Q (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the best tour (requires matplotlib)")
    args = parser.parse_args()

    print("=" * 60)
    print("Ant Colony Optimisation – Travelling Salesman Problem")
    print("=" * 60)
    print(f"  Cities     : {args.cities}")
    print(f"  Ants       : {args.ants}")
    print(f"  Iterations : {args.iterations}")
    print(f"  α={args.alpha}  β={args.beta}  ρ={args.rho}  Q={args.Q}")
    print(f"  Seed       : {args.seed}")

    cities = generate_cities(args.cities, seed=args.seed)
    dist = distance_matrix(cities)

    print(f"\n[Running ACO ...]\n")
    aco = AntColonyTSP(
        dist=dist,
        n_ants=args.ants,
        n_iter=args.iterations,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        Q=args.Q,
        seed=args.seed,
    )
    best_tour, best_length, history = aco.solve(verbose=True)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Best tour length : {best_length:.4f}")
    print(f"  Best tour order  : {best_tour}")
    print(f"  Improvement      : {history[0]:.2f} → {history[-1]:.2f}")

    if args.plot:
        plot_tour(cities, best_tour, best_length)

    print("\nDone.")


if __name__ == "__main__":
    main()
