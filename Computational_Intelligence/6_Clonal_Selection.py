"""
Practical 7: Clonal Selection Algorithm (CLONALG)
============================================================
Implements the Clonal Selection Algorithm from Artificial Immune
Systems (AIS) theory and applies it to minimise the 2-D Rastrigin
function — a challenging multimodal benchmark with many local minima.

Algorithm steps (per generation):
    1. Evaluate the affinity of every antibody in the population.
    2. Select the top-n antibodies (highest affinity = nearest to optimum).
    3. Clone each selected antibody proportionally to its affinity.
    4. Hypermutate all clones (mutation rate inversely proportional to affinity).
    5. Re-evaluate clones; keep the best clone from each clone group.
    6. Combine survivors with the original population; retain the top
       (pop_size - n_replace) individuals.
    7. Inject n_replace fresh random antibodies (receptor editing /
       diversity maintenance).

Objective function:
    Rastrigin(x, y) = 20 + x^2 - 10*cos(2*pi*x) + y^2 - 10*cos(2*pi*y)
    Global minimum : f(0, 0) = 0  in x,y in [-5.12, 5.12]

Reference:
    de Castro & Von Zuben (2002). "Learning and Optimization Using the
    Clonal Selection Principle." IEEE Trans. Evol. Comput., 6(3), 239-251.

Usage:
    python 6_Clonal_Selection.py

Dependencies: numpy, matplotlib
"""

import math
import os
import random

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
BOUNDS = (-5.12, 5.12)


def rastrigin(x: float, y: float) -> float:
    """2-D Rastrigin function.  Global minimum = 0 at (0, 0).

    Args:
        x: First coordinate.
        y: Second coordinate.

    Returns:
        Function value (non-negative).
    """
    return (
        20.0
        + x ** 2 - 10.0 * math.cos(2 * math.pi * x)
        + y ** 2 - 10.0 * math.cos(2 * math.pi * y)
    )


def affinity(antibody: np.ndarray) -> float:
    """Compute antibody affinity = 1 / (1 + f(x, y)).

    Higher affinity means a better (lower-cost) solution.

    Args:
        antibody: 1-D array [x, y].

    Returns:
        Affinity value in (0, 1].
    """
    return 1.0 / (1.0 + rastrigin(antibody[0], antibody[1]))


# ---------------------------------------------------------------------------
# CLONALG core
# ---------------------------------------------------------------------------

def random_antibody() -> np.ndarray:
    """Create a random antibody (candidate solution) within the search space."""
    lo, hi = BOUNDS
    return np.array([random.uniform(lo, hi), random.uniform(lo, hi)])


def hypermutate(antibody: np.ndarray, aff: float,
                max_mutation: float = 2.0) -> np.ndarray:
    """Apply inversely proportional hypermutation to an antibody.

    Higher affinity (closer to optimum) → smaller perturbation.

    Args:
        antibody:     The antibody to mutate.
        aff:          Its affinity value.
        max_mutation: Maximum mutation strength at zero affinity.

    Returns:
        Mutated antibody clipped to the search-space bounds.
    """
    strength = max_mutation * math.exp(-aff)
    lo, hi = BOUNDS
    mutant = antibody + np.random.uniform(-strength, strength, size=antibody.shape)
    return np.clip(mutant, lo, hi)


def clonal_selection(
    pop_size: int = 30,
    n_select: int = 10,
    clone_factor: int = 5,
    n_replace: int = 5,
    generations: int = 50,
) -> tuple:
    """Run the Clonal Selection Algorithm (CLONALG).

    Args:
        pop_size:     Total population size.
        n_select:     Number of best antibodies selected for cloning each generation.
        clone_factor: Number of clones generated per selected antibody.
        n_replace:    Number of worst antibodies replaced by random newcomers.
        generations:  Number of iterations.

    Returns:
        (best_antibody, best_value, history)
        history is a list of (generation, best_f, best_position) tuples.
    """
    # 1. Initialise population
    population = [random_antibody() for _ in range(pop_size)]
    history = []

    for gen in range(generations):
        # 2. Evaluate affinity
        affinities = [affinity(ab) for ab in population]

        # 3. Select top-n_select antibodies
        ranked = sorted(
            zip(population, affinities), key=lambda x: x[1], reverse=True
        )
        selected     = [ab  for ab, _ in ranked[:n_select]]
        selected_aff = [aff for _,  aff in ranked[:n_select]]

        # 4. Clone each selected antibody (clone_factor clones each)
        clones_pool = []
        for ab in selected:
            clones_pool.extend([ab.copy() for _ in range(clone_factor)])

        # 5. Hypermutate all clones
        mutated_clones = [
            hypermutate(c, affinity(c)) for c in clones_pool
        ]

        # 6. Select the best mutated clone per original antibody
        best_clones = []
        for i in range(n_select):
            group = mutated_clones[i * clone_factor: (i + 1) * clone_factor]
            best_clones.append(max(group, key=affinity))

        # 7. Merge and keep the top (pop_size - n_replace) individuals
        combined = list(population) + best_clones
        combined.sort(key=affinity, reverse=True)
        survivors = combined[: pop_size - n_replace]

        # 8. Inject fresh random antibodies (receptor editing)
        newcomers  = [random_antibody() for _ in range(n_replace)]
        population = survivors + newcomers

        # Track best this generation
        best_ab  = combined[0]
        best_val = rastrigin(best_ab[0], best_ab[1])
        history.append((gen + 1, best_val, best_ab.copy()))

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"  Gen {gen + 1:03d}/{generations} | "
                  f"Best f(x,y) = {best_val:.6f} | "
                  f"Position = ({best_ab[0]:.4f}, {best_ab[1]:.4f})")

    # Final best
    final_affs = [affinity(ab) for ab in population]
    best_idx   = max(range(len(final_affs)), key=lambda i: final_affs[i])
    best_ab    = population[best_idx]
    best_val   = rastrigin(best_ab[0], best_ab[1])
    return best_ab, best_val, history


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_convergence(history: list, save_path: str = "clonal_selection_convergence.png"):
    """Plot and save the convergence curve.

    Args:
        history:   List of (generation, best_f, position) tuples.
        save_path: File path for the output PNG.
    """
    gens       = [h[0] for h in history]
    best_vals  = [h[1] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, best_vals, "b-o", markersize=3, linewidth=1.5, label="Best f(x,y)")
    ax.axhline(0, color="r", linestyle="--", linewidth=1, label="Global minimum (0)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best f(x, y)")
    ax.set_title("Clonal Selection Algorithm – Convergence on Rastrigin Function")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"  Convergence plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run CLONALG on the 2-D Rastrigin function and report results."""
    print("--- Clonal Selection Algorithm (CLONALG) ---")
    print("Objective   : Minimise Rastrigin function f(x, y)")
    print(f"Search space: x, y in [{BOUNDS[0]}, {BOUNDS[1]}]")
    print("Global min  : f(0, 0) = 0\n")

    best_ab, best_val, history = clonal_selection(
        pop_size=30,
        n_select=10,
        clone_factor=5,
        n_replace=5,
        generations=50,
    )

    print(f"\nOptimal solution found:")
    print(f"  x       = {best_ab[0]:.6f}")
    print(f"  y       = {best_ab[1]:.6f}")
    print(f"  f(x, y) = {best_val:.6f}  (global minimum = 0)")

    plot_convergence(history)


if __name__ == "__main__":
    main()
