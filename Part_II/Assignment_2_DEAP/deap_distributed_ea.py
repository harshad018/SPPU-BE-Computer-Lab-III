"""
Assignment 2 – Distributed Evolutionary Algorithms with DEAP
=============================================================
Demonstrates DEAP by solving a simple real-valued optimisation problem
(maximise  f(x) = -sum(x_i^2)  i.e., find the global maximum at origin)
using a **Genetic Algorithm** and **distributed evaluation** via Python's
`multiprocessing.Pool`.

Usage
-----
    python deap_distributed_ea.py              # single-process (baseline)
    python deap_distributed_ea.py --workers 4  # distributed with 4 workers
    python deap_distributed_ea.py --help       # show all options

Requirements
------------
    pip install deap
"""

import argparse
import multiprocessing
import random
import time

import numpy as np

# DEAP imports
from deap import algorithms, base, creator, tools


# ---------------------------------------------------------------------------
# Problem definition  (maximise sum(-x_i^2), i.e. global optimum at origin)
# ---------------------------------------------------------------------------

NDIM = 10   # number of dimensions


def evaluate(individual):
    """Fitness function: f(x) = -sum(x_i^2).  Maximum is 0 at origin."""
    return (-sum(x ** 2 for x in individual),)


# ---------------------------------------------------------------------------
# DEAP setup
# ---------------------------------------------------------------------------

def build_toolbox(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    # Create fitness and individual types (only once, guard against re-creation)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute: random float in [-5, 5]
    toolbox.register("attr_float", random.uniform, -5.0, 5.0)

    # Individual and population
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_float, NDIM)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


# ---------------------------------------------------------------------------
# Evolution runner
# ---------------------------------------------------------------------------

def run_evolution(toolbox, pop_size: int = 100, generations: int = 50,
                  cx_prob: float = 0.7, mut_prob: float = 0.2,
                  seed: int = 42, workers: int = 1, verbose: bool = True):
    """Run the genetic algorithm and return (best_individual, logbook)."""
    random.seed(seed)
    np.random.seed(seed)

    # Register distributed map if workers > 1
    if workers > 1:
        pool = multiprocessing.Pool(processes=workers)
        toolbox.register("map", pool.map)
        if verbose:
            print(f"  Distributed mode: {workers} worker processes.")
    else:
        if verbose:
            print("  Single-process mode.")

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)
    pop = toolbox.population(n=pop_size)

    t0 = time.perf_counter()
    pop, log = algorithms.eaSimple(
        pop, toolbox,
        cxpb=cx_prob, mutpb=mut_prob, ngen=generations,
        stats=stats, halloffame=hof,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - t0

    if workers > 1:
        pool.close()
        pool.join()

    return hof[0], log, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DEAP Distributed EA – maximise f(x) = -sum(x_i^2)."
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--pop_size", type=int, default=100,
                        help="Population size (default: 100)")
    parser.add_argument("--generations", type=int, default=50,
                        help="Number of generations (default: 50)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes for distributed eval (default: 1 = serial)")
    parser.add_argument("--cx_prob", type=float, default=0.7,
                        help="Crossover probability (default: 0.7)")
    parser.add_argument("--mut_prob", type=float, default=0.2,
                        help="Mutation probability (default: 0.2)")
    args = parser.parse_args()

    print("=" * 60)
    print("DEAP – Distributed Evolutionary Algorithm Demo")
    print("=" * 60)
    print(f"  Problem : maximise f(x) = -sum(x_i^2), {NDIM}-D")
    print(f"  Optimum : f=0 at x=[0,...,0]")
    print(f"  Seed    : {args.seed}")
    print(f"  Workers : {args.workers}")
    print()

    toolbox = build_toolbox(seed=args.seed)
    best, log, elapsed = run_evolution(
        toolbox,
        pop_size=args.pop_size,
        generations=args.generations,
        cx_prob=args.cx_prob,
        mut_prob=args.mut_prob,
        seed=args.seed,
        workers=args.workers,
    )

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Best fitness : {best.fitness.values[0]:.6f}  (optimum = 0.0)")
    print(f"  Best individual (x): {[round(v, 4) for v in best]}")
    print(f"  Wall time    : {elapsed:.2f}s")

    # Show last 5 generations from logbook
    print("\nLast 5 generations:")
    print(f"{'gen':>4}  {'avg':>10}  {'max':>10}  {'std':>10}")
    for rec in log[-5:]:
        print(f"{rec['gen']:>4}  {rec['avg']:>10.4f}  {rec['max']:>10.4f}  {rec['std']:>10.4f}")
    print()


if __name__ == "__main__":
    main()
