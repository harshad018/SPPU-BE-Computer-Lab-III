# Assignment 5 – Ant Colony Optimization (ACO) for TSP

## Overview
Implements the **Ant Colony Optimization** meta-heuristic to solve the
**Travelling Salesman Problem (TSP)**:
> A salesman must visit each city exactly once and return to the starting city
> via the shortest possible route.

ACO simulates a colony of virtual ants that deposit pheromone on paths.
Shorter paths accumulate more pheromone and are more likely to be chosen by
future ants, converging on a near-optimal tour.

## Algorithm Summary
```
Initialise pheromone matrix (all 1s)
For each iteration:
    For each ant:
        Construct a tour probabilistically (pheromone × heuristic)
        Record tour length
    Update pheromone: evaporate by ρ, deposit Q/length on each used edge
Return best tour found
```

## Requirements
```
numpy
matplotlib   (optional – only needed for --plot)
```
Install with:
```bash
pip install numpy matplotlib
```

## Usage
```bash
# Default: 20 random cities, 20 ants, 100 iterations
python aco_tsp.py

# 30 cities, 50 ants, custom parameters
python aco_tsp.py --cities 30 --ants 50 --iterations 200

# Reproducible run with tour plot saved as PNG
python aco_tsp.py --seed 7 --plot

# All options
python aco_tsp.py --help
```

## Parameters
| Argument | Default | Description |
|---|---|---|
| `--cities` | 20 | Number of cities (random coordinates) |
| `--ants` | 20 | Ants per iteration |
| `--iterations` | 100 | Number of ACO iterations |
| `--alpha` | 1.0 | Pheromone weight (α) |
| `--beta` | 5.0 | Heuristic (1/dist) weight (β) |
| `--rho` | 0.5 | Pheromone evaporation rate (ρ) |
| `--Q` | 100.0 | Pheromone deposit constant |
| `--seed` | 42 | Random seed for reproducibility |
| `--plot` | off | Save & display tour plot as PNG |

## Expected Output
```
============================================================
Ant Colony Optimisation – Travelling Salesman Problem
============================================================
  Cities     : 20
  Ants       : 20
  Iterations : 100
  α=1.0  β=5.0  ρ=0.5  Q=100.0

[Running ACO ...]

  Iteration    1/100  Best tour length: 478.32
  Iteration   10/100  Best tour length: 312.45
  ...
  Iteration  100/100  Best tour length: 267.18

============================================================
Results
============================================================
  Best tour length : 267.1800
  Best tour order  : [4, 11, 2, ...]
  Improvement      : 478.32 → 267.18

Done.
```
