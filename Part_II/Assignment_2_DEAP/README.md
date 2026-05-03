# Assignment 2 – Distributed Evolutionary Algorithms with DEAP

## Overview
Uses the **DEAP** (Distributed Evolutionary Algorithms in Python) framework to
run a Genetic Algorithm that maximises `f(x) = -∑xᵢ²` (global optimum at origin).
Distributed evaluation is achieved by registering Python's `multiprocessing.Pool.map`
as DEAP's evaluation map, so fitness evaluations run in parallel worker processes.

## Requirements
```
deap
numpy
```
Install with:
```bash
pip install deap numpy
```

## Usage
```bash
# Serial (baseline)
python deap_distributed_ea.py

# Distributed with 4 parallel workers
python deap_distributed_ea.py --workers 4

# Custom settings
python deap_distributed_ea.py --seed 7 --pop_size 200 --generations 100

# All options
python deap_distributed_ea.py --help
```

## How Distributed Execution Works
```
Main process
  │
  ├─ builds toolbox, registers pool.map as the evaluation map
  │
  └─ DEAP eaSimple loop
        for each generation:
          │
          └─ toolbox.map(evaluate, population)   ← parallel via Pool
                 ├─ Worker 0: evaluate(ind[0..24])
                 ├─ Worker 1: evaluate(ind[25..49])
                 ├─ Worker 2: evaluate(ind[50..74])
                 └─ Worker 3: evaluate(ind[75..99])
```

## Parameters
| Argument | Default | Description |
|---|---|---|
| `--seed` | 42 | Random seed |
| `--pop_size` | 100 | Population size |
| `--generations` | 50 | Generations to evolve |
| `--workers` | 1 | Worker processes (1 = serial) |
| `--cx_prob` | 0.7 | Crossover probability |
| `--mut_prob` | 0.2 | Mutation probability |

## Expected Output
```
============================================================
DEAP – Distributed Evolutionary Algorithm Demo
============================================================
  Problem : maximise f(x) = -sum(x_i^2), 10-D
  Optimum : f=0 at x=[0,...,0]
  ...

Last 5 generations:
 gen         avg         max         std
  46     -0.0312      -0.001      0.042
  47     -0.0287      -0.001      0.039
  ...
```
