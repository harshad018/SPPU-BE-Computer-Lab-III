"""
Practical 6: Hybrid Genetic Algorithm – Neural Network Optimisation
============================================================
Optimises the hyperparameters of a Multi-Layer Perceptron (MLP) neural
network using a Genetic Algorithm (GA) for a spray-drying coconut milk
quality prediction task.

Problem Background:
    Spray drying converts liquid coconut milk into powder.  Process
    parameters (inlet temperature, feed flow rate, air flow rate)
    determine product quality attributes (moisture content, solubility
    index).  A GA-ANN hybrid finds the MLP architecture that predicts
    these quality attributes with minimum mean-squared error.

Synthetic Dataset:
    Generated from domain-inspired polynomial functions + Gaussian noise
    to simulate real spray-drying experimental data.

Chromosome encoding (4 genes):
    [ hidden_layer_1 (int 4-32),  hidden_layer_2 (int 0-16),
      log10(learning_rate) (float -3 to -1),  max_iter (int 100-500) ]

Fitness: 1 / (1 + MSE_validation)   (higher = better)

Usage:
    python 5_GA_NN_Optimization.py

Dependencies: numpy, scikit-learn
"""

import random

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Synthetic Dataset Generator
# ---------------------------------------------------------------------------

def generate_spray_drying_data(n_samples: int = 200, seed: int = 42):
    """Generate a synthetic spray-drying coconut milk dataset.

    Features (inputs):
        inlet_temp   : Inlet temperature (°C)     [150, 200]
        feed_flow    : Feed flow rate (mL/min)     [5,  20]
        air_flow     : Air flow rate (L/min)       [30, 50]

    Targets (outputs):
        moisture     : Moisture content (%)        [~1, 6]
        solubility   : Solubility index (%)        [~70, 95]

    Args:
        n_samples: Number of data points to generate.
        seed:      NumPy random seed for reproducibility.

    Returns:
        X: (n_samples, 3) feature matrix.
        y: (n_samples, 2) target matrix.
    """
    rng = np.random.RandomState(seed)

    inlet_temp = rng.uniform(150, 200, n_samples)
    feed_flow  = rng.uniform(5,   20,  n_samples)
    air_flow   = rng.uniform(30,  50,  n_samples)

    # Domain-inspired nonlinear relationships + noise
    moisture = (
        6.0
        - 0.02 * (inlet_temp - 150)
        + 0.08 * feed_flow
        - 0.05 * (air_flow - 30)
        + rng.normal(0, 0.2, n_samples)
    )
    moisture = np.clip(moisture, 0.5, 8.0)

    solubility = (
        95.0
        - 0.10 * (inlet_temp - 150)
        - 0.30 * feed_flow
        + 0.20 * (air_flow - 30)
        + rng.normal(0, 1.0, n_samples)
    )
    solubility = np.clip(solubility, 60.0, 99.0)

    X = np.column_stack([inlet_temp, feed_flow, air_flow])
    y = np.column_stack([moisture, solubility])
    return X, y


# ---------------------------------------------------------------------------
# GA chromosome definition
# ---------------------------------------------------------------------------

# Gene bounds: (min, max) — int genes use randint, float genes use uniform
GENE_BOUNDS = {
    "hl1":      (4,    32),
    "hl2":      (0,    16),
    "lr_log10": (-3.0, -1.0),
    "max_iter": (100,  500),
}


def random_individual() -> list:
    """Create a random chromosome (list of 4 genes)."""
    return [
        random.randint(*GENE_BOUNDS["hl1"]),
        random.randint(*GENE_BOUNDS["hl2"]),
        round(random.uniform(*GENE_BOUNDS["lr_log10"]), 3),
        random.randint(*GENE_BOUNDS["max_iter"]),
    ]


def decode(individual: list) -> dict:
    """Decode a chromosome into a dict of MLP hyperparameters.

    Args:
        individual: [hl1, hl2, lr_log10, max_iter]

    Returns:
        Dict with keys: hidden_layer_sizes, learning_rate_init, max_iter.
    """
    hl1, hl2, lr_log10, max_iter = individual
    hidden = (int(hl1),) if int(hl2) == 0 else (int(hl1), int(hl2))
    return {
        "hidden_layer_sizes": hidden,
        "learning_rate_init": 10 ** lr_log10,
        "max_iter": int(max_iter),
    }


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

def fitness(individual: list, X_train, X_val, y_train, y_val) -> float:
    """Train an MLP with the decoded hyperparameters; return 1/(1+MSE).

    Args:
        individual: Chromosome to evaluate.
        X_train, X_val: Scaled feature matrices.
        y_train, y_val: Scaled target matrices.

    Returns:
        Fitness score in (0, 1]; higher = better.
    """
    params = decode(individual)
    model = MLPRegressor(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        learning_rate_init=params["learning_rate_init"],
        max_iter=params["max_iter"],
        random_state=RANDOM_SEED,
        solver="adam",
        n_iter_no_change=10,
        early_stopping=False,
    )
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
    except Exception:
        mse = 1e6  # penalise failed training runs
    return 1.0 / (1.0 + mse)


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

def tournament_selection(population: list, fitnesses: list, k: int = 3) -> list:
    """Select one individual via k-tournament selection.

    Args:
        population: List of individuals.
        fitnesses:  Corresponding fitness scores.
        k:          Tournament size.

    Returns:
        The individual with the highest fitness among k random candidates.
    """
    candidates = random.sample(list(zip(population, fitnesses)), k)
    return max(candidates, key=lambda x: x[1])[0]


def crossover(parent1: list, parent2: list) -> tuple:
    """Single-point crossover between two parents.

    Args:
        parent1: First parent chromosome.
        parent2: Second parent chromosome.

    Returns:
        Two child chromosomes.
    """
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(individual: list, mutation_rate: float = 0.25) -> list:
    """Mutate each gene with a given probability.

    Args:
        individual:    Chromosome to mutate.
        mutation_rate: Probability of mutating each gene.

    Returns:
        Mutated chromosome.
    """
    keys = list(GENE_BOUNDS.keys())
    mutant = individual[:]
    for i, key in enumerate(keys):
        if random.random() < mutation_rate:
            lo, hi = GENE_BOUNDS[key]
            if key in ("hl1", "hl2", "max_iter"):
                mutant[i] = random.randint(int(lo), int(hi))
            else:
                mutant[i] = round(random.uniform(lo, hi), 3)
    return mutant


# ---------------------------------------------------------------------------
# GA main loop
# ---------------------------------------------------------------------------

def run_ga(X_train, X_val, y_train, y_val,
           pop_size: int = 10, generations: int = 10) -> tuple:
    """Run the Genetic Algorithm to optimise MLP hyperparameters.

    Args:
        X_train, X_val: Scaled feature matrices.
        y_train, y_val: Scaled target matrices.
        pop_size:       Population size.
        generations:    Number of GA generations.

    Returns:
        (best_individual, best_fitness, history_list)
    """
    population = [random_individual() for _ in range(pop_size)]
    history = []

    for gen in range(generations):
        fitnesses = [
            fitness(ind, X_train, X_val, y_train, y_val)
            for ind in population
        ]
        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        best_fit = fitnesses[best_idx]
        best_ind = population[best_idx]
        history.append((gen + 1, best_fit, decode(best_ind)))

        print(f"  Gen {gen + 1:02d}/{generations} | "
              f"Best fitness = {best_fit:.5f} | "
              f"Params = {decode(best_ind)}")

        # Build next generation (elitism: carry best individual unchanged)
        new_population = [best_ind]
        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            new_population.extend([mutate(c1), mutate(c2)])
        population = new_population[:pop_size]

    # Final evaluation
    fitnesses = [
        fitness(ind, X_train, X_val, y_train, y_val)
        for ind in population
    ]
    best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    return population[best_idx], fitnesses[best_idx], history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Generate dataset, optimise MLP with GA, evaluate best model on test set."""
    print("--- Hybrid GA-NN Optimisation: Spray Drying Coconut Milk ---\n")

    # 1. Generate synthetic dataset
    X, y = generate_spray_drying_data(n_samples=200, seed=RANDOM_SEED)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=RANDOM_SEED
    )

    # 2. Scale features and targets
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s   = scaler_X.transform(X_val)
    X_test_s  = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s   = scaler_y.transform(y_val)

    print(f"Dataset : {X.shape[0]} samples  |  "
          f"Train: {len(X_train_s)}  |  Val: {len(X_val_s)}  |  Test: {len(X_test_s)}\n")

    # 3. Run GA
    print("Running GA to find optimal MLP hyperparameters...\n")
    best_ind, best_fit, history = run_ga(
        X_train_s, X_val_s, y_train_s, y_val_s,
        pop_size=10, generations=10,
    )

    # 4. Train best model on combined train+val set; evaluate on test set
    best_params = decode(best_ind)
    print(f"\nBest hyperparameters found : {best_params}")

    final_X = np.vstack([X_train_s, X_val_s])
    final_y = np.vstack([y_train_s, y_val_s])

    final_model = MLPRegressor(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        learning_rate_init=best_params["learning_rate_init"],
        max_iter=best_params["max_iter"],
        random_state=RANDOM_SEED,
        solver="adam",
    )
    final_model.fit(final_X, final_y)

    y_pred_s = final_model.predict(X_test_s)
    y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 2))

    mse  = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print("\n--- Test Set Evaluation ---")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")

    print("\n  Sample predictions (first 5 test rows):")
    header = f"  {'Act. Moisture':>14} {'Pred Moisture':>14} " \
             f"{'Act. Solubility':>16} {'Pred Solubility':>16}"
    print(header)
    print("  " + "-" * 62)
    for i in range(min(5, len(y_test))):
        print(f"  {y_test[i, 0]:>14.3f} {y_pred[i, 0]:>14.3f} "
              f"{y_test[i, 1]:>16.3f} {y_pred[i, 1]:>16.3f}")


if __name__ == "__main__":
    main()
