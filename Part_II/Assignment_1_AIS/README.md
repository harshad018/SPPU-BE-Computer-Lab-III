# Assignment 1 – Artificial Immune System (AIS) Structure Damage Classification

## Overview
Implements a **Clonal Selection**-inspired Artificial Immune System (AIS) classifier
applied to a synthetic structural-sensor dataset with four damage classes:
`Healthy`, `Minor Damage`, `Moderate Damage`, `Severe Damage`.

## How it Works
1. Each training sample becomes an **antibody** in the memory pool.
2. Each generation, antibodies are **cloned** (proportional to their class affinity)
   and subjected to **hypermutation** (mutation amplitude is inversely proportional
   to affinity — high affinity → low mutation).
3. At inference, prediction is made by finding the **highest-affinity antibody**
   (nearest-neighbour in feature space).

## Requirements
```
numpy
scikit-learn
```
Install with:
```bash
pip install numpy scikit-learn
```

## Usage
```bash
# Default run
python ais_classifier.py

# Custom seed for reproducibility
python ais_classifier.py --seed 99

# All options
python ais_classifier.py --help
```

## Expected Output
```
============================================================
Artificial Immune System – Structure Damage Classification
============================================================

[1] Generating synthetic dataset ...
    Samples: 400, Features: 8, Classes: 4
    Train: 300, Test: 100

[2] Training AIS classifier ...
    Antibody memory size after training: ...

[3] Evaluating on test set ...
    Accuracy: ~90%+ (varies by seed / hyperparameters)

              precision    recall  f1-score   support
     Healthy     ...
Minor Damage     ...
     ...
```

## Parameters
| Argument | Default | Description |
|---|---|---|
| `--seed` | 42 | Random seed |
| `--n_samples` | 400 | Total synthetic samples |
| `--n_features` | 8 | Sensor feature count |
| `--n_clones` | 5 | Clones per antibody per generation |
| `--generations` | 5 | Clonal selection generations |
| `--mutation_rate` | 0.2 | Base hypermutation rate |
| `--test_size` | 0.25 | Test fraction |
