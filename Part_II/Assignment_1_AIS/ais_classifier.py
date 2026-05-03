"""
Assignment 1 – Artificial Immune System (AIS) for Structure Damage Classification
==================================================================================
Implements a Clonal Selection / Negative Selection inspired AIS classifier and
applies it to a synthetic structural-sensor dataset with four damage classes.

Usage
-----
    python ais_classifier.py            # run with default settings
    python ais_classifier.py --seed 99  # reproducible run with custom seed
    python ais_classifier.py --help     # show all options

Requirements
------------
    pip install numpy scikit-learn
"""

import argparse
import random
import math
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def generate_dataset(n_samples: int = 400, n_features: int = 8, seed: int = 42):
    """Generate a synthetic structural-sensor dataset.

    Classes
    -------
    0 – Healthy
    1 – Minor damage (small cracks)
    2 – Moderate damage (large cracks)
    3 – Severe damage (structural failure risk)

    Each class is drawn from a Gaussian distribution with a distinct mean
    vector and shared covariance.
    """
    rng = np.random.RandomState(seed)
    n_classes = 4
    samples_per_class = n_samples // n_classes
    means = [
        np.zeros(n_features),                          # healthy
        np.array([1.5] * n_features),                  # minor
        np.array([3.0, 3.5, 2.5, 3.0, 3.2, 2.8, 3.1, 2.9]),  # moderate
        np.array([5.0] * n_features),                  # severe
    ]
    cov = np.eye(n_features) * 0.5
    X_parts, y_parts = [], []
    for cls, mean in enumerate(means):
        X_parts.append(rng.multivariate_normal(mean, cov, samples_per_class))
        y_parts.append(np.full(samples_per_class, cls, dtype=int))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    # shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# AIS Classifier (Clonal Selection + Affinity-based)
# ---------------------------------------------------------------------------

class AISClassifier:
    """Simple Clonal-Selection-inspired Artificial Immune System classifier.

    Each training sample becomes a candidate antibody.  The system clones
    high-affinity antibodies and applies hypermutation inversely proportional
    to affinity.  At inference time the class is determined by the nearest
    antibody (1-NN style using Euclidean affinity).

    Parameters
    ----------
    n_clones      : clones per antibody per generation
    mutation_rate : base mutation rate (scaled by 1/affinity)
    generations   : number of clonal selection iterations
    seed          : random seed for reproducibility
    """

    def __init__(self, n_clones: int = 5, mutation_rate: float = 0.1,
                 generations: int = 10, seed: int = 42):
        self.n_clones = n_clones
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.seed = seed
        self.antibodies_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _affinity(a: np.ndarray, b: np.ndarray) -> float:
        """Affinity = 1 / (1 + Euclidean distance)."""
        return 1.0 / (1.0 + np.linalg.norm(a - b))

    def _mutate(self, antibody: np.ndarray, affinity: float,
                rng: np.random.RandomState) -> np.ndarray:
        """Hypermutation: mutation amplitude inversely proportional to affinity."""
        amplitude = self.mutation_rate / (affinity + 1e-9)
        noise = rng.randn(*antibody.shape) * amplitude
        return antibody + noise

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.seed)
        # Initialise antibody memory = training samples
        antibodies = X.copy()
        labels = y.copy()

        for gen in range(self.generations):
            new_abs, new_lbls = [], []
            for i, ab in enumerate(antibodies):
                lbl = labels[i]
                # Compute affinity to all same-class antibodies
                same_class = antibodies[labels == lbl]
                if len(same_class) == 0:
                    continue
                affinities = np.array([self._affinity(ab, s) for s in same_class])
                avg_aff = float(np.mean(affinities))
                # Clone and mutate
                for _ in range(self.n_clones):
                    clone = self._mutate(ab, avg_aff, rng)
                    new_abs.append(clone)
                    new_lbls.append(lbl)

            if new_abs:
                # Merge originals + clones, then prune duplicates loosely
                antibodies = np.vstack([antibodies, np.array(new_abs)])
                labels = np.concatenate([labels, np.array(new_lbls)])

        self.antibodies_ = antibodies
        self.labels_ = labels
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.antibodies_ is None:
            raise RuntimeError("Call fit() first.")
        predictions = []
        for x in X:
            affinities = np.array([self._affinity(x, ab)
                                   for ab in self.antibodies_])
            best_idx = int(np.argmax(affinities))
            predictions.append(self.labels_[best_idx])
        return np.array(predictions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AIS classifier for structure damage classification."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n_samples", type=int, default=400,
                        help="Total synthetic samples (default: 400)")
    parser.add_argument("--n_features", type=int, default=8,
                        help="Number of sensor features (default: 8)")
    parser.add_argument("--n_clones", type=int, default=5,
                        help="Clones per antibody per generation (default: 5)")
    parser.add_argument("--generations", type=int, default=5,
                        help="Clonal selection generations (default: 5)")
    parser.add_argument("--mutation_rate", type=float, default=0.2,
                        help="Base mutation rate (default: 0.2)")
    parser.add_argument("--test_size", type=float, default=0.25,
                        help="Fraction for test set (default: 0.25)")
    args = parser.parse_args()

    print("=" * 60)
    print("Artificial Immune System – Structure Damage Classification")
    print("=" * 60)

    # 1. Dataset
    print(f"\n[1] Generating synthetic dataset ...")
    X, y = generate_dataset(args.n_samples, args.n_features, seed=args.seed)
    class_names = ["Healthy", "Minor Damage", "Moderate Damage", "Severe Damage"]
    print(f"    Samples: {len(X)}, Features: {X.shape[1]}, Classes: {len(set(y))}")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

    # 3. Normalise
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Train AIS
    print(f"\n[2] Training AIS classifier ...")
    clf = AISClassifier(
        n_clones=args.n_clones,
        mutation_rate=args.mutation_rate,
        generations=args.generations,
        seed=args.seed,
    )
    clf.fit(X_train, y_train)
    print(f"    Antibody memory size after training: {len(clf.antibodies_)}")

    # 5. Evaluate
    print(f"\n[3] Evaluating on test set ...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n    Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Done.")


if __name__ == "__main__":
    main()
