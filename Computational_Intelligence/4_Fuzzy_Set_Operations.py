"""
Practical 4: Fuzzy Set Operations
============================================================
Implements the following operations on fuzzy sets and fuzzy relations:

  Fuzzy Set Operations:
    • Union        : mu_{A union B}(x) = max(mu_A(x), mu_B(x))
    • Intersection : mu_{A intersect B}(x) = min(mu_A(x), mu_B(x))
    • Complement   : mu_{A'}(x)   = 1 - mu_A(x)
    • Difference   : mu_{A-B}(x) = min(mu_A(x), 1 - mu_B(x))

  Fuzzy Relation Operations:
    • Max-Min Composition (R o S):
        result[i][j] = max_k( min(R[i][k], S[k][j]) )

All membership values must lie in the interval [0, 1].

Usage:
    python 4_Fuzzy_Set_Operations.py

Dependencies: numpy
"""

import numpy as np


# ---------------------------------------------------------------------------
# Fuzzy Set Operations
# ---------------------------------------------------------------------------

def fuzzy_union(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the fuzzy union of sets A and B.

    Args:
        A: 1-D array of membership values in [0, 1].
        B: 1-D array of membership values in [0, 1] (same size as A).

    Returns:
        Element-wise maximum: max(A, B).
    """
    return np.maximum(A, B)


def fuzzy_intersection(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the fuzzy intersection of sets A and B.

    Args:
        A: 1-D array of membership values in [0, 1].
        B: 1-D array of membership values in [0, 1] (same size as A).

    Returns:
        Element-wise minimum: min(A, B).
    """
    return np.minimum(A, B)


def fuzzy_complement(A: np.ndarray) -> np.ndarray:
    """Compute the fuzzy complement (negation) of set A.

    Args:
        A: 1-D array of membership values in [0, 1].

    Returns:
        1 - A element-wise.
    """
    return 1.0 - A


def fuzzy_difference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the fuzzy difference A \\ B = A intersect complement(B).

    Args:
        A: 1-D array of membership values in [0, 1].
        B: 1-D array of membership values in [0, 1] (same size as A).

    Returns:
        min(A, 1 - B) element-wise.
    """
    return np.minimum(A, 1.0 - B)


# ---------------------------------------------------------------------------
# Fuzzy Relation (Matrix) Operations
# ---------------------------------------------------------------------------

def max_min_composition(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Compute the max-min composition T = R o S of two fuzzy relations.

    For each element T[i, j]:
        T[i, j] = max over k of  min(R[i, k], S[k, j])

    Args:
        R: m x n fuzzy relation matrix (values in [0, 1]).
        S: n x p fuzzy relation matrix (values in [0, 1]).

    Returns:
        T: m x p result matrix.

    Raises:
        ValueError: If inner dimensions do not match.
    """
    m, n = R.shape
    n2, p = S.shape
    if n != n2:
        raise ValueError(
            f"Inner dimensions must match: R is {m}x{n}, S is {n2}x{p}."
        )
    # Vectorised: broadcast R[:,k,np.newaxis] and S[np.newaxis,k,:] over k
    # R[:, :, np.newaxis] shape: (m, n, 1)
    # S[np.newaxis, :, :] shape: (1, n, p)
    # min across axis-1 (the k dimension) then max
    T = np.max(np.minimum(R[:, :, np.newaxis], S[np.newaxis, :, :]), axis=1)
    return T


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


def _show(label: str, arr: np.ndarray, precision: int = 2):
    np.set_printoptions(precision=precision, suppress=True)
    print(f"  {label}:")
    if arr.ndim == 1:
        print(f"    {arr}")
    else:
        for row in arr:
            print(f"    {row}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Demonstrate all fuzzy set and fuzzy relation operations."""

    # -----------------------------------------------------------------------
    # 1. Fuzzy Set Operations
    # -----------------------------------------------------------------------
    _section("Fuzzy Set Operations")

    # Universe of discourse X = {x1, x2, x3, x4, x5}
    A = np.array([0.2, 0.5, 0.8, 1.0, 0.6])
    B = np.array([0.4, 0.3, 0.9, 0.7, 0.1])

    _show("Set A", A)
    _show("Set B", B)
    print()
    _show("A union B        (Union,       max(A, B))",      fuzzy_union(A, B))
    _show("A intersect B    (Intersection, min(A, B))",      fuzzy_intersection(A, B))
    _show("A'               (Complement A, 1 - A)",          fuzzy_complement(A))
    _show("B'               (Complement B, 1 - B)",          fuzzy_complement(B))
    _show("A - B            (Difference,  min(A, 1-B))",     fuzzy_difference(A, B))
    _show("B - A            (Difference,  min(B, 1-A))",     fuzzy_difference(B, A))

    # -----------------------------------------------------------------------
    # 2. Max-Min Composition – square relations
    # -----------------------------------------------------------------------
    _section("Max-Min Composition of Fuzzy Relations (3x3)")

    # R: relation from X to Y  (3x3)
    R = np.array([
        [0.2, 0.5, 0.8],
        [0.6, 0.3, 0.1],
        [0.9, 0.7, 0.4],
    ])

    # S: relation from Y to Z  (3x3)
    S = np.array([
        [0.1, 0.4, 0.7],
        [0.5, 0.8, 0.2],
        [0.3, 0.6, 0.9],
    ])

    _show("Relation R (X -> Y)", R)
    print()
    _show("Relation S (Y -> Z)", S)

    T = max_min_composition(R, S)
    print()
    _show("Composition T = R o S  (X -> Z)", T)

    # Manual verification of T[0,0]:
    # max( min(0.2,0.1), min(0.5,0.5), min(0.8,0.3) ) = max(0.1, 0.5, 0.3) = 0.5
    expected_00 = 0.5
    check = "OK" if abs(T[0, 0] - expected_00) < 1e-9 else "FAILED"
    print(f"\n  Verification T[0,0]: max(min(0.2,0.1), min(0.5,0.5), min(0.8,0.3))")
    print(f"                     = max(0.1, 0.5, 0.3) = {expected_00}  [{check}]")

    # -----------------------------------------------------------------------
    # 3. Max-Min Composition – non-square relations
    # -----------------------------------------------------------------------
    _section("Max-Min Composition – Non-Square Relations (2x3 o 3x4)")

    R2 = np.array([
        [0.3, 0.7, 0.5],
        [0.8, 0.2, 0.6],
    ])

    S2 = np.array([
        [0.4, 0.1, 0.6, 0.9],
        [0.7, 0.5, 0.3, 0.2],
        [0.2, 0.8, 0.4, 0.6],
    ])

    _show("R2 (2x3)", R2)
    print()
    _show("S2 (3x4)", S2)

    T2 = max_min_composition(R2, S2)
    print()
    _show("T2 = R2 o S2  (2x4)", T2)


if __name__ == "__main__":
    main()
