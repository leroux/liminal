"""Feedback matrix — controls how energy flows between FDN delay lines.

All matrices are unitary (energy-preserving) unless noted otherwise.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Matrix constructors
# ---------------------------------------------------------------------------

def householder(n: int) -> np.ndarray:
    """Householder reflection: A = I - (2/N) * ones * ones^T

    Every node feeds into every other node equally (uniform coupling).
    The standard choice for FDN reverbs since Jot 1991.
    """
    return np.eye(n) - (2.0 / n) * np.ones((n, n))


def hadamard(n: int) -> np.ndarray:
    """Hadamard matrix (normalized to be unitary).

    Requires n to be a power of 2. More structured coupling than Householder —
    some node pairs add, others subtract. Can sound slightly different in the
    high-frequency decay pattern.
    """
    # Build recursively: H_1 = [1], H_2k = [[H_k, H_k], [H_k, -H_k]]
    if n == 1:
        return np.array([[1.0]])
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    h_half = hadamard(n // 2)
    h = np.block([[h_half, h_half], [h_half, -h_half]])
    return h / np.sqrt(2)  # normalize so top-level is unitary


def diagonal(n: int) -> np.ndarray:
    """Identity matrix — no coupling between delay lines.

    Each delay line is independent. Sounds metallic/comb-filter-like.
    """
    return np.eye(n)


def random_orthogonal(n: int, seed: int = 42) -> np.ndarray:
    """Random orthogonal matrix via QR decomposition.

    Every run with the same seed gives the same matrix. Unitary.
    Non-uniform coupling — some paths stronger than others.
    """
    rng = np.random.RandomState(seed)
    M = rng.randn(n, n)
    Q, R = np.linalg.qr(M)
    # Ensure determinant is +1 (proper rotation, not reflection)
    d = np.diag(R)
    Q *= np.sign(d)
    return Q


def circulant_shift(n: int) -> np.ndarray:
    """Circulant permutation — each node feeds into the next.

    Node 0 -> Node 1 -> Node 2 -> ... -> Node 0. Energy travels in a ring.
    Unitary (permutation matrices are orthogonal).
    """
    P = np.zeros((n, n))
    for i in range(n):
        P[(i + 1) % n, i] = 1.0
    return P


def stautner_puckette(n: int) -> np.ndarray:
    """Stautner-Puckette style matrix — pairs of nodes cross-coupled.

    Classic 4-channel reverb topology (1982). For n=8, creates 4 cross-coupled
    pairs. Each pair has a rotation angle of pi/4.
    """
    assert n % 2 == 0, f"n must be even, got {n}"
    M = np.zeros((n, n))
    angle = np.pi / 4  # 45 degrees
    c, s = np.cos(angle), np.sin(angle)
    for i in range(0, n, 2):
        M[i, i] = c
        M[i, i + 1] = s
        M[i + 1, i] = -s
        M[i + 1, i + 1] = c
    return M


# ---------------------------------------------------------------------------
# Fast apply functions (avoid full matrix multiply when possible)
# ---------------------------------------------------------------------------

def apply_householder(x: np.ndarray) -> np.ndarray:
    """O(N) Householder apply — no matrix multiply needed."""
    s = np.sum(x) * (2.0 / len(x))
    return x - s


def apply_hadamard(x: np.ndarray) -> np.ndarray:
    """O(N log N) Hadamard apply via recursive butterfly."""
    n = len(x)
    if n == 1:
        return x.copy()
    half = n // 2
    top = x[:half] + x[half:]
    bot = x[:half] - x[half:]
    inv_sqrt2 = 1.0 / np.sqrt(2)
    return np.concatenate([apply_hadamard(top), apply_hadamard(bot)]) * inv_sqrt2


def apply_matrix(matrix: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Generic matrix-vector multiply. Fallback for any matrix."""
    return matrix @ x


# ---------------------------------------------------------------------------
# Registry: name -> (constructor, fast_apply_or_None)
# ---------------------------------------------------------------------------

MATRIX_TYPES = {
    "householder": (householder, apply_householder),
    "hadamard": (hadamard, apply_hadamard),
    "diagonal": (diagonal, None),
    "random_orthogonal": (random_orthogonal, None),
    "circulant": (circulant_shift, None),
    "stautner_puckette": (stautner_puckette, None),
}


def build_matrix_apply(name: str, n: int, seed: int = 42):
    """Build a matrix and return a fast apply function.

    Returns:
        apply_fn: callable that takes an n-vector and returns an n-vector
    """
    if name not in MATRIX_TYPES:
        raise ValueError(f"Unknown matrix type '{name}'. Options: {list(MATRIX_TYPES.keys())}")

    constructor, fast_fn = MATRIX_TYPES[name]

    if fast_fn is not None:
        return fast_fn

    # Build the matrix and close over it
    if name == "random_orthogonal":
        mat = constructor(n, seed=seed)
    else:
        mat = constructor(n)
    return lambda x: apply_matrix(mat, x)
