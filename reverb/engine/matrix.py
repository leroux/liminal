"""Feedback matrix constructors for the FDN reverb GUI.

These functions are used by the GUI's heatmap editor for matrix visualization,
editing, and unitary projection. The actual DSP uses the Rust backend.
"""

import numpy as np


def householder(n: int) -> np.ndarray:
    return np.eye(n) - (2.0 / n) * np.ones((n, n))


def hadamard(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1.0]])
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    h_half = hadamard(n // 2)
    h = np.block([[h_half, h_half], [h_half, -h_half]])
    return h / np.sqrt(2)


def diagonal(n: int) -> np.ndarray:
    return np.eye(n)


def random_orthogonal(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    M = rng.randn(n, n)
    Q, R = np.linalg.qr(M)
    d = np.diag(R)
    Q *= np.sign(d)
    return Q


def circulant_shift(n: int) -> np.ndarray:
    P = np.zeros((n, n))
    for i in range(n):
        P[(i + 1) % n, i] = 1.0
    return P


def stautner_puckette(n: int) -> np.ndarray:
    assert n % 2 == 0, f"n must be even, got {n}"
    M = np.zeros((n, n))
    angle = np.pi / 4
    c, s = np.cos(angle), np.sin(angle)
    for i in range(0, n, 2):
        M[i, i] = c
        M[i, i + 1] = s
        M[i + 1, i] = -s
        M[i + 1, i + 1] = c
    return M


def zero(n: int) -> np.ndarray:
    return np.zeros((n, n))


MATRIX_TYPES = {
    "householder": householder,
    "hadamard": hadamard,
    "diagonal": diagonal,
    "random_orthogonal": random_orthogonal,
    "circulant": circulant_shift,
    "stautner_puckette": stautner_puckette,
}


def get_matrix(name: str, n: int, seed: int = 42) -> np.ndarray:
    if name not in MATRIX_TYPES:
        raise ValueError(f"Unknown matrix type '{name}'.")
    constructor = MATRIX_TYPES[name]
    if name == "random_orthogonal":
        return constructor(n, seed=seed)
    return constructor(n)


def nearest_unitary(matrix: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(matrix)
    return U @ Vt


def is_unitary(matrix: np.ndarray, tol: float = 1e-6) -> bool:
    product = matrix @ matrix.T
    return np.allclose(product, np.eye(len(matrix)), atol=tol)
