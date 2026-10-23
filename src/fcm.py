"""
Fuzzy C-Means (FCM) clustering — implemented directly from the equations
given in Section II of the reference paper:

    Vijayalakshmi & Muruganand, "Smoke detection in video images using
    background subtraction method for early fire alarm system", ICCES 2017.

    Jm(U,V) = sum_i sum_j (u_ij)^m * d^2(x_j, v_i)                (Eq. 3)
    u_ij    = [ sum_k ( d(x_j,v_i) / d(x_j,v_k) )^(2/(m-1)) ]^-1  (Eq. 4)
    v_i     = sum_j (u_ij^m * x_j) / sum_j (u_ij^m)               (Eq. 5)
    stop when max_i || v_i^(t) - v_i^(t-1) || < epsilon           (Eq. 6)

This is a small, dependency-free implementation (numpy only) so it is easy
to inspect/mark against the paper's formulas.
"""
from __future__ import annotations
import numpy as np


class FuzzyCMeans:
    """Standard Fuzzy C-Means clustering.

    Parameters
    ----------
    c : int
        Number of clusters (paper uses c=2: smoke / non-smoke).
    m : float
        Fuzziness coefficient, m >= 1. Paper uses m = 2.
    epsilon : float
        Termination threshold on centroid movement. Paper uses 0.001.
    max_iter : int
        Safety cap on iterations.
    random_state : int
        Seed for reproducible centroid initialisation.
    """

    def __init__(self, c: int = 2, m: float = 2.0, epsilon: float = 1e-3,
                 max_iter: int = 100, random_state: int = 42):
        if c < 2:
            raise ValueError("c must be >= 2")
        if m <= 1:
            raise ValueError("m must be > 1")
        self.c = c
        self.m = m
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers_: np.ndarray | None = None
        self.u_: np.ndarray | None = None
        self.n_iter_: int = 0

    @staticmethod
    def _pairwise_dist(X: np.ndarray, V: np.ndarray) -> np.ndarray:
        # X: (n, f), V: (c, f) -> (n, c) Euclidean distance d(x_j, v_i)
        diff = X[:, None, :] - V[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=2))
        return np.maximum(d, 1e-12)  # avoid div-by-zero

    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        """X: (n_samples, n_features) feature matrix (each row = one candidate blob)."""
        X = np.asarray(X, dtype=np.float64)
        n, f = X.shape
        rng = np.random.default_rng(self.random_state)

        # init membership matrix U randomly, rows sum to 1 (constraint sum_i u_ik = 1)
        U = rng.random((n, self.c))
        U = U / U.sum(axis=1, keepdims=True)

        V = np.zeros((self.c, f))
        power = 2.0 / (self.m - 1.0)

        for it in range(1, self.max_iter + 1):
            Um = U ** self.m
            # Eq. 5: update centroids
            V_new = (Um.T @ X) / np.maximum(Um.sum(axis=0)[:, None], 1e-12)

            # Eq. 4: update membership using distances to the NEW centroids
            d = self._pairwise_dist(X, V_new)  # (n, c)
            ratio = d[:, :, None] / d[:, None, :]  # d(x_j,v_i)/d(x_j,v_k)  -> (n,c,c)
            U_new = 1.0 / np.sum(ratio ** power, axis=2)

            # Eq. 6: stopping criterion
            shift = np.max(np.linalg.norm(V_new - V, axis=1)) if it > 1 else np.inf
            V, U = V_new, U_new
            self.n_iter_ = it
            if shift < self.epsilon:
                break

        self.centers_ = V
        self.u_ = U
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard label = cluster of maximum membership (as described in the paper)."""
        d = self._pairwise_dist(np.asarray(X, dtype=np.float64), self.centers_)
        power = 2.0 / (self.m - 1.0)
        ratio = d[:, :, None] / d[:, None, :]
        U = 1.0 / np.sum(ratio ** power, axis=2)
        return np.argmax(U, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return np.argmax(self.u_, axis=1)
