"""Reduced-order models: compress the field so the twin runs in real time."""

import numpy as np
from scipy.linalg import qr


class POD:
    """Proper orthogonal decomposition of snapshots X (n_snapshots, n_space)."""

    def fit(self, X, r=None, center=True):
        self.mean = X.mean(0) if center else np.zeros(X.shape[1])
        Uh, s, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.singular_values = s
        r = r or s.size
        self.modes = Vt[:r].T  # (n_space, r), orthonormal columns
        return self

    @property
    def energy(self):
        e = self.singular_values**2
        return np.cumsum(e) / e.sum()

    def rank_for(self, fraction):
        return int(np.searchsorted(self.energy, fraction) + 1)

    def encode(self, X, r=None):
        Phi = self.modes[:, :r]
        return (X - self.mean) @ Phi

    def decode(self, a):
        r = a.shape[-1]
        return self.mean + a @ self.modes[:, :r].T


class DMD:
    """Exact dynamic mode decomposition: x_{k+1} ~ A x_k with A of rank r."""

    def fit(self, X, r=10, dt=1.0):
        X0, X1 = X[:-1].T, X[1:].T  # (n, m)
        U, s, Vh = np.linalg.svd(X0, full_matrices=False)
        U, s, V = U[:, :r], s[:r], Vh[:r].conj().T
        Atilde = U.conj().T @ X1 @ V / s
        lam, W = np.linalg.eig(Atilde)
        self.eigs = lam
        self.omega = np.log(lam.astype(complex)) / dt  # continuous-time growth/frequency
        self.modes = X1 @ V / s @ W
        self.b = np.linalg.lstsq(self.modes, X[0].astype(complex), rcond=None)[0]
        self.dt = dt
        return self

    def predict(self, n_steps, x0=None):
        b = self.b if x0 is None else np.linalg.lstsq(self.modes, x0.astype(complex), rcond=None)[0]
        k = np.arange(n_steps)
        dyn = b[:, None] * self.eigs[:, None] ** k[None, :]
        return (self.modes @ dyn).real.T  # (n_steps, n)


def qr_sensors(modes, p):
    """Greedy near-optimal sensor locations for reconstructing the span of `modes` (QR pivoting)."""
    _, _, piv = qr(modes.T, pivoting=True, mode="economic")
    return piv[:p]


def gappy_reconstruct(y, idx, modes, mean):
    """Least-squares fit of POD coefficients to point measurements y at flat indices idx."""
    a, *_ = np.linalg.lstsq(modes[idx], y - mean[idx], rcond=None)
    return mean + modes @ a
