"""Virtual sensors: where we measure, how noisy, and the observation operator H."""

import numpy as np


class PointSensors:
    """Point sensors reading entries `idx` of a flattened state vector.

    y = H x + v,  v ~ N(0, noise_std^2 I)
    """

    def __init__(self, idx, n_state, noise_std=0.0):
        self.idx = np.asarray(idx, int)
        self.n_state = n_state
        self.noise_std = noise_std

    @property
    def m(self):
        return self.idx.size

    @property
    def H(self):
        H = np.zeros((self.m, self.n_state))
        H[np.arange(self.m), self.idx] = 1.0
        return H

    @property
    def R(self):
        return self.noise_std**2 * np.eye(self.m)

    def observe(self, state, rng=None):
        """state: (..., n_state) or (..., ny, nx). Returns (..., m)."""
        state = np.asarray(state)
        if state.shape[-1] != self.n_state:  # (..., ny, nx) field -> (..., ny*nx)
            state = state.reshape(*state.shape[:-2], -1)
        y = state[..., self.idx]
        if rng is not None and self.noise_std > 0:
            y = y + rng.normal(0.0, self.noise_std, y.shape)
        return y


def evenly_spaced_1d(nx, m, margin=0.08):
    """m sensor indices spread over a 1D grid of nx nodes, away from the ends."""
    lo, hi = int(margin * (nx - 1)), int((1 - margin) * (nx - 1))
    return np.unique(np.linspace(lo, hi, m).round().astype(int))


def grid_2d(n, k, margin=0.1):
    """k x k grid of sensors on an n x n field. Returns flat indices and (iy, ix)."""
    pos = np.linspace(margin * (n - 1), (1 - margin) * (n - 1), k).round().astype(int)
    iy, ix = np.meshgrid(pos, pos, indexing="ij")
    iy, ix = iy.ravel(), ix.ravel()
    return iy * n + ix, (iy, ix)


def random_2d(n, m, rng, margin=0.05):
    lo, hi = int(margin * n), int((1 - margin) * n)
    iy = rng.integers(lo, hi, m)
    ix = rng.integers(lo, hi, m)
    return iy * n + ix, (iy, ix)


def flat_to_xy(idx, n, x):
    """Flat indices on an n x n grid -> physical coordinates using the 1D cell centres x."""
    idx = np.asarray(idx)
    return x[idx % n], x[idx // n]
