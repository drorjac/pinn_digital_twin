"""Data assimilation: keep the twin's state synchronised with the asset's sensors."""

import numpy as np


class KalmanFilter:
    """Linear Kalman filter for x_{k+1} = M x_k + G u_k + w,  y_k = H x_k + v."""

    def __init__(self, M, H, Q, R, G=None):
        self.M, self.H, self.Q, self.R, self.G = M, H, Q, R, G

    def predict(self, x, P, u=None):
        x = self.M @ x
        if u is not None and self.G is not None:
            x = x + self.G @ u
        P = self.M @ P @ self.M.T + self.Q
        return x, P

    def update(self, x, P, y):
        H, R = self.H, self.R
        innov = y - H @ x
        S = H @ P @ H.T + R
        K = np.linalg.solve(S, H @ P).T  # = P H^T S^-1 (P, S symmetric)
        x = x + K @ innov
        IKH = np.eye(P.shape[0]) - K @ H
        P = IKH @ P @ IKH.T + K @ R @ K.T  # Joseph form: stays symmetric positive definite
        return x, P, innov, S


def gaspari_cohn(r):
    """Compactly supported 5th-order correlation function; r = distance / radius, zero for r >= 2."""
    r = np.abs(r)
    out = np.zeros_like(r)
    a = r <= 1
    b = (r > 1) & (r < 2)
    ra, rb = r[a], r[b]
    out[a] = -0.25 * ra**5 + 0.5 * ra**4 + 0.625 * ra**3 - 5 / 3 * ra**2 + 1
    out[b] = (rb**5 / 12 - 0.5 * rb**4 + 0.625 * rb**3 + 5 / 3 * rb**2
              - 5 * rb + 4 - 2 / 3 / rb)
    return out


class EnKF:
    """Stochastic (perturbed-observation) ensemble Kalman filter.

    The ensemble E has shape (N, n_state + n_params). Only the first n_state entries
    are field values; trailing entries are parameters estimated jointly
    (state augmentation). Optional multiplicative inflation and Gaspari-Cohn
    covariance localization (needs `coords`, an (n_state, d) array of positions).
    """

    def __init__(self, obs_idx, obs_std, inflation=1.0, loc_radius=None, coords=None):
        self.obs_idx = np.asarray(obs_idx, int)
        self.obs_std = obs_std
        self.inflation = inflation
        self.loc_radius = loc_radius
        self.coords = coords
        self._rho = None

    def _localization(self, n_total):
        if self._rho is None or self._rho[0].shape[0] != n_total:
            n_state = self.coords.shape[0]
            oc = self.coords[self.obs_idx]
            d_xo = np.linalg.norm(self.coords[:, None, :] - oc[None], axis=-1)
            d_oo = np.linalg.norm(oc[:, None, :] - oc[None], axis=-1)
            rho_xo = np.ones((n_total, self.obs_idx.size))
            rho_xo[:n_state] = gaspari_cohn(d_xo / self.loc_radius)  # parameters: global, no taper
            self._rho = (rho_xo, gaspari_cohn(d_oo / self.loc_radius))
        return self._rho

    def analysis(self, E, y, rng):
        N = E.shape[0]
        mean = E.mean(0)
        A = (E - mean) * self.inflation
        E = mean + A
        HE = E[:, self.obs_idx]
        HA = HE - HE.mean(0)
        PHt = A.T @ HA / (N - 1)
        HPHt = HA.T @ HA / (N - 1)
        m = self.obs_idx.size
        R = self.obs_std**2 * np.eye(m)
        Y = y[None, :] + rng.normal(0.0, self.obs_std, (N, m))
        D = Y - HE
        if self.loc_radius is None:
            return E + np.linalg.solve(HPHt + R, D.T).T @ PHt.T
        # field: localized covariances. Parameters are global quantities, so they are updated
        # with the un-localized covariances (mixing the two inside one S^-1 is inconsistent).
        n_state = self.coords.shape[0]
        rho_xo, rho_oo = self._localization(E.shape[1])
        E = E.copy()
        E[:, :n_state] += np.linalg.solve(HPHt * rho_oo + R, D.T).T @ (PHt[:n_state] * rho_xo[:n_state]).T
        if E.shape[1] > n_state:
            E[:, n_state:] += np.linalg.solve(HPHt + R, D.T).T @ PHt[n_state:].T
        return E

    @staticmethod
    def nis(E, y, obs_idx, obs_std):
        """Normalised innovation squared of the ensemble mean; ~ chi^2(m) when the model is right."""
        HE = E[:, obs_idx]
        d = y - HE.mean(0)
        HA = HE - HE.mean(0)
        S = HA.T @ HA / (E.shape[0] - 1) + obs_std**2 * np.eye(d.size)
        return float(d @ np.linalg.solve(S, d))
