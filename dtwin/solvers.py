"""Physics models used as the 'physical asset' and as the twin's internal model."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter


# ----------------------------------------------------------------------------
# 1D heat equation
# ----------------------------------------------------------------------------
class Heat1D:
    """u_t = alpha * u_xx + f(x, t) on [0, L], Crank-Nicolson time stepping.

    bc = "dirichlet": the two end values stay fixed at their initial values.
    bc = "neumann":   insulated ends (zero flux).
    """

    def __init__(self, nx=101, L=1.0, alpha=0.01, dt=0.01, bc="dirichlet"):
        self.nx, self.L, self.alpha, self.dt, self.bc = nx, L, alpha, dt, bc
        self.x = np.linspace(0.0, L, nx)
        self.dx = self.x[1] - self.x[0]

        main = -2.0 * np.ones(nx)
        off = np.ones(nx - 1)
        D = sp.diags([off, main, off], [-1, 0, 1], format="lil")
        if bc == "dirichlet":
            D[0, :] = 0.0
            D[-1, :] = 0.0
        elif bc == "neumann":
            D[0, 1] = 2.0
            D[-1, -2] = 2.0
        else:
            raise ValueError(bc)
        self.D = (D / self.dx**2).tocsc()

        # forcing is switched off on fixed (Dirichlet) nodes
        self.mask = np.ones(nx)
        if bc == "dirichlet":
            self.mask[[0, -1]] = 0.0

        I = sp.identity(nx, format="csc")
        self._A = (I - 0.5 * dt * alpha * self.D).tocsc()
        self._B = (I + 0.5 * dt * alpha * self.D).tocsc()
        self._lu = spla.splu(self._A)

    def step(self, u, f=None):
        rhs = self._B @ u
        if f is not None:
            rhs = rhs + self.dt * self.mask * f
        return self._lu.solve(rhs)

    def simulate(self, u0, n_steps, source=None):
        """source: callable (x, t) -> array, or None. Returns t (n+1,), U (n+1, nx)."""
        U = np.empty((n_steps + 1, self.nx))
        U[0] = u0
        t = np.arange(n_steps + 1) * self.dt
        for k in range(n_steps):
            f = None if source is None else source(self.x, t[k])
            U[k + 1] = self.step(U[k], f)
        return t, U

    def transition_matrices(self):
        """Dense M, G such that u_{k+1} = M u_k + G f_k (for Kalman filtering)."""
        A = self._A.toarray()
        M = np.linalg.solve(A, self._B.toarray())
        G = np.linalg.solve(A, self.dt * np.diag(self.mask))
        return M, G


# ----------------------------------------------------------------------------
# 1D viscous Burgers: exact solution via the Cole-Hopf transform
# ----------------------------------------------------------------------------
def burgers_exact(x, t, nu=0.01 / np.pi, n_quad=4000):
    """Exact solution of u_t + u u_x = nu u_xx, u(x,0) = -sin(pi x), u(+-1,t) = 0.

    Cole-Hopf: u = -int sin(pi(x-eta)) w(eta) d eta / int w(eta) d eta with
    w = exp(-cos(pi(x-eta)) / (2 pi nu) - eta^2 / (4 nu t)), evaluated stably in log-space.
    Returns array of shape (len(t), len(x)).
    """
    x = np.asarray(x, float)
    t = np.atleast_1d(np.asarray(t, float))
    U = np.empty((t.size, x.size))
    for i, ti in enumerate(t):
        if ti <= 0:
            U[i] = -np.sin(np.pi * x)
            continue
        s = np.sqrt(2 * nu * ti)
        eta = np.linspace(-10 * s, 10 * s, n_quad)
        xe = x[:, None] - eta[None, :]
        logw = -np.cos(np.pi * xe) / (2 * np.pi * nu) - eta[None, :] ** 2 / (4 * nu * ti)
        w = np.exp(logw - logw.max(axis=1, keepdims=True))
        U[i] = -(np.sin(np.pi * xe) * w).sum(1) / w.sum(1)
    return U


# ----------------------------------------------------------------------------
# 2D advection-diffusion-reaction on the unit square
# ----------------------------------------------------------------------------
def cellular_flow(X, Y, amplitude=1.0, L=1.0):
    """Divergence-free closed vortex: psi = A sin(pi x/L) sin(pi y/L). Zero normal flow at walls."""
    k = np.pi / L
    U = amplitude * k * np.sin(k * X) * np.cos(k * Y)
    V = -amplitude * k * np.cos(k * X) * np.sin(k * Y)
    return U, V


def gaussian_blob(X, Y, x0, y0, width=0.05, amp=1.0):
    return amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * width**2))


class AdvDiff2D:
    """c_t + a (u c_x + v c_y) = kappa lap(c) - decay c + s(x, y, t), zero-flux walls.

    Finite volumes on an n x n cell-centred grid, conservative first-order upwind advection,
    central diffusion, explicit Euler. Every array may carry leading batch axes
    (..., ny, nx), so a whole ensemble is stepped at once. kappa and the flow scale
    `a` may be scalars or arrays broadcastable to (..., 1, 1) for per-member parameters.
    """

    def __init__(self, n=64, L=1.0, kappa=1e-3, flow_amplitude=0.05, decay=0.0,
                 dt=None, kappa_max=None, flow_max=None):
        self.n, self.L, self.kappa, self.decay = n, L, kappa, decay
        self.dx = L / n
        self.x = (np.arange(n) + 0.5) * self.dx
        self.X, self.Y = np.meshgrid(self.x, self.x)  # arrays (ny, nx); rows are y
        self.flow_amplitude = flow_amplitude
        self.U, self.V = cellular_flow(self.X, self.Y, 1.0, L)  # unit-amplitude, cell centres (for plots)
        # face velocities from the stream function at cell corners: exactly divergence-free on the
        # grid and zero normal flow through the walls, so the scheme conserves mass to round-off
        xc = np.arange(n + 1) * self.dx
        psi = np.sin(np.pi * xc / L)[:, None] * np.sin(np.pi * xc / L)[None, :]  # (corner y, corner x)
        self.Uf = (psi[1:, :] - psi[:-1, :]) / self.dx    # x-faces, shape (ny, nx+1)
        self.Vf = -(psi[:, 1:] - psi[:, :-1]) / self.dx   # y-faces, shape (ny+1, nx)
        km = kappa_max if kappa_max is not None else kappa
        fm = flow_max if flow_max is not None else flow_amplitude
        if dt is None:
            vmax = fm * np.pi / L + 1e-12
            dt = 0.5 * min(self.dx / (2 * vmax), self.dx**2 / (4 * km + 1e-12))
        self.dt = dt

    def rhs(self, c, s=None, kappa=None, flow=None, decay=None):
        kappa = self.kappa if kappa is None else kappa
        flow = self.flow_amplitude if flow is None else flow
        decay = self.decay if decay is None else decay
        pad = [(0, 0)] * (c.ndim - 2) + [(1, 1), (1, 1)]
        cp = np.pad(c, pad, mode="edge")  # zero-gradient walls
        lap = (cp[..., 1:-1, 2:] + cp[..., 1:-1, :-2] + cp[..., 2:, 1:-1] + cp[..., :-2, 1:-1]
               - 4 * c) / self.dx**2
        # upwind fluxes on interior faces; wall faces carry no flux
        uf = flow * self.Uf[:, 1:-1]
        vf = flow * self.Vf[1:-1, :]
        Fx = np.maximum(uf, 0) * c[..., :, :-1] + np.minimum(uf, 0) * c[..., :, 1:]
        Fy = np.maximum(vf, 0) * c[..., :-1, :] + np.minimum(vf, 0) * c[..., 1:, :]
        zx = [(0, 0)] * (c.ndim - 1) + [(1, 1)]
        zy = [(0, 0)] * (c.ndim - 2) + [(1, 1), (0, 0)]
        Fx, Fy = np.pad(Fx, zx), np.pad(Fy, zy)
        adv = (Fx[..., :, 1:] - Fx[..., :, :-1] + Fy[..., 1:, :] - Fy[..., :-1, :]) / self.dx
        out = kappa * lap - adv - decay * c
        if s is not None:
            out = out + s
        return out

    def step(self, c, s=None, **params):
        return c + self.dt * self.rhs(c, s, **params)

    def simulate(self, c0, t_end, source=None, save_every=1, **params):
        """source: callable t -> field (or None). Returns t_saved, C (n_saved, ..., n, n)."""
        n_steps = int(round(t_end / self.dt))
        c = np.array(c0, float)
        ts, Cs = [0.0], [c.copy()]
        for k in range(n_steps):
            t = k * self.dt
            c = self.step(c, None if source is None else source(t), **params)
            if (k + 1) % save_every == 0:
                ts.append((k + 1) * self.dt)
                Cs.append(c.copy())
        return np.array(ts), np.array(Cs)


def smooth_noise(shape, rng, sigma_cells=3.0, std=1.0):
    """Spatially correlated Gaussian noise on the last two axes (model-error samples)."""
    z = rng.standard_normal(shape)
    sig = [0] * (len(shape) - 2) + [sigma_cells, sigma_cells]
    z = gaussian_filter(z, sig, mode="nearest")
    z /= z.reshape(-1, shape[-2] * shape[-1]).std(axis=1).mean() + 1e-12
    return std * z


def random_blobs(X, Y, rng, n_blobs=3, width=(0.04, 0.1), amp=(0.5, 1.0)):
    """A random initial condition made of a few Gaussian blobs."""
    c = np.zeros_like(X)
    for _ in range(n_blobs):
        x0, y0 = rng.uniform(0.15, 0.85, 2)
        c += gaussian_blob(X, Y, x0, y0, rng.uniform(*width), rng.uniform(*amp))
    return c
