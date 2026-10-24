"""Rain twin: link observation operator, baselines, and an ensemble Kalman filter with motion."""

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter, map_coordinates


# ----------------------------------------------------------------------------
# Observation operator: a link sees the average rain along its path
# ----------------------------------------------------------------------------
def path_operator(xa, ya, xb, yb, x, y, n_samples=24):
    """Sparse H (n_links, ny*nx) averaging the field along straight paths.

    x, y: 1D cell-centre coordinates (regular, increasing). Path ends in the same units.
    Each path is sampled at n_samples points; each point spreads bilinearly onto its
    four surrounding cells. Points outside the grid are dropped and the row renormalised.
    A point sensor is a path with identical ends.
    """
    xa, ya, xb, yb = (np.atleast_1d(np.asarray(v, float)) for v in (xa, ya, xb, yb))
    nx, ny = x.size, y.size
    dx, dy = x[1] - x[0], y[1] - y[0]
    s = np.linspace(0.0, 1.0, n_samples)[None, :]
    px = (xa[:, None] + s * (xb - xa)[:, None] - x[0]) / dx
    py = (ya[:, None] + s * (yb - ya)[:, None] - y[0]) / dy
    rows, cols, vals = [], [], []
    for i in range(xa.size):
        fx, fy = px[i], py[i]
        ok = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
        fx, fy = fx[ok], fy[ok]
        if fx.size == 0:
            continue
        ix = np.minimum(fx.astype(int), nx - 2)
        iy = np.minimum(fy.astype(int), ny - 2)
        tx, ty = fx - ix, fy - iy
        for ddy, ddx, w in [(0, 0, (1 - tx) * (1 - ty)), (0, 1, tx * (1 - ty)),
                            (1, 0, (1 - tx) * ty), (1, 1, tx * ty)]:
            rows.append(np.full(fx.size, i))
            cols.append((iy + ddy) * nx + ix + ddx)
            vals.append(w / fx.size)
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(xa.size, nx * ny))
    H.sum_duplicates()
    return H


def segment_distance(px, py, xa, ya, xb, yb):
    """Distance from points (px, py) (any shape P) to segments (m,). Returns (P..., m)."""
    px, py = np.asarray(px)[..., None], np.asarray(py)[..., None]
    dx, dy = xb - xa, yb - ya
    L2 = np.maximum(dx**2 + dy**2, 1e-12)
    t = np.clip(((px - xa) * dx + (py - ya) * dy) / L2, 0.0, 1.0)
    return np.hypot(px - (xa + t * dx), py - (ya + t * dy))


# ----------------------------------------------------------------------------
# Baselines: inverse distance weighting
# ----------------------------------------------------------------------------
def idw_points(values, sx, sy, X, Y, power=2.0, radius=None):
    """IDW of point values onto grid coordinates X, Y. NaN values are skipped."""
    ok = np.isfinite(values)
    v, sx, sy = values[ok], sx[ok], sy[ok]
    d = np.hypot(X[..., None] - sx, Y[..., None] - sy)
    w = 1.0 / np.maximum(d, 1e-3) ** power
    if radius is not None:
        w = np.where(d <= radius, w, 0.0)
    wsum = w.sum(-1)
    out = (w * v).sum(-1) / np.where(wsum > 0, wsum, 1.0)
    return np.where(wsum > 0, out, 0.0)


def line_idw(values, xa, ya, xb, yb, X, Y, n_virtual=5, power=2.0, radius=None):
    """IDW with virtual gauges along each path, each carrying the link's path average."""
    s = np.linspace(0.1, 0.9, n_virtual)[None, :]
    vx = (xa[:, None] + s * (xb - xa)[:, None]).ravel()
    vy = (ya[:, None] + s * (yb - ya)[:, None]).ravel()
    return idw_points(np.repeat(values, n_virtual), vx, vy, X, Y, power, radius)


# ----------------------------------------------------------------------------
# Model pieces
# ----------------------------------------------------------------------------
def advect(F, u, v, hours, dx, periodic=False):
    """Move fields F (N, ny, nx) by velocities u, v (N,) [length/h] for `hours`.

    Semi-Lagrangian: the value at a cell is read from where it came from. Outside the grid,
    the edge value is used (rain entering from upstream looks like the boundary).
    """
    N, ny, nx = F.shape
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    out = np.empty_like(F)
    mode = "grid-wrap" if periodic else "nearest"
    for k in range(N):
        src_c = jj - u[k] * hours / dx
        src_r = ii - v[k] * hours / dx
        out[k] = map_coordinates(F[k], [src_r, src_c], order=1, mode=mode)
    return out


def smooth_noise(shape, rng, sigma_cells, mode="nearest"):
    """Gaussian noise smoothed over the last two axes, rescaled to unit std."""
    z = rng.standard_normal(shape)
    axes_sigma = (0,) * (len(shape) - 2) + (sigma_cells, sigma_cells)
    z = gaussian_filter(z, axes_sigma, mode=mode)
    return z / z.std()


def gaspari_cohn(r):
    r = np.abs(r)
    out = np.zeros_like(r)
    a = r <= 1
    b = (r > 1) & (r < 2)
    ra, rb = r[a], r[b]
    out[a] = -0.25 * ra**5 + 0.5 * ra**4 + 0.625 * ra**3 - 5 / 3 * ra**2 + 1
    out[b] = rb**5 / 12 - 0.5 * rb**4 + 0.625 * rb**3 + 5 / 3 * rb**2 - 5 * rb + 4 - 2 / 3 / rb
    return out


# ----------------------------------------------------------------------------
# The twin
# ----------------------------------------------------------------------------
class RainTwin:
    """Ensemble twin of a rain field and its motion.

    Each member stores z = log(R + r0) on the grid, plus a velocity (u, v) in length units
    per hour. Working in log space keeps rain positive and makes model errors multiplicative.
    Forecast: advect z, relax it towards a background level mu with time scale tau_h (so
    regions no link sees cannot drift without bound), add smooth noise. Analysis: a
    stochastic EnKF; links observe the path average of R, computed member by member, so the
    nonlinearity of exp() is handled by the ensemble. Field updates are localised around
    each link (Gaspari-Cohn on the cell-to-segment distance); the velocity update is not.
    """

    def __init__(self, x, y, links, n_members=40, loc_radius=15.0, inflation=1.03,
                 q_log=0.15, q_vel=0.5, tau_h=2.0, noise_cells=3.0, r0=0.1,
                 periodic=False, err_abs=0.3, err_rel=0.15, r_max=200.0, growth=False,
                 q_growth=0.05, link_bias_gain=0.0, v_max=None, seed=0):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.dx = self.x[1] - self.x[0]
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.links = {k: np.asarray(v, float) for k, v in links.items()}   # xa, ya, xb, yb
        L = self.links
        self.H = path_operator(L["xa"], L["ya"], L["xb"], L["yb"], self.x, self.y)
        self.m = self.H.shape[0]
        self.N = n_members
        self.inflation, self.periodic = inflation, periodic
        self.q_log, self.q_vel, self.tau_h, self.noise_cells, self.r0 = q_log, q_vel, tau_h, noise_cells, r0
        self.err_abs, self.err_rel = err_abs, err_rel
        self.z_max = np.log(r_max + r0)           # cap: no rain rate above r_max mm/h
        self.z_min = np.log(r0)                   # = no rain
        self.rng = np.random.default_rng(seed)
        self.loc_radius = loc_radius
        self.growth, self.q_growth = growth, q_growth
        # per-link offset [mm/h], learned as a slow running mean of what the map cannot explain
        self.link_bias_gain = link_bias_gain
        self.v_max = v_max                         # optional cap on each member's speed
        self.b_link = np.zeros(self.m)
        # observation sets; the links are always the first one
        self.sets = []
        self._geom = {k: np.empty(0) for k in ("xa", "ya", "xb", "yb")}
        self.add_obs("links", L["xa"], L["ya"], L["xb"], L["yb"], err_abs, err_rel, H=self.H)

    def add_obs(self, name, xa, ya, xb, yb, err_abs, err_rel, bias=False, H=None, log=False):
        """Add a sensor set (paths; a point is a path with equal ends). With bias=True the set
        reads exp(b) times the true value, and the log-bias b is estimated with the state.

        log=True: the set observes log(R + r0) averaged by H, which is linear in the state, so
        the update is exact for it. Its error is then err_abs in log units (err_rel unused).
        Suits gridded sensors such as radar, whose cells are the twin's own cells."""
        xa, ya, xb, yb = (np.atleast_1d(np.asarray(v, float)) for v in (xa, ya, xb, yb))
        if H is None:
            H = path_operator(xa, ya, xb, yb, self.x, self.y)
        start = sum(s["n"] for s in self.sets)
        self.sets.append(dict(name=name, H=H, n=H.shape[0], sl=slice(start, start + H.shape[0]),
                              err_abs=err_abs, err_rel=err_rel, bias=bias, log=log))
        for k, v in zip(("xa", "ya", "xb", "yb"), (xa, ya, xb, yb)):
            self._geom[k] = np.r_[self._geom[k], v]
        self.H_all = sp.vstack([s["H"] for s in self.sets]).tocsr()
        if self.loc_radius:
            g, c = self._geom, self.loc_radius / 2
            d_xy = segment_distance(self.X.ravel(), self.Y.ravel(), g["xa"], g["ya"], g["xb"], g["yb"])
            self.rho_xy = gaspari_cohn(d_xy / c).astype(np.float32)
            mx, my = 0.5 * (g["xa"] + g["xb"]), 0.5 * (g["ya"] + g["yb"])
            # midpoint distances: GC of a Euclidean distance keeps rho_yy positive definite
            self.rho_yy = gaspari_cohn(np.hypot(mx[:, None] - mx, my[:, None] - my) / c)

    def to_rain(self, Z):
        return np.clip(np.exp(Z) - self.r0, 0.0, None)

    def to_log(self, R):
        return np.log(np.clip(R, 0.0, None) + self.r0)

    # -- set-up ----------------------------------------------------------------
    def initialise(self, first_guess, background=None, vel_mean=(0.0, 0.0), vel_std=15.0, spread=0.8):
        """Members = log(first_guess) + smooth noise of size `spread`; velocities ~ N(vel_mean, vel_std^2).

        background: rain rate the field relaxes to where no data arrive (default: mean of first_guess).
        """
        bg = float(np.mean(first_guess)) if background is None else background
        self.mu = float(self.to_log(np.array(bg)))
        z = smooth_noise((self.N,) + first_guess.shape, self.rng, self.noise_cells)
        self.Z = np.clip(self.to_log(first_guess)[None] + spread * z, self.z_min, self.z_max)
        self.V = np.asarray(vel_mean, float)[None] + vel_std * self.rng.standard_normal((self.N, 2))
        # growth rate of rain anomalies [1/h] (0 = only the fixed relaxation tau_h)
        self.G = 0.3 * self.rng.standard_normal(self.N) if self.growth else np.zeros(self.N)
        self.B = {s["name"]: 0.3 * self.rng.standard_normal(self.N) for s in self.sets if s["bias"]}

    # -- forecast --------------------------------------------------------------
    def _step(self, Z, hours):
        Z = advect(Z, self.V[:, 0], self.V[:, 1], hours, self.dx, self.periodic)
        phi = np.exp((self.G - 1 / self.tau_h) * hours)[:, None, None]
        return np.minimum(self.mu + phi * (Z - self.mu), self.z_max)

    def forecast(self, hours, stochastic=True):
        self.Z = self._step(self.Z, hours)
        if stochastic:
            f = np.sqrt(hours * 12)                       # noise sizes are per 5 minutes
            self.Z = self.Z + self.q_log * f * smooth_noise(self.Z.shape, self.rng, self.noise_cells)
            self.V = self.V + self.q_vel * f * self.rng.standard_normal(self.V.shape)
            if self.growth:
                self.G = self.G + self.q_growth * f * self.rng.standard_normal(self.N)

    def predict(self, n_steps, hours):
        """Forecast without model noise. Returns ensemble-mean rain (n_steps, ny, nx) and spread."""
        Z, mean, sd = self.Z.copy(), [], []
        for _ in range(n_steps):
            Z = self._step(Z, hours)
            R = self.to_rain(Z)
            mean.append(R.mean(0)); sd.append(R.std(0))
        return np.array(mean), np.array(sd)

    # -- analysis --------------------------------------------------------------
    def obs_std(self, y):
        return self.err_abs + self.err_rel * np.abs(y)

    def _params(self):
        cols = [self.V, self.G[:, None]] + [b[:, None] for b in self.B.values()]
        return np.hstack(cols)

    def _set_params(self, P):
        self.V = P[:, :2]
        if self.v_max:
            sp_ = np.hypot(self.V[:, 0], self.V[:, 1])[:, None]
            self.V = self.V * np.minimum(1.0, self.v_max / np.maximum(sp_, 1e-9))
        if self.growth:
            self.G = P[:, 2]
        for j, k in enumerate(self.B):
            self.B[k] = P[:, 3 + j]

    def _predict_obs(self, Zf, P):
        """Each member's readings of every sensor, (N, n_obs)."""
        R = self.to_rain(Zf)
        out = []
        for s in self.sets:
            if s["log"]:
                Y = (s["H"] @ Zf.T).T
                if s["bias"]:
                    Y = Y + P[:, 3 + list(self.B).index(s["name"])][:, None]
            else:
                Y = (s["H"] @ R.T).T
                if s["bias"]:
                    Y = Y * np.exp(P[:, 3 + list(self.B).index(s["name"])])[:, None]
            out.append(Y)
        return np.hstack(out)

    def analysis(self, y, extra=None, n_iter=1):
        """Assimilate link path averages y (m,), NaN = missing, plus optional readings of other
        sensor sets, extra = {name: values}. Returns innovation statistics.

        n_iter > 1 runs multiple data assimilation (ES-MDA): the same data are assimilated
        n_iter times with the error variance inflated n_iter-fold, which handles the
        nonlinear exp() better than one large update.
        """
        y = np.asarray(y, float) - self.b_link
        yall = np.full(self.H_all.shape[0], np.nan)
        yall[self.sets[0]["sl"]] = y
        for s in self.sets[1:]:
            if extra is not None and s["name"] in extra:
                v = np.asarray(extra[s["name"]], float)
                yall[s["sl"]] = self.to_log(v) if s["log"] else v
        ok = np.isfinite(yall)
        if not ok.any():
            return {}
        sig = np.concatenate([np.full(s["n"], s["err_abs"]) if s["log"] else
                              s["err_abs"] + s["err_rel"] * np.abs(np.nan_to_num(yall[s["sl"]]))
                              for s in self.sets])
        stats = None
        for it in range(n_iter):
            st = self._update(yall, sig, ok, alpha=n_iter, inflate=(it == 0))
            stats = stats or st
        lk = ok[self.sets[0]["sl"]]
        Ym = self.H @ self.mean.ravel()
        if self.link_bias_gain:
            self.b_link[lk] += self.link_bias_gain * (y[lk] - Ym[lk])
        if lk.any() and y[lk].std() > 0:
            stats["fit_corr"] = float(np.corrcoef(Ym[lk] + self.b_link[lk], y[lk] + self.b_link[lk])[0, 1])
        else:
            stats["fit_corr"] = np.nan
        return stats

    def link_readings(self, R):
        """What the links would read for rain field(s) R (..., ny, nx), offsets included."""
        R = np.asarray(R)
        flat = R.reshape(-1, R.shape[-2] * R.shape[-1])
        return (self.H @ flat.T).T.reshape(R.shape[:-2] + (self.m,)) + self.b_link

    def _update(self, y, sig_all, ok, alpha=1.0, inflate=True):
        N = self.N
        Zf = self.Z.reshape(N, -1)
        infl = self.inflation if inflate else 1.0
        A = (Zf - Zf.mean(0)) * infl
        Zf = Zf.mean(0) + A
        P = self._params()
        Ap = (P - P.mean(0)) * infl
        P = P.mean(0) + Ap
        Yp = self._predict_obs(Zf, P)[:, ok]                # (N, m_ok): each member's readings
        B = Yp - Yp.mean(0)
        yo = y[ok]
        sig = sig_all[ok] * np.sqrt(alpha)
        Cyy = B.T @ B / (N - 1)
        R = np.diag(sig**2)
        D = yo + sig * self.rng.standard_normal((N, yo.size)) - Yp
        if self.loc_radius:
            W = np.linalg.solve(self.rho_yy[np.ix_(ok, ok)] * Cyy + R, D.T)      # (m, N)
            # field increment = (rho_xy * Cxy) W, with Cxy = A^T B / (N-1), done in blocks
            dZ = np.empty_like(A.T)
            for i in range(0, A.shape[1], 8192):
                blk = slice(i, i + 8192)
                dZ[blk] = (self.rho_xy[blk][:, ok] * (A[:, blk].T @ B / (N - 1))) @ W
            dZ = dZ.T
        else:
            dZ = D @ np.linalg.solve(Cyy + R, (A.T @ B / (N - 1)).T)
        Wp = np.linalg.solve(Cyy + R, D.T)
        P = P + (Ap.T @ B / (N - 1) @ Wp).T
        self.Z = np.clip(Zf + dZ, self.z_min, self.z_max).reshape(self.Z.shape)
        self._set_params(P)
        d = yo - Yp.mean(0)
        return {"nis": float(d @ np.linalg.solve(Cyy + R, d)), "m": int(ok.sum()),
                "rmse_obs": float(np.sqrt(np.mean(d**2)))}

    def now_map(self, y, radius=3.0, extra=None, point_sets=()):
        """The map to show now: twin mean plus an IDW of what the twin misses at each sensor,
        spread no further than `radius` from it. Far from sensors it is the twin.

        y: link readings; extra/point_sets: readings of other sets (by name) whose misses are
        also used, each sensor as a point at its path midpoint (stations are points anyway).
        """
        L = self.links
        mean = self.mean
        s = np.linspace(0.1, 0.9, 5)[None, :]
        miss = np.asarray(y, float) - self.link_readings(mean)
        vx = (L["xa"][:, None] + s * (L["xb"] - L["xa"])[:, None]).ravel()
        vy = (L["ya"][:, None] + s * (L["yb"] - L["ya"])[:, None]).ravel()
        vals = [np.repeat(miss, 5)]
        px, py = [vx], [vy]
        for st in self.sets[1:]:
            if st["name"] in point_sets and extra is not None and st["name"] in extra:
                g = self._geom
                vals.append(np.asarray(extra[st["name"]], float) - st["H"] @ mean.ravel())
                px.append(0.5 * (g["xa"][st["sl"]] + g["xb"][st["sl"]]))
                py.append(0.5 * (g["ya"][st["sl"]] + g["yb"][st["sl"]]))
        fix = idw_points(np.concatenate(vals), np.concatenate(px), np.concatenate(py),
                         self.X, self.Y, radius=radius)
        return np.clip(mean + fix, 0.0, None)

    @property
    def rain(self):
        return self.to_rain(self.Z)

    @property
    def mean(self):
        return self.rain.mean(0)

    @property
    def spread(self):
        return self.rain.std(0)


# ----------------------------------------------------------------------------
# Forecast blending at the links
# ----------------------------------------------------------------------------
class ResidualBlend:
    """Link forecast = twin forecast + the twin's current miss at each link, faded with lead
    time as exp(-lead / tau). tau = 0 is the pure twin, tau = inf keeps the miss for ever.

    tau is chosen among `taus` before every forecast, by the recent squared error of each
    candidate's forecasts at `verify_lead` steps. Only data already seen are used.
    """

    def __init__(self, taus=(0.0, 0.25, 0.5, 1.0, 2.0, np.inf), verify_lead=3, memory=0.7,
                 dt_h=5 / 60, start=1):
        self.taus, self.lead, self.memory, self.dt_h = taus, verify_lead, memory, dt_h
        self.err = None
        self.best = start
        self.issued = {}

    def fade(self, tau, h):
        return 0.0 if tau == 0 else float(np.exp(-h * self.dt_h / tau))

    def blend(self, P, miss, tau):
        """P: twin link forecasts (n_leads, m); miss: (m,) current y - twin at the links."""
        miss = np.nan_to_num(miss)
        return np.array([P[h - 1] + self.fade(tau, h) * miss for h in range(1, len(P) + 1)])

    def verify(self, k, y):
        """Score the forecasts issued `verify_lead` steps before step k against y (m,)."""
        kp = k - self.lead
        if kp not in self.issued:
            return
        P, miss = self.issued.pop(kp)
        ok = np.isfinite(y)
        e = np.array([np.mean((self.blend(P, miss, t)[self.lead - 1] - y)[ok] ** 2) for t in self.taus])
        self.err = e if self.err is None else self.memory * self.err + (1 - self.memory) * e
        self.best = int(np.argmin(self.err))

    def issue(self, k, P, miss):
        """Blend with the current best tau and remember the raw parts for later scoring."""
        self.issued[k] = (P, miss)
        return self.blend(P, miss, self.taus[self.best])

    @property
    def tau(self):
        return self.taus[self.best]


# ----------------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------------
def scores(est, ref, thr=1.0, mask=None):
    """RMSE, bias, correlation and CSI (rain >= thr mm/h) of est vs ref."""
    e, r = np.asarray(est, float), np.asarray(ref, float)
    if mask is not None:
        e, r = e[..., mask], r[..., mask]
    e, r = e.ravel(), r.ravel()
    ok = np.isfinite(e) & np.isfinite(r)
    e, r = e[ok], r[ok]
    hit = np.sum((e >= thr) & (r >= thr))
    miss = np.sum((e < thr) & (r >= thr))
    fa = np.sum((e >= thr) & (r < thr))
    corr = np.corrcoef(e, r)[0, 1] if e.std() > 0 and r.std() > 0 else np.nan
    return {"rmse": float(np.sqrt(np.mean((e - r) ** 2))), "bias": float(e.mean() - r.mean()),
            "corr": float(corr), "csi": float(hit / max(hit + miss + fa, 1))}
