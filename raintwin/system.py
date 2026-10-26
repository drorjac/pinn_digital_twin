"""The twin as a system: one ensemble, any mix of links (L), weather stations (P) and radar (R)."""

import numpy as np

from .core import RainTwin, smooth_noise

# observation errors: links and stations std = abs + rel * value (mm/h); the radar observes
# log(R + r0) directly, with a constant std in log units, and gets a learned bias
ERRORS = {"L": (0.3, 0.15), "P": (0.5, 0.3), "R": (0.4, 0.0)}     # radar error in log units
POINT_LOG_ERR = 0.3                                                 # stations observed in log units


def build_twin(cube, sensors, radar_thin=2, seed=1, errors=None, radar_bias=None, radar_log=True,
                point_log=False, **kw):
    """A RainTwin on the cube's grid observing the sensor letters in `sensors` (e.g. "LPR").
    The links are always the first set; when L is not used their readings are passed as NaN.
    point_log=True: the stations observe log(R + r0) like the radar, with error POINT_LOG_ERR."""
    E = dict(ERRORS, **(errors or {}))
    tw = RainTwin(cube.x, cube.y, cube.links, err_abs=E["L"][0], err_rel=E["L"][1],
                  seed=seed, **kw)
    if "P" in sensors:
        tw.add_obs("P", cube.pws["x"], cube.pws["y"], cube.pws["x"], cube.pws["y"],
                   *((POINT_LOG_ERR, 0.0) if point_log else E["P"]), log=point_log)
    if "R" in sensors:
        ny, nx = cube.y.size, cube.x.size
        r, c = np.meshgrid(np.arange(0, ny, radar_thin), np.arange(0, nx, radar_thin), indexing="ij")
        tw._radar_cells = (r * nx + c).ravel()
        X, Y = cube.x[c].ravel(), cube.y[r].ravel()
        # the radar's bias is only identifiable when another sensor fixes the level
        bias = len(sensors) > 1 if radar_bias is None else radar_bias
        tw.add_obs("R", X, Y, X, Y, *E["R"], bias=bias, log=radar_log)
    tw.sensors = sensors
    return tw


def observe(tw, cube, k):
    """The analysis inputs at step k for the sensors this twin uses."""
    y = cube.link_rain([k])[0] if "L" in tw.sensors else np.full(tw.m, np.nan)
    extra = {}
    if "P" in tw.sensors:
        extra["P"] = cube.pws_rain([k])[0]
    if "R" in tw.sensors:
        extra["R"] = cube.radar([k])[0].ravel()[tw._radar_cells]
    return y, extra


def first_guess(tw, cube, k):
    """Start from the radar if the twin uses it, otherwise from the mean of its sensors."""
    if "R" in tw.sensors:
        f = np.nan_to_num(cube.radar([k])[0])
    else:
        y, extra = observe(tw, cube, k)
        v = np.concatenate([np.ravel(y)] + [np.ravel(e) for e in extra.values()])
        f = np.full((cube.y.size, cube.x.size), max(float(np.nanmean(v)) if np.isfinite(v).any() else 0.1, 0.1))
    return f


def run(cube, idx, sensors, keep=None, radius=3.0, **kw):
    """Run the twin over steps idx. Returns a dict of (T, ny, nx) maps: `mean` and `spread` of
    the ensemble, `now` (mean plus the local correction at links and stations, see
    RainTwin.now_map), and the learned velocity per step. keep: optional callable(k, tw)."""
    opts = dict(r0=1.0, v_max=60.0)
    opts.update(kw)
    tw = build_twin(cube, sensors, **opts)
    tw.initialise(first_guess(tw, cube, idx[0]), background=0.1, vel_std=20)
    out = {"mean": [], "spread": [], "now": [], "V": []}
    for j, k in enumerate(idx):
        if j:
            tw.forecast(getattr(cube, "step_h", 5 / 60))
        y, extra = observe(tw, cube, k)
        tw.analysis(y, extra=extra)
        R = tw.rain
        out["mean"].append(R.mean(0)); out["spread"].append(R.std(0)); out["V"].append(tw.V.mean(0))
        out["now"].append(tw.now_map(y, radius=radius, extra=extra, point_sets=("P",)))
        if keep is not None:
            keep(k, tw)
    return {key: np.array(v) for key, v in out.items()}


class RadarErrorTwin(RainTwin):
    """A twin of the radar's error instead of the rain.

    State: z(x, t) = log of the factor by which the radar is wrong, plus the storm velocity.
    Rain = (radar + floor) * exp(z) - floor, so the radar gives the detail and the twin a smooth
    correction; `floor` lets links and stations add rain where the radar sees none. The factor
    moves with the storm and relaxes to no correction (z = 0) with time scale tau_h. Links
    and stations observe the rain, so they correct z. Call set_radar() before each analysis.
    """

    def __init__(self, x, y, links, floor=0.5, z_lim=3.0, **kw):
        kw.setdefault("tau_h", 3.0)
        super().__init__(x, y, links, **kw)
        self.floor = floor
        self.z_min, self.z_max = -z_lim, z_lim
        self.base = np.zeros(self.X.shape)

    def set_radar(self, R):
        self.base = np.nan_to_num(np.asarray(R, float))

    def to_rain(self, Z):
        b = self.base if Z.shape[-2:] == self.base.shape else self.base.ravel()
        return np.clip((b + self.floor) * np.exp(Z) - self.floor, 0.0, None)

    def initialise(self, vel_mean=(0.0, 0.0), vel_std=15.0, spread=0.5, **_):
        self.mu = 0.0
        self.Z = spread * smooth_noise((self.N,) + self.X.shape, self.rng, self.noise_cells)
        self.V = np.asarray(vel_mean, float)[None] + vel_std * self.rng.standard_normal((self.N, 2))
        self.G = np.zeros(self.N)
        self.B = {}


def run_error_twin(cube, idx, sensors="LP", radius=3.0, seed=1, errors=None, **kw):
    """The radar-error twin over steps idx, corrected by links (L) and/or stations (P)."""
    E = dict(ERRORS, **(errors or {}))
    opts = dict(r0=1.0, v_max=60.0, seed=seed, err_abs=E["L"][0], err_rel=E["L"][1])
    opts.update(kw)
    tw = RadarErrorTwin(cube.x, cube.y, cube.links, **opts)
    if "P" in sensors:
        tw.add_obs("P", cube.pws["x"], cube.pws["y"], cube.pws["x"], cube.pws["y"], *E["P"])
    tw.sensors = sensors
    tw.initialise(vel_std=20)
    out = {"mean": [], "spread": [], "now": [], "V": [], "factor": []}
    for j, k in enumerate(idx):
        if j:
            tw.forecast(getattr(cube, "step_h", 5 / 60))
        tw.set_radar(cube.radar([k])[0])
        y, extra = observe(tw, cube, k)
        tw.analysis(y, extra=extra)
        R = tw.rain
        out["mean"].append(R.mean(0)); out["spread"].append(R.std(0)); out["V"].append(tw.V.mean(0))
        out["factor"].append(np.exp(tw.Z.mean(0)))
        out["now"].append(tw.now_map(y, radius=radius, extra=extra, point_sets=("P",)))
    return {key: np.array(v) for key, v in out.items()}


def wet_windows(cube, split_values=(0, 1), min_rate=0.03, pad_steps=12, gap_steps=24):
    """Contiguous runs of steps in the given splits where links or radar see rain (domain mean
    above min_rate mm/h), padded and merged across short gaps. List of index arrays."""
    sel = np.isin(cube.split, split_values)
    lk = np.nanmean(cube._link_rain, 1)
    wet = sel & (np.nan_to_num(lk) > min_rate)
    k = np.flatnonzero(wet)
    if k.size == 0:
        return []
    # a jump in the clock (New York's cube keeps only wet days) also ends a window
    jump = np.r_[False, np.diff(cube.times.values).astype("timedelta64[m]").astype(int) > 5]
    runs, start, prev = [], k[0], k[0]
    for i in k[1:]:
        if i - prev > gap_steps or jump[prev + 1:i + 1].any():
            runs.append((start, prev)); start = i
        prev = i
    runs.append((start, prev))
    out = []
    for a, b in runs:
        idx = np.arange(max(a - pad_steps, 0), min(b + pad_steps, len(cube.times) - 1) + 1)
        idx = idx[np.isin(cube.split[idx], split_values)]
        t = np.flatnonzero(np.r_[True, jump[idx[1:]]])            # keep the part around the rain
        piece = np.searchsorted(t, np.searchsorted(idx, a), side="right") - 1
        idx = idx[t[piece]:(t[piece + 1] if piece + 1 < t.size else None)]
        if idx.size > 2 * pad_steps:
            out.append(idx)
    return out
