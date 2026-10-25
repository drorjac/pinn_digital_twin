"""Extra runs for Part III: how sure are the scores, how many links are needed, is the spread right.

    python tools/extend_runs.py A [openmrg|openmesh]   # bootstrap CIs over storms, leave-one-gauge-out
    python tools/extend_runs.py D                      # link density: keep 10/25/50/100 % of the links
    python tools/extend_runs.py E                      # spread calibration: inflation / model-error sweep
    python tools/extend_runs.py report                 # print the tables of D and E from the cache

A only rescores the cached runs of chapters 10 and 12. D and E rerun the twin on the 10 Gothenburg
storms (about 6 min per configuration, spread over the CPU cores) and cache to cache/ext_*.pkl.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, pickle, time, copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from raintwin import system
from raintwin.core import line_idw, segment_distance
from raintwin.cube import Cube

CACHE = ROOT / "cache"
SPIN = 12
STATIC = ("R", "C", "P", "CP", "RC", "RP", "RCP")


class Setup:
    """The storms, scored steps and hourly gauge rain, as in chapters 10 and 12."""

    def __init__(self, network="openmrg"):
        c = self.c = Cube(network)
        self.events = c.events()
        self.GI = np.round(c.gauges["y"] / c.dx).astype(int), np.round(c.gauges["x"] / c.dx).astype(int)
        self.steps = [c.index(s - pd.Timedelta("1h"), e) for _, s, e in self.events]
        self.scored = np.concatenate([st[SPIN:] for st in self.steps])
        self.storm = np.concatenate([np.full(st.size - SPIN, i) for i, st in enumerate(self.steps)])
        self.hours = c.times[self.scored].ceil("h")
        self.G = c.gauge_rain(self.scored)
        self.GH = self.hourly(self.G)
        # the storm of each hourly row (an hour never spans two storms: they are days apart)
        first = pd.Series(self.storm, index=self.hours).groupby(level=0).first()
        self.storm_h = first.values

    def hourly(self, a):
        """5-min rates (T, ...) to hourly totals (H, ...); an hour needs all 12 steps."""
        a = np.asarray(a, float)
        flat = a.reshape(a.shape[0], -1) / 12
        h = pd.DataFrame(flat, index=self.hours).groupby(level=0).sum(min_count=12).values
        return h.reshape((h.shape[0],) + a.shape[1:])


def nrmse(E, G):
    ok = np.isfinite(E) & np.isfinite(G)
    return np.sqrt(np.mean((E[ok] - G[ok]) ** 2)) / G[ok].mean() if ok.any() else np.nan


def sq_parts(Eh, GH, storm_h, n):
    """Per storm: sum of squared errors, sum of gauge rain, count. Bootstrapping storms then only
    needs these sums, and NRMSE = sqrt(sse / n) / (sum / n) is exact for any resample."""
    ok = np.isfinite(Eh) & np.isfinite(GH)
    sse, tot, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    for s in range(n):
        m = ok & (storm_h == s)[:, None]
        sse[s], tot[s], cnt[s] = ((Eh - GH) ** 2)[m].sum(), GH[m].sum(), m.sum()
    return sse, tot, cnt


# ----------------------------------------------------------------------------
# A: confidence intervals
# ----------------------------------------------------------------------------
def run_A(network="openmrg", n_boot=5000, seed=0):
    S = Setup(network)
    runs = pickle.load(open(CACHE / ("ch10_runs.pkl" if network == "openmrg" else "ch12_runs.pkl"), "rb"))
    maps = {f"static {p}": S.c.product(p, S.scored)[:, S.GI[0], S.GI[1]] for p in STATIC}
    for cf, r in runs.items():
        maps[f"twin {cf} mean"] = r["mean"]
        maps[f"twin {cf} now"] = r["now"]
    H = {k: S.hourly(v) for k, v in maps.items()}
    n = len(S.events)
    parts = {k: sq_parts(v, S.GH, S.storm_h, n) for k, v in H.items()}

    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)])   # storm weights

    def boot(p):
        sse, tot, cnt = (W @ x for x in p)
        return np.sqrt(sse / cnt) / (tot / cnt)

    B = {k: boot(p) for k, p in parts.items()}
    rows = {}
    for k, h in H.items():
        lo, hi = np.percentile(B[k], [2.5, 97.5])
        # leave one gauge out: the range of the score over the 10 (or 4) choices
        logo = [nrmse(np.delete(h, g, 1), np.delete(S.GH, g, 1)) for g in range(h.shape[1])]
        per_storm = [nrmse(h[S.storm_h == s], S.GH[S.storm_h == s]) for s in range(n)]
        rows[k] = dict(nrmse=nrmse(h, S.GH), ci_lo=lo, ci_hi=hi, logo_min=min(logo), logo_max=max(logo),
                       storm_median=np.nanmedian(per_storm))
    T = pd.DataFrame(rows).T.sort_values("nrmse")

    # paired: is map a better than map b? difference of the bootstrap draws on the same storms
    refs = ["static RCP", "static C", "static CP"]
    pairs = {}
    for k in T.index:
        for b in refs:
            if k == b:
                continue
            d = B[k] - B[b]
            wins = np.mean([nrmse(H[k][S.storm_h == s], S.GH[S.storm_h == s])
                            < nrmse(H[b][S.storm_h == s], S.GH[S.storm_h == s]) for s in range(n)])
            pairs[(k, b)] = dict(diff=T.loc[k, "nrmse"] - T.loc[b, "nrmse"],
                                 d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5),
                                 p_better=np.mean(d < 0), storms_won=wins)
    P = pd.DataFrame(pairs).T
    out = dict(table=T, pairs=P, boot=B, network=network)
    pickle.dump(out, open(CACHE / f"ext_A_{network}.pkl", "wb"))
    pd.set_option("display.width", 160)
    print(f"\n== A · {network}: hourly NRMSE, 95 % bootstrap CI over storms, leave-one-gauge-out range ==")
    print(T.round(2).to_string())
    print("\n== paired differences (negative = row map better); p_better = share of bootstrap draws ==")
    print(P.round(2).to_string())
    return out


# ----------------------------------------------------------------------------
# workers for D and E (one process per configuration, each loads the cube once)
# ----------------------------------------------------------------------------
_S = None


def _init():
    global _S
    _S = Setup("openmrg")


def subset_cube(c, keep):
    """A shallow copy of the cube that only has the links in `keep`."""
    s = copy.copy(c)
    s.links = {k: v[keep] for k, v in c.links.items()}
    s.link_id = c.link_id[keep]
    s._link_rain = c._link_rain[:, keep]
    return s


def _task_D(args):
    frac, draw = args
    S, c = _S, _S.c
    nl = c.link_id.size
    rng = np.random.default_rng(100 + draw)
    keep = np.sort(rng.choice(nl, max(int(round(frac * nl)), 2), replace=False)) if frac < 1 else np.arange(nl)
    sc = subset_cube(c, keep)
    L = sc.links
    gx, gy = c.gauges["x"], c.gauges["y"]
    dist = np.array([segment_distance(gx[g], gy[g], L["xa"], L["ya"], L["xb"], L["yb"]).min()
                     for g in range(gx.size)])
    t0 = time.time()
    mean, now = [], []
    for st in S.steps:
        o = system.run(sc, st, "L")
        mean.append(o["mean"][SPIN:][:, S.GI[0], S.GI[1]]); now.append(o["now"][SPIN:][:, S.GI[0], S.GI[1]])
    Y = sc.link_rain(S.scored)
    idw = np.array([line_idw(y, L["xa"], L["ya"], L["xb"], L["yb"], gx, gy) if np.isfinite(y).any()
                    else np.full(gx.size, np.nan) for y in Y])
    return dict(frac=frac, draw=draw, n_links=keep.size, dist=dist, mean=np.concatenate(mean),
                now=np.concatenate(now), idw=idw, seconds=time.time() - t0)


def _task_E(args):
    name, sensors, kw = args
    S = _S
    GI = S.GI
    t0 = time.time()
    mean, now, ens = [], [], []
    for st in S.steps:
        members = []
        o = system.run(S.c, st, sensors, keep=lambda k, tw: members.append(tw.rain[:, GI[0], GI[1]]), **kw)
        mean.append(o["mean"][SPIN:][:, GI[0], GI[1]]); now.append(o["now"][SPIN:][:, GI[0], GI[1]])
        ens.append(np.array(members, dtype="float32")[SPIN:])          # (T, N, gauges)
    return dict(name=name, sensors=sensors, kw=kw, mean=np.concatenate(mean), now=np.concatenate(now),
                ens=np.concatenate(ens), seconds=time.time() - t0)


def _pool(fn, tasks, out_file):
    out = pickle.load(open(out_file, "rb")) if out_file.exists() else {}
    todo = [t for t in tasks if _key(t) not in out]
    print(f"{len(tasks) - len(todo)} cached, {len(todo)} to run on {min(len(todo), os.cpu_count())} workers")
    if todo:
        with ProcessPoolExecutor(min(len(todo), os.cpu_count()), initializer=_init) as ex:
            for t, r in zip(todo, ex.map(fn, todo)):
                out[_key(t)] = r
                pickle.dump(out, open(out_file, "wb"))                  # save as each one finishes
                print(f"  {_key(t)}: {r['seconds']:.0f}s", flush=True)
    return out


def _key(t):
    return t[0] if isinstance(t[0], str) else t


# ----------------------------------------------------------------------------
# D: link density
# ----------------------------------------------------------------------------
D_TASKS = [(f, d) for f in (0.1, 0.25, 0.5) for d in range(3)] + [(1.0, 0)]


def report_D(S, out):
    rows, per = [], []
    for (f, d), r in sorted(out.items()):
        row = dict(frac=f, draw=d, n_links=r["n_links"])
        for m in ("idw", "mean", "now"):
            row[m] = nrmse(S.hourly(r[m]), S.GH)
            for g in range(S.GH.shape[1]):
                per.append(dict(frac=f, draw=d, map=m, gauge=g, dist=r["dist"][g],
                                nrmse=nrmse(S.hourly(r[m][:, [g]]), S.GH[:, [g]])))
        rows.append(row)
    T = pd.DataFrame(rows)
    print("\n== D · hourly NRMSE at the 10 gauges vs share of links kept (links only) ==")
    print(T.groupby("frac")[["n_links", "idw", "mean", "now"]].agg(["mean", "min", "max"]).round(2).to_string())
    Pg = pd.DataFrame(per)
    bins = pd.cut(Pg.dist, [0, 0.5, 1, 2, 4, 8, 50])
    print("\n== D · per-gauge NRMSE vs distance to the nearest kept link [km] (all draws pooled) ==")
    print(Pg.groupby([bins, "map"]).nrmse.median().unstack().round(2).to_string())
    print(Pg.groupby(bins).size().rename("n").to_string())
    return T, Pg


# ----------------------------------------------------------------------------
# E: spread calibration
# ----------------------------------------------------------------------------
E_TASKS = [("base", "L", {}),
           ("infl 1.10", "L", dict(inflation=1.10)),
           ("infl 1.25", "L", dict(inflation=1.25)),
           ("q_log 0.3", "L", dict(q_log=0.3)),
           ("q_log 0.6", "L", dict(q_log=0.6)),
           ("infl 1.10 + q_log 0.3", "L", dict(inflation=1.10, q_log=0.3)),
           ("LPR base", "LPR", {}),
           ("LPR infl 1.10 + q_log 0.3", "LPR", dict(inflation=1.10, q_log=0.3))]


def crps_ens(ens, y):
    """CRPS of an ensemble (..., N) against y (...): E|X - y| - 0.5 E|X - X'|."""
    a = np.abs(ens - y[..., None]).mean(-1)
    s = np.sort(ens, -1)
    N = ens.shape[-1]
    w = 2 * np.arange(1, N + 1) - N - 1                                 # sum_ij |xi-xj| = 2 sum_i w_i x_(i)
    b = (s * w).sum(-1) / N ** 2
    return a - b


def calib(ens, G):
    """ens (T, N, g), G (T, g). Spread/skill, coverage of the 80 % band, CRPS, rank histogram."""
    E = np.moveaxis(ens, 1, -1)                                          # (T, g, N)
    ok = np.isfinite(G) & np.isfinite(E).all(-1)
    e, y = E[ok], G[ok]
    m = e.mean(-1)
    lo, hi = np.percentile(e, [10, 90], -1)
    rank = (e < y[:, None]).sum(-1)
    return dict(rmse=np.sqrt(np.mean((m - y) ** 2)), spread=np.sqrt(np.mean(e.var(-1, ddof=1))),
                cover80=np.mean((y >= lo) & (y <= hi)), crps=crps_ens(e, y).mean(),
                rank=np.bincount(rank, minlength=e.shape[-1] + 1) / rank.size)


def point_error_model(S, max_km=2.5):
    """How much a point gauge differs from the rain around it, from the gauges alone: two gauges
    max_km apart differ by d, and if each is the local rain plus independent point error,
    var(d) = 2 sigma^2. Fit sigma = a + b * rain (hourly mm/h) by least squares on d^2 / 2."""
    from scipy.optimize import least_squares
    g = S.c.gauges
    D = np.hypot(g["x"][:, None] - g["x"], g["y"][:, None] - g["y"])
    i, j = np.triu_indices(D.shape[0], 1)
    d, m = [], []
    for a, b in zip(i[D[i, j] <= max_km], j[D[i, j] <= max_km]):
        x, y = S.GH[:, a], S.GH[:, b]
        ok = np.isfinite(x) & np.isfinite(y)
        d.append(x[ok] - y[ok]); m.append(0.5 * (x[ok] + y[ok]))
    d, m = np.concatenate(d), np.concatenate(m)
    p = least_squares(lambda p: (p[0] + p[1] * m) ** 2 - d ** 2 / 2, [0.1, 0.5], bounds=([0, 0], [5, 5])).x
    return p


def with_point_error(ens_h, ab, seed=0):
    """Each member plus the point error a gauge would see in that member's rain (clipped at 0)."""
    a, b = ab
    rng = np.random.default_rng(seed)
    return np.clip(ens_h + (a + b * ens_h) * rng.standard_normal(ens_h.shape), 0, None)


def report_E(S, out):
    ab = point_error_model(S)
    print(f"\npoint error of a gauge: sigma = {ab[0]:.2f} + {ab[1]:.2f} * rain  (hourly, from gauge pairs <= 2.5 km)")
    rows = {}
    for name, r in out.items():
        ens_h = S.hourly(r["ens"])                                       # hourly member totals
        c5, ch = calib(r["ens"], S.G), calib(ens_h, S.GH)
        wet = S.GH > 0.5
        cw = calib(np.where(wet[:, None], ens_h, np.nan), np.where(wet, S.GH, np.nan))
        cp = calib(with_point_error(ens_h, ab), S.GH)
        rows[name + " +point err"] = dict(nrmse_mean=nrmse(S.hourly(r["mean"]), S.GH),
                                          nrmse_now=nrmse(S.hourly(r["now"]), S.GH),
                                          ss_hour=cp["spread"] / cp["rmse"], cover80_hour=cp["cover80"],
                                          crps_hour=cp["crps"], rank_outside=cp["rank"][0] + cp["rank"][-1])
        rows[name] = dict(nrmse_mean=nrmse(S.hourly(r["mean"]), S.GH), nrmse_now=nrmse(S.hourly(r["now"]), S.GH),
                          ss_5min=c5["spread"] / c5["rmse"], ss_hour=ch["spread"] / ch["rmse"],
                          ss_hour_wet=cw["spread"] / cw["rmse"], cover80_hour=ch["cover80"],
                          cover80_wet=cw["cover80"], crps_hour=ch["crps"],
                          rank_outside=ch["rank"][0] + ch["rank"][-1])
    T = pd.DataFrame(rows).T
    print("\n== E · spread calibration (links-only twin unless named LPR); ideal: spread/skill 1, "
          "cover80 0.8, rank_outside 2/(N+1) ~ 0.05 ==")
    print(T.round(3).to_string())
    return T


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "report"
    if what == "A":
        nets = sys.argv[2:] or ["openmrg", "openmesh"]
        for net in nets:
            run_A(net)
    elif what == "D":
        out = _pool(_task_D, D_TASKS, CACHE / "ext_D.pkl")
        report_D(Setup("openmrg"), out)
    elif what == "E":
        out = _pool(_task_E, E_TASKS, CACHE / "ext_E.pkl")
        report_E(Setup("openmrg"), out)
    else:
        S = Setup("openmrg")
        for f, rep in (("ext_D.pkl", report_D), ("ext_E.pkl", report_E)):
            if (CACHE / f).exists():
                rep(S, pickle.load(open(CACHE / f, "rb")))
