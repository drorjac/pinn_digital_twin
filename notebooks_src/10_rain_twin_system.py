# %% [markdown]
# # Chapter 10 · Project: one twin, three kinds of sensor (Gothenburg, summer 2015)
# 2D · first run about 40 min, then cached · *Rain twin, part 4*
#
# Chapters 8 and 9 used links only, on 8 days. Here the twin gets the whole summer of 2015 and
# every sensor Gothenburg has:
#
# | letter | sensor | what it sees | count |
# |---|---|---|---|
# | **L** | commercial microwave links | path-averaged rain, from FieldSense's RNN retrieval | 358 |
# | **P** | personal weather stations (Netatmo), after quality control | rain in a bucket | 27 cells |
# | **R** | SMHI C-band radar | rain rate on 2 km cells, every 5 min | 1540 cells |
# | | 10 city gauges | **never assimilated**: they only score | 10 |
#
# All of it comes from the 5-minute data cube that FieldSense's `multisensor_nowcasting` project
# builds, read by `raintwin.cube.Cube`. That project's sister study (`multisensor_maps`) already
# answered "what is the best *static* map from these sensors": the radar adjusted by the
# residuals at links and stations, product **RCP**. It is the benchmark here, scored on the
# same 10 storms (FieldSense's study storms, all in the test weeks) with the same rule: hourly
# NRMSE at the 10 city gauges, RMSE divided by the mean gauge rain.
#
# **Questions**
# 1. Which sensor mix gives the twin its best map, and does the twin beat the static merge?
# 2. What goes wrong when a smooth twin meets a sharp radar, and how should the radar enter?
# 3. What does a twin give that a static map does not?

# %%
import sys, time, pickle; sys.path.insert(0, "..")
from pathlib import Path
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtwin import viz
from raintwin import system
from raintwin.cube import Cube
viz.set_style()

CACHE = Path("../cache"); CACHE.mkdir(exist_ok=True)
c = Cube("openmrg")
EVENTS = c.events()
GI = np.round(c.gauges["y"] / c.dx).astype(int), np.round(c.gauges["x"] / c.dx).astype(int)
print(f"{len(c.times)} steps of 5 min, {c.y.size} x {c.x.size} cells of {c.dx:.0f} km")
print(f"{len(c.link_id)} links, {len(c.pws['x'])} station cells, {len(c.gauges['id'])} city gauges")
print("split of the summer (train / validation / test steps):", np.bincount(c.split))
for name, s, e in EVENTS:
    print(f"  {name}: {s:%d %b %H:%M} to {e:%d %b %H:%M}, split {set(c.split[c.index(s, e)])}")

# %% [markdown]
# ## 1. The network

# %%
tot = sum(np.nansum(c.radar(c.index(s, e)), 0) for _, s, e in EVENTS) / 12
ext = (-1, c.x[-1] + 1, -1, c.y[-1] + 1)
fig, ax = plt.subplots(figsize=(7.5, 8))
ax.grid(False)
im = ax.imshow(tot, origin="lower", cmap=viz.FIELD, extent=ext)
plt.colorbar(im, ax=ax, shrink=0.7, label="radar rain over the 10 storms [mm]")
L = c.links
for i in range(L["xa"].size):
    ax.plot([L["xa"][i], L["xb"][i]], [L["ya"][i], L["yb"][i]], color=viz.INK, lw=0.6)
ax.plot(c.pws["x"], c.pws["y"], "s", color=viz.ORANGE, ms=6, mec="white", label="weather stations (P)")
ax.plot(c.gauges["x"], c.gauges["y"], "^", color=viz.BLUE, ms=9, mec="white", label="city gauges (score only)")
ax.plot([], [], color=viz.INK, lw=1, label="links (L)")
ax.set(xlabel="x [km]", ylabel="y [km]", title="Gothenburg: links, stations, gauges, radar")
ax.legend(loc="lower right", fontsize=8)
viz.save(fig, "10_network.png")
plt.show()

# %% [markdown]
# The links and gauges sit in the city; the stations are spread a bit wider; the radar covers
# everything. Every gauge is within a few km of a link, so near-link accuracy is what the score
# rewards.
#
# ## 2. Scoring rule and the static benchmark
# Each map is read at the gauge's cell, summed to hourly rain, and compared with the gauge.
# FieldSense's cube stores its static maps for every step, so they can be scored first:
# **R** radar, **C** link IDW, **P** station IDW, **CP** both, and the radar adjusted by the
# misses at links (**RC**), stations (**RP**) or both (**RCP**).

# %%
SPIN = 12                                                     # one hour of spin-up before a storm
steps = [c.index(s - pd.Timedelta("1h"), e) for _, s, e in EVENTS]
scored = np.concatenate([st[SPIN:] for st in steps])
G = c.gauge_rain(scored)
HOURS = c.times[scored].ceil("h")


def hourly(a):
    return pd.DataFrame(a / 12, index=HOURS).groupby(level=0).sum(min_count=12).values


GH = hourly(G)


def score(E):
    """Hourly NRMSE, hourly correlation and relative bias at the gauges; 5-min correlation."""
    Eh = hourly(E)
    ok, ok5 = np.isfinite(Eh) & np.isfinite(GH), np.isfinite(E) & np.isfinite(G)
    return dict(nrmse=np.sqrt(np.mean((Eh[ok] - GH[ok]) ** 2)) / GH[ok].mean(),
                corr=np.corrcoef(Eh[ok], GH[ok])[0, 1], bias=Eh[ok].mean() / GH[ok].mean() - 1,
                corr5=np.corrcoef(E[ok5], G[ok5])[0, 1])


table = {}
for p in ("R", "C", "P", "CP", "RC", "RP", "RCP"):
    table["static " + p] = score(c.product(p, scored)[:, GI[0], GI[1]])
print(pd.DataFrame(table).T.round(2))

# %% [markdown]
# The radar alone is the worst map at the gauges (NRMSE 1.47; FieldSense's report has the same
# number). Links alone are much better (1.07). The radar adjusted with links and stations is
# the best, **0.99**. That is the bar.
#
# ## 3. The twin with every sensor mix
# `raintwin.system.run(cube, steps, "LPR")` builds one `RainTwin` on the cube grid and adds an
# observation set for each letter:
#
# * **L**: path averages of rain, error 0.3 mm/h + 15 %.
# * **P**: point readings in the station's cell, error 0.5 mm/h + 30 % (buckets under trees,
#   on balconies).
# * **R**: every other radar cell, observed **in log space**: the radar reading becomes
#   log(R + 1) and is compared with the twin's own log state. That observation is linear in the
#   state, so the Kalman update is exact for it. When the radar is not the only sensor, its
#   bias is a state parameter too.
#
# Three settings changed from Chapter 8, each for a reason found on these storms:
# * The log offset r0 is 1 mm/h instead of 0.1. With 0.1, members near zero rain have almost no
#   spread in rain units, so a link reading 5 mm/h barely moves them, and a storm that starts dry
#   is never picked up (NRMSE 2.97 for links only).
# * The speed is capped at 60 km/h. Without the cap the velocity, updated by hundreds of
#   observations without localization, sometimes ran off.
# * Radar in rain space (error 1 mm/h + 50 %) left the radar-only twin 40 % low and made one storm
#   overshoot; the log-space observation fixes the overshoot.
#
# Every run also makes the **now map** of Chapter 9: the twin mean plus a 3 km IDW of what it
# misses at the links and stations.
#
# The ten storms times six mixes take about 35 minutes, so the results are cached.

# %%
CONFIGS = ["L", "P", "R", "LR", "LPR"]
FIG_EVENT = "20150728T03"


def run_all():
    out = {}
    for cf in CONFIGS + ["E:LP"]:
        t0 = time.time()
        g_mean, g_now, vel, maps = [], [], [], None
        for (name, _, _), st in zip(EVENTS, steps):
            o = system.run_error_twin(c, st, "LP") if cf == "E:LP" else system.run(c, st, cf)
            g_mean.append(o["mean"][SPIN:][:, GI[0], GI[1]]); g_now.append(o["now"][SPIN:][:, GI[0], GI[1]])
            vel.append(o["V"])
            if name == FIG_EVENT and cf in ("LPR", "E:LP"):
                maps = {k: o[k].astype("float32") for k in o if k != "V"}
        out[cf] = dict(mean=np.concatenate(g_mean), now=np.concatenate(g_now), V=vel, maps=maps,
                       seconds=time.time() - t0)
        print(f"{cf:5s}: {time.time() - t0:5.0f}s")
    return out


f = CACHE / "ch10_runs.pkl"
if f.exists():
    runs = pickle.load(open(f, "rb"))
else:
    runs = run_all()
    pickle.dump(runs, open(f, "wb"))
for cf, r in runs.items():
    table[f"twin {cf} (mean)"] = score(r["mean"])
    table[f"twin {cf} (now map)"] = score(r["now"])
T = pd.DataFrame(table).T
print(T.round(2).to_string())

# %%
order = T.sort_values("nrmse")
fig, ax = plt.subplots(figsize=(9, 7))
cols = [viz.MUTED if k.startswith("static") else (viz.TWIN if "now" in k else viz.AQUA) for k in order.index]
ax.barh(range(len(order)), order.nrmse, color=cols, height=0.7)
ax.set_yticks(range(len(order)), order.index, fontsize=8)
ax.invert_yaxis()
ax.axvline(T.loc["static RCP", "nrmse"], color=viz.INK, lw=1, ls="--")
for i, v in enumerate(order.nrmse):
    ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=7, color=viz.INK2)
ax.set(xlabel="hourly NRMSE at the 10 city gauges (lower is better)",
       title="Static maps (grey), twin mean (aqua), twin now map (blue)")
viz.save(fig, "10_scores.png")
plt.show()

# %% [markdown]
# **Reading the table.**
# * The best twin maps are the now maps with links: about **1.05**. That beats every static map
#   that does not use the radar (link IDW 1.07, links + stations 1.06), and loses to the static
#   radar merges (RC 1.02, RCP 0.99).
# * The twin mean on its own is worse (1.15 with links). It is smooth: 40 members with 6 km noise
#   cannot hold a 2 km convective cell. The now map's local correction puts the cells back.
# * Stations are weak in this twin. Alone, the twin mean is 60 % low at the gauges, because
#   between the 27 station cells it falls back to the background. Added to links they do not
#   help.
# * The radar helps the twin little. Section 4 shows why.
#
# ### Storm by storm: where does the merge win?
# One score over ten storms hides how the storms differ. Below, the score of each storm on its
# own, for link IDW, the static merge RCP and the links-only twin, plus each storm's share of
# the gap between the twin and RCP in squared error (the sum that NRMSE is made of).

# %%
STORM_H = pd.Series(np.concatenate([np.full(st.size - SPIN, i) for i, st in enumerate(steps)]),
                    index=HOURS).groupby(level=0).first().values
per_storm = {"static C": c.product("C", scored)[:, GI[0], GI[1]],
             "static RCP": c.product("RCP", scored)[:, GI[0], GI[1]],
             "twin L (now map)": runs["L"]["now"]}
per_storm = {k: hourly(v) for k, v in per_storm.items()}


def storm_nrmse(Eh, s):
    m = STORM_H == s
    ok = np.isfinite(Eh[m]) & np.isfinite(GH[m])
    return np.sqrt(np.mean((Eh[m][ok] - GH[m][ok]) ** 2)) / GH[m][ok].mean()


def storm_sse(Eh, s):
    m = STORM_H == s
    return np.nansum((Eh[m] - GH[m]) ** 2)


PS = pd.DataFrame({k: [storm_nrmse(v, s) for s in range(len(EVENTS))] for k, v in per_storm.items()},
                  index=[n for n, _, _ in EVENTS])
PS.insert(0, "hours", [int((STORM_H == s).sum()) for s in range(len(EVENTS))])
gap = np.array([storm_sse(per_storm["twin L (now map)"], s) - storm_sse(per_storm["static RCP"], s)
                for s in range(len(EVENTS))])
PS["share of gap"] = gap / gap.sum()
print(PS.round(2).to_string())
print(f"\nRCP better than the twin in {(PS['static RCP'] < PS['twin L (now map)']).sum()} of {len(PS)} storms; "
      f"the two largest shares of the gap add up to {np.sort(PS['share of gap'])[-2:].sum():.0%}")

# %% [markdown]
# * RCP is better than the twin in most storms, but in most of them only by a hair.
# * Nearly 80 % of RCP's lead comes from **two storms**, 2 June and 28 July (about 30 hours
#   each). They are neither the most intense (the end of August has higher rates) nor the
#   longest (7 July lasts 41 hours and adds almost nothing to the gap). What sets them apart is
#   an open question (a map of where RCP and the twin differ most in those two storms would be the next check).
# * So "the merge beats the twin" is mostly a statement about those two storms. With ten storms,
#   one or two of them can decide the table; the bootstrap in `tools/extend_runs.py A` puts a
#   95 % range of about 0.75 to 1.3 on every map's total score, while the *paired* difference
#   twin minus RCP stays between about 0.0 and 0.11.
#
# ## 4. A smooth twin and a sharp radar
# The radar has the structure the twin lacks, but the twin can only hold a smooth version of it.
# On the heaviest hour of 28 July:

# %%
r = runs["LPR"]["maps"]
fe = [i for i, (n, _, _) in enumerate(EVENTS) if n == FIG_EVENT][0]
st = steps[fe]
k = int(np.nanargmax(np.nanmean(c.radar(st), (1, 2))))
fields = [(np.nan_to_num(c.radar(st[[k]])[0]), "Radar", viz.FIELD),
          (c.product("RCP", st[[k]])[0], "Static merge RCP", viz.FIELD),
          (r["mean"][k], "Twin LPR: mean", viz.FIELD),
          (r["now"][k], "Twin LPR: now map", viz.FIELD),
          (r["spread"][k], "Twin LPR: spread", viz.UNCERTAINTY)]
vm = np.nanpercentile(fields[0][0], 99.5)
fig, axes = plt.subplots(1, 5, figsize=(19, 4.8), constrained_layout=True)
for ax, (F, ttl, cm) in zip(axes, fields):
    ax.grid(False)
    im = ax.imshow(F, origin="lower", cmap=cm, vmin=0, vmax=vm if cm is viz.FIELD else None, extent=ext)
    ax.plot(c.gauges["x"], c.gauges["y"], "^", color=viz.BLUE, ms=5, mec="white")
    ax.set(title=ttl, xticks=[], yticks=[])
    plt.colorbar(im, ax=ax, shrink=0.75, label="mm/h")
fig.suptitle(f"{c.times[st[k]]:%d %b %Y %H:%M}", x=0.01, ha="left", fontweight="bold")
viz.save(fig, "10_maps.png")
plt.show()
print(f"peak rain: radar {np.nanmax(fields[0][0]):.1f}, RCP {np.nanmax(fields[1][0]):.1f}, "
      f"twin mean {np.nanmax(fields[2][0]):.1f}, now map {np.nanmax(fields[3][0]):.1f} mm/h")

# %% [markdown]
# The twin keeps the shape of the storm and loses more than half of the peak; the now map gets
# some of it back near the links. That is the price of a
# filter built on a 40-member ensemble with a smooth noise model: it can only represent what its
# ensemble spans.
#
# **A different design: twin the radar's error, not the rain.** `RadarErrorTwin` keeps the radar
# as the detailed picture and makes its *correction* the state:
#
# $$R(x,t) = \big(R_\text{radar}(x,t) + 0.5\big)\,e^{z(x,t)} - 0.5$$
#
# The state z is a smooth log-factor, moved with the storm, relaxing back to zero over 3 hours,
# and corrected by the links and stations. This is the static merge made dynamic: the correction
# found an hour ago is carried along instead of forgotten. In the table it is `twin E:LP`.

# %%
for key in ("twin E:LP (mean)", "twin E:LP (now map)", "twin LPR (now map)", "static RCP", "static RC"):
    print(f"{key:22s} NRMSE {T.loc[key, 'nrmse']:.2f}   corr {T.loc[key, 'corr']:.2f}   "
          f"bias {T.loc[key, 'bias']:+.0%}")
fac = runs["E:LP"]["maps"]["factor"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
for ax, (F, ttl) in zip(axes, [(runs["E:LP"]["maps"]["mean"][k], "Radar-error twin: rain"),
                               (fac[k], "Its correction factor exp(z)")]):
    ax.grid(False)
    im = ax.imshow(F, origin="lower", cmap=viz.FIELD if "rain" in ttl else viz.DIVERGING,
                   vmin=0 if "rain" in ttl else 0.25, vmax=vm if "rain" in ttl else 4.0, extent=ext)
    if "factor" in ttl:
        im.set_norm(plt.matplotlib.colors.LogNorm(0.25, 4.0))
    ax.set(title=ttl, xticks=[], yticks=[])
    plt.colorbar(im, ax=ax, shrink=0.8)
viz.save(fig, "10_error_twin.png")
plt.show()

# %% [markdown]
# The radar-error twin keeps the radar's peaks, and its now map scores about like the rain twin's,
# still behind the static merge. On these storms the extra machinery buys no accuracy *at the
# gauges* over FieldSense's static RCP.
#
# ## 5. What the twin adds that a static map cannot
# Accuracy at 10 gauges is only one column. Two others:
#
# **Motion.** Every twin learns the storm velocity from its sensors. Compare with the motion of the
# radar itself (phase correlation of frames 30 minutes apart, in 4 km/h steps):

# %%
def radar_motion(F, lag=6):
    F = np.nan_to_num(F); F = F - F.mean((1, 2), keepdims=True)
    acc = 0
    for a, b in zip(F[:-lag], F[lag:]):
        A, B = np.fft.fft2(a), np.fft.fft2(b)
        acc = acc + B * np.conj(A)
    cc = np.real(np.fft.ifft2(acc / (np.abs(acc) + 1e-9)))
    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    ny, nx = cc.shape
    return np.array([ix if ix <= nx // 2 else ix - nx, iy if iy <= ny // 2 else iy - ny]) * c.dx / (lag * 5 / 60)


print(f"{'storm':12s} {'radar':>12s} {'twin L':>12s} {'twin LPR':>12s}")
mot = []
for j, ((name, _, _), st) in enumerate(zip(EVENTS, steps)):
    vr = radar_motion(c.radar(st))
    vl = runs["L"]["V"][j][len(st) // 2:].mean(0)
    va = runs["LPR"]["V"][j][len(st) // 2:].mean(0)
    mot.append((vr, vl, va))
    print(f"{name:12s} {str(vr.round(0)):>12s} {str(vl.round(0)):>12s} {str(va.round(0)):>12s}")
mot = np.array(mot)
moving = np.hypot(*mot[:, 0].T) > 0                 # storms where the radar shows a shift
print(f"{moving.sum()} storms with a measurable radar motion:")
for j, lab in ((1, "twin L"), (2, "twin LPR")):
    m_ = mot[moving]
    ang = np.degrees(np.arctan2(m_[:, j, 1], m_[:, j, 0]) - np.arctan2(m_[:, 0, 1], m_[:, 0, 0]))
    ang = (ang + 180) % 360 - 180
    print(f"  {lab}: median direction error {np.median(np.abs(ang)):.0f} degrees, "
          f"median speed ratio {np.median(np.hypot(*m_[:, j].T) / np.hypot(*m_[:, 0].T)):.2f}")

# %% [markdown]
# With the radar in the state the twin's motion is closer to the radar's, in direction and speed.
# With links only the direction is often far off and the speed too low: a network 30 km across
# sees a storm for only an hour, and the filter is slow to move its velocity. Both twins are
# slower than the radar's estimate, which itself comes in coarse 4 km/h steps.
#
# **Uncertainty.** The twin gives a spread with every map. Is it honest? If it were, about 95 %
# of gauge readings would fall within the twin mean ± 2 spreads.

# %%
sp = np.concatenate([runs["LPR"]["maps"]["spread"][SPIN:][:, GI[0], GI[1]]])
mn = runs["LPR"]["maps"]["mean"][SPIN:][:, GI[0], GI[1]]
g = c.gauge_rain(steps[fe][SPIN:])
ok = np.isfinite(g) & (g > 0.1)
print(f"28 July, wet 5-min steps: {np.mean(np.abs(g[ok] - mn[ok]) <= 2 * sp[ok]):.0%} of gauge readings "
      f"inside mean ± 2 spread (95 % if the spread were right)")

# %% [markdown]
# The spread is far too small. An ensemble of 40 members spans only the errors its noise model
# allows, and that model was set for the sensors, not for a gauge 1 km from the nearest link,
# where sub-cell rain variability dominates. It is still useful as a *relative* map (it is large where no sensor
# looks and during heavy rain), and Chapter 11 shows it can be learned too.
#
# %% [markdown]
# ## 6. The radar's bias, unfrozen
# Chapter 13 found a bug in this twin. The radar's log-bias b is part of the state, but nothing
# makes it move between updates: it has no process noise. At the first update hundreds of radar
# cells collapse its ensemble spread, and from then on it stays where it landed.
# `tools/bias_fix_runs.py` reruns the ten storms with the two fixes of Chapter 13: a random walk
# on the bias (`q_bias=0.02` per 5 min, like the one on the velocity), and the stations observed
# in log space (`point_log`). A `+b` mix has only the first fix, a `+` mix has both (P has no
# radar, so P+ has only the second). The cell checks first that the rerun of the plain LPR twin is
# identical to the run in section 3.

# %%
NET, FIG_PREFIX = "openmrg", "10"

# %%
FIX = CACHE / "fix"
FIX_MIXES = ["LPR", "LPR+b", "LPR+", "LR+", "P+"]
assert all((FIX / f"{NET}_{m}_{n}.npz").exists() for m in FIX_MIXES for n, _, _ in EVENTS), \
    "run python tools/bias_fix_runs.py first"
fix = {m: [np.load(FIX / f"{NET}_{m}_{n}.npz") for n, _, _ in EVENTS] for m in FIX_MIXES}
assert np.allclose(np.concatenate([d["now"] for d in fix["LPR"]]), runs["LPR"]["now"], equal_nan=True)
fix_table = {}
for base, fixed in [("LPR", ["LPR+b", "LPR+"]), ("LR", ["LR+"]), ("P", ["P+"])]:
    for m in [base] + fixed:
        for k in ("mean", "now"):
            g = runs[base][k] if m == base else np.concatenate([d[k] for d in fix[m]])
            fix_table[f"twin {m} ({k} map)"] = score(g)
print(pd.DataFrame(fix_table).T.round(3).to_string())
ratio = [np.nansum(c.product("R", st[SPIN:])[:, GI[0], GI[1]]) / np.nansum(c.gauge_rain(st[SPIN:])) for st in steps]
print("radar / gauges per storm:", " ".join(f"{r:.2f}" for r in ratio))
for m in ("LPR", "LPR+b"):
    f = [d["radar_factor"] for d in fix[m]]
    print(f"radar factor, {m:5s}: range within storms {min(x.min() for x in f):.2f} to {max(x.max() for x in f):.2f}, "
          f"at the end {np.median([x[-1] for x in f]):.2f} (median)")

fig, ax = plt.subplots(figsize=(9, 4))
for j, m in enumerate(("LPR", "LPR+b")):
    for i, d in enumerate(fix[m]):
        h = np.arange(d["radar_factor"].size) / 12 - 1
        ax.plot(h, d["radar_factor"], color=(viz.MUTED, viz.TWIN)[j], lw=(1.6, 1.0)[j],
                label=(None if i else ("frozen (the chapter's twin)", "random walk, q_bias = 0.02")[j]))
ax.axhline(1, color=viz.INK, lw=0.8, ls="--")
ax.set(xlabel="hours since the storm started", ylabel="learned radar factor exp(b)",
       title="The radar's learned bias in each storm: frozen, and with a random walk")
ax.legend(fontsize=8)
viz.save(fig, f"{FIG_PREFIX}_bias.png")
plt.show()

# %% [markdown]
# **Reading the rerun.**
# * The bug is real. Frozen, the radar factor is 0.87 to 0.93 in every storm, set by the first
#   update. With the random walk it moves, between 0.5 and 1.8 within a storm. It does not follow
#   the radar's actual error per storm, though (radar / gauges 0.59 to 1.70): the links set the
#   level, and the bias only takes up what is left over.
# * The twin **mean** gets better (1.26 with the frozen bias, 1.17 with the random walk, 1.14 with
#   the stations in log space as well). The **now map** does not change (1.06 in all three), and
#   neither does LR (1.05). The now map's local correction at the links already absorbs a wrong
#   level, so the bias barely reaches the gauges.
# * Stations in log space alone help a little (P now map 1.32 to 1.31).
#
# So the numbers of this chapter stand. The frozen bias acted as a fixed radar calibration of
# about 0.9, and that was harmless here, because 358 links pin the rain level. Chapter 12 shows
# that this does not hold everywhere.
#
# ## Conclusions
# | | result on the 10 storms |
# |---|---|
# | best static map (FieldSense RCP) | NRMSE 0.99 |
# | best twin map (now map, links in the state) | about 1.05, better than any static map without radar |
# | twin mean alone | 1.15 or worse: too smooth for convective cells |
# | radar in the twin | in log space, or as the base of a radar-error twin; neither beats RCP at the gauges |
# | motion | learned; closer to the radar's when the radar is in the state |
# | spread | far too narrow in absolute terms (15 % coverage where 95 % is expected, 28 July) |
# | radar bias | frozen at its first estimate (about 0.9); unfreezing it improves the mean (1.26 to 1.14), not the now map (1.06) |
#
# So the twin is not the best *map maker* when a good radar merge is available. It is a
# **system**: one state that takes any sensor, holds motion and uncertainty, and runs through
# time. That makes it a good *teacher*. Chapter 11 distils it into a network that needs no radar
# and makes a map in two milliseconds.
