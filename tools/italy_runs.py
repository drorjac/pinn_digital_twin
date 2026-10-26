"""Run the twin on FieldSense's 10 Emilia-Romagna storms for Chapter 13, one job per
(sensor mix, storm, held-out gauge fold), in parallel: cache/ch13/<mix>_<storm>_f<fold>.npz.

Mixes that use the gauges (P) run once per fold, without that fold; the others run once
(fold "all"). E-... is the radar-error twin. A "+" mix has the two fixes of Chapter 13,
section 5: gauges observed in log space, and a random walk on the radar's bias. R-4km is
the radar-only twin with the radar observed every 4 km, as in Gothenburg. Finished
jobs are skipped, so a stopped run resumes. Usage: python tools/italy_runs.py [n_processes]
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")    # threaded BLAS is far slower on these small solves
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys                                            # noqa: E402
import time                                           # noqa: E402
import warnings                                       # noqa: E402
from multiprocessing import Pool                      # noqa: E402
from pathlib import Path                              # noqa: E402

import numpy as np                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raintwin import openrainer, system               # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "cache" / "ch13"
MIXES = ["L", "R", "LR", "P", "LP", "LPR", "E-L", "E-LP", "P+", "LPR+", "R-4km"]
FIXES = dict(point_log=True, q_bias=0.02)
RADAR_THIN = 3                     # radar observed every 6 km: the grid has 6x Gothenburg's cells
FIG_EVENT = "20220817T10"          # full maps are kept for this storm, fold 0


def jobs():
    for ev, _, _ in openrainer.events():
        for mix in MIXES:
            for fold in (range(openrainer.FOLDS) if "P" in mix else [None]):
                yield mix, ev, fold


def path(mix, ev, fold):
    return OUT / f"{mix}_{ev}_f{'all' if fold is None else fold}.npz"


def run_job(job):
    mix, ev, fold = job
    f = path(*job)
    if f.exists():
        return job, 0.0
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    s = openrainer.Storm(ev, hold_out=fold)
    idx = np.arange(len(s.times))
    bias = []
    if mix.startswith("E-"):
        o = system.run_error_twin(s, idx, mix[2:])
    else:
        base = mix.rstrip("+").replace("-4km", "")
        keep = (lambda k, tw: bias.append(np.exp(tw.B["R"]).mean())) if "R" in base and len(base) > 1 else None
        fixes = FIXES if mix.endswith("+") else {}
        thin = 2 if mix.endswith("-4km") else RADAR_THIN          # the cost of the thinning, radar only
        o = system.run(s, idx, base, radar_thin=thin, keep=keep, **fixes)
    gi = s.cells(s.gauges["x"], s.gauges["y"])
    out = dict(mean=o["mean"][:, gi[0], gi[1]], now=o["now"][:, gi[0], gi[1]],
               spread=o["spread"][:, gi[0], gi[1]], V=o["V"], station=s.gauges["id"],
               time=s.times.values, radar_factor=np.array(bias))
    if ev == FIG_EVENT and fold in (None, 0):
        out.update({f"map_{k}": o[k].astype("float32") for k in ("mean", "now", "spread", "factor") if k in o})
    np.savez_compressed(f, **out)
    return job, time.time() - t0


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs() if not path(*j).exists()]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"{len(todo)} jobs on {n} processes", flush=True)
    t0 = time.time()
    with Pool(n) as pool:
        for i, (job, sec) in enumerate(pool.imap_unordered(run_job, todo), 1):
            print(f"{i}/{len(todo)} {job[0]} {job[1]} fold {job[2]}: {sec:.0f}s "
                  f"(total {time.time() - t0:.0f}s)", flush=True)
