"""Rerun the twins of Chapters 10 (Gothenburg) and 12 (New York) with the two fixes found in
Chapter 13, one job per (network, mix, storm), in parallel: cache/fix/<network>_<mix>_<storm>.npz.

The radar's log-bias had no process noise, so it froze after the first update. Mixes:
  LPR      the chapter's twin, rerun only to record its radar factor
  LPR+b    LPR with a random walk on the radar's bias (q_bias=0.02 per 5 min)
  LPR+     LPR with that and the stations observed in log space (point_log)
  LR+      links and radar with the random walk
  P+       stations alone, in log space
Each job keeps the twin mean and now map at the scoring gauges (after the one-hour spin-up)
and the radar factor exp(bias) at every step. Finished jobs are skipped, so a stopped run
resumes. Usage: python tools/bias_fix_runs.py [n_processes]
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
import pandas as pd                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raintwin import system                           # noqa: E402
from raintwin.cube import Cube                        # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "cache" / "fix"
NETWORKS = ["openmrg", "openmesh"]
MIXES = {"LPR": {}, "LPR+b": dict(q_bias=0.02), "LPR+": dict(q_bias=0.02, point_log=True),
         "LR+": dict(q_bias=0.02), "P+": dict(point_log=True)}
SPIN = 12                                             # one hour of spin-up, as in the chapters
_CUBES = {}


def cube(network):
    if network not in _CUBES:
        _CUBES[network] = Cube(network)
    return _CUBES[network]


def jobs():
    for net in NETWORKS:
        for name, _, _ in cube(net).events():
            for mix in MIXES:
                yield net, mix, name


def path(net, mix, ev):
    return OUT / f"{net}_{mix}_{ev}.npz"


def run_job(job):
    net, mix, ev = job
    f = path(*job)
    if f.exists():
        return job, 0.0
    warnings.filterwarnings("ignore")
    t0 = time.time()
    c = cube(net)
    (_, s, e), = [x for x in c.events() if x[0] == ev]
    st = c.index(s - pd.Timedelta("1h"), e)
    sensors = mix.split("+")[0]
    bias = []
    keep = (lambda k, tw: bias.append(np.exp(tw.B["R"]).mean())) if "R" in sensors and len(sensors) > 1 else None
    o = system.run(c, st, sensors, keep=keep, **MIXES[mix])
    gi = np.round(c.gauges["y"] / c.dx).astype(int), np.round(c.gauges["x"] / c.dx).astype(int)
    np.savez_compressed(f, mean=o["mean"][SPIN:][:, gi[0], gi[1]], now=o["now"][SPIN:][:, gi[0], gi[1]],
                        steps=st, radar_factor=np.array(bias))
    return job, time.time() - t0


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs() if not path(*j).exists()]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"{len(todo)} jobs on {n} processes", flush=True)
    t0 = time.time()
    with Pool(n) as pool:
        for i, (job, sec) in enumerate(pool.imap_unordered(run_job, todo), 1):
            print(f"{i}/{len(todo)} {' '.join(job)}: {sec:.0f}s (total {time.time() - t0:.0f}s)", flush=True)
