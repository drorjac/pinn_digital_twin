"""Run the full multi-sensor twin over the wet periods of the training and validation weeks and
cache its maps as teacher labels for Chapter 11: cache/teacher_<network>.npz."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raintwin import system                       # noqa: E402
from raintwin.cube import Cube                    # noqa: E402

network = sys.argv[1] if len(sys.argv) > 1 else "openmrg"
sensors = sys.argv[2] if len(sys.argv) > 2 else "LPR"
c = Cube(network)
wins = system.wet_windows(c)
print(f"{len(wins)} wet windows, {sum(len(w) for w in wins)} steps", flush=True)
idx, now, spread = [], [], []
t0 = time.time()
for j, w in enumerate(wins):
    o = system.run(c, w, sensors)
    idx.append(w); now.append(o["now"].astype("float16")); spread.append(o["spread"].astype("float16"))
    print(f"window {j + 1}/{len(wins)}: {len(w)} steps, {time.time() - t0:.0f}s", flush=True)
out = Path(__file__).resolve().parents[1] / "cache" / f"teacher_{network}.npz"
np.savez_compressed(out, idx=np.concatenate(idx), now=np.concatenate(now), spread=np.concatenate(spread),
                    window=np.concatenate([np.full(len(w), j) for j, w in enumerate(wins)]), sensors=sensors)
print("saved", out)
