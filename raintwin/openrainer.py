"""Emilia-Romagna (OpenRainER), one storm at a time, from FieldSense's merging inputs.

Reads `<FIELDSENSE_DIR>/data/outputs/_multisensor_maps/merging/inputs/openrainer_<event>/`,
which FieldSense's `projects/maps/multisensor` writes for its 10 study storms:

* links: RNN link rain (FieldSense `cml_rnn`), hourly, per link (about 130 per storm)
* gauges: the 319 regional gauges (ARPAE), hourly
* radar: the regional radar composite, hourly, on a 0.02 degree grid

Everything here is hourly, so a storm has `step_h = 1`. The radar is resampled (bilinear) to
a square 2 km grid in km from the south-west corner (cell (i, j) has its centre at (2j, 2i)).

The gauges are split into FieldSense's 5 folds (md5 of the station name, as in
`multisensor_maps.merging.fold_of`). `hold_out=k` keeps fold k out of the twin: those gauges
are only used for scoring, the other four folds are the twin's point sensors (the set "P"
of `raintwin.system`). FieldSense's scores of its own maps, on the same held-out folds, come
with `fieldsense_scores()`.
"""

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

FIELDSENSE = Path(os.environ.get("FIELDSENSE_DIR", Path.home() / "PycharmProjects/FieldSense"))
MERGING = FIELDSENSE / "data/outputs/_multisensor_maps/merging"
EVENTS_CSV = FIELDSENSE / "projects/maps/multisensor/results/events.csv"
FOLDS = 5
LAT0, LON0 = 43.75, 9.25                 # south-west corner of the radar grid
KM_LAT = 110.57
KM_LON = 111.32 * np.cos(np.deg2rad(44.37))


def fold_of(station, folds=FOLDS):
    return int(hashlib.md5(str(station).encode()).hexdigest(), 16) % folds


def to_xy(lat, lon):
    """lat/lon to km east and north of the grid's south-west corner."""
    return (np.asarray(lon) - LON0) * KM_LON, (np.asarray(lat) - LAT0) * KM_LAT


def events():
    """FieldSense's 10 study storms for Emilia-Romagna: list of (name, start, end)."""
    df = pd.read_csv(EVENTS_CSV)
    df = df[df.network == "openrainer"]
    return [(r.event_id, pd.Timestamp(r.start), pd.Timestamp(r.end)) for r in df.itertuples()]


class Storm:
    """One storm on a 2 km grid, with the interface of `raintwin.cube.Cube` that
    `raintwin.system` uses (x, y, links, pws, link_rain, pws_rain, radar)."""

    step_h = 1.0

    def __init__(self, event, hold_out=None, dx=2.0):
        d = MERGING / "inputs" / f"openrainer_{event}"
        self.event = event
        meta = json.loads((d / "meta.json").read_text())
        lk = xr.open_dataset(d / "links_rnn.nc").load()
        g = xr.open_dataset(d / "points_gauges.nc").load()
        r = xr.open_dataset(d / "radar_0.nc").load()
        self.times = pd.DatetimeIndex(r.time.values)
        self.start, self.end = pd.Timestamp(meta["start"]), pd.Timestamp(meta["end"])

        xmax, ymax = to_xy(r.lat.values.max(), r.lon.values.max())
        self.dx = dx
        self.x = dx * np.arange(int(np.floor(xmax / dx)) + 1)
        self.y = dx * np.arange(int(np.floor(ymax / dx)) + 1)
        lon = LON0 + self.x / KM_LON
        lat = LAT0 + self.y / KM_LAT
        self._radar = (r.radar.interp(lat=xr.DataArray(lat, dims="y"), lon=xr.DataArray(lon, dims="x"))
                       .transpose("time", "y", "x").values.astype("float32"))

        xa, ya = to_xy(lk.site_0_lat.values, lk.site_0_lon.values)
        xb, yb = to_xy(lk.site_1_lat.values, lk.site_1_lon.values)
        self.links = dict(xa=xa, ya=ya, xb=xb, yb=yb)
        self.link_id = lk.link.values
        self._link_rain = lk.rain.reindex(time=self.times).transpose("time", "link").values.astype("float32")

        st = g.station.values.astype(str)
        gx, gy = to_xy(g.lat.values, g.lon.values)
        inside = (gx >= 0) & (gx <= self.x[-1]) & (gy >= 0) & (gy <= self.y[-1])
        self.fold = np.array([fold_of(s) for s in st])
        self.gauges = dict(x=gx, y=gy, id=st, inside=inside, fold=self.fold)
        self._gauge = g.rain.reindex(time=self.times).transpose("time", "station").values
        self.hold_out = hold_out
        use = inside & (self.fold != hold_out)
        self._use = np.flatnonzero(use)
        self.pws = dict(x=gx[use], y=gy[use])                         # the twin's point sensors

    # -- data at steps (indices into self.times) --------------------------------------
    def link_rain(self, idx):
        return self._link_rain[idx]

    def pws_rain(self, idx):
        return self._gauge[idx][:, self._use]

    def gauge_rain(self, idx):
        return self._gauge[idx]

    def radar(self, idx):
        return self._radar[idx]

    def cells(self, x, y):
        """Grid cell (row, column) nearest to points x, y [km]."""
        return (np.clip(np.round(np.asarray(y) / self.dx).astype(int), 0, self.y.size - 1),
                np.clip(np.round(np.asarray(x) / self.dx).astype(int), 0, self.x.size - 1))


def fieldsense_scores(event, products=None):
    """FieldSense's table for one storm: one row per (held-out gauge, hour), with the gauge
    reading `obs`, its fold, its distance to the nearest link and each product's value there
    (products that use gauges are made without the gauge's fold)."""
    t = pickle.load(open(MERGING / "scores" / f"openrainer_{event}.pkl", "rb"))["table"]
    t = t[t.subset == "check"]
    keep = ["station", "time", "fold", "km_to_link", "obs"] + list(products or t.columns[8:])
    return t[keep].reset_index(drop=True)


def fieldsense_table(min_coverage=0.9):
    """All 10 storms' tables, with `event` and `common`: the gauge-hours on which FieldSense
    ranks its maps (gauge reading present and every product with at least min_coverage of
    the best coverage present; link-only maps, which stop far from the links, are not)."""
    t = pd.concat([fieldsense_scores(ev).assign(event=ev) for ev, _, _ in events()], ignore_index=True)
    prods = list(t.columns[5:-1])
    ok = t.obs.notna().values
    n = {p: int((ok & t[p].notna().values).sum()) for p in prods}
    full = [p for p in prods if n[p] >= min_coverage * max(n.values())]
    t["common"] = ok & np.all([t[p].notna().values for p in full], axis=0)
    return t
