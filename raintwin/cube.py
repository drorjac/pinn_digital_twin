"""Links, weather stations, radar and gauges at 5 minutes on one 2 km grid, from FieldSense.

Reads the cache that FieldSense's `projects/nowcasting/multisensor` builds (run its `cube`
step once): `<FIELDSENSE_DIR>/data/outputs/_multisensor_nowcasting/<network>/`.

* links: RNN link rain (FieldSense `cml_rnn`, hourly totals spread over 5-min steps), per link
* stations: quality-controlled personal weather stations (PWS), binned to their 2 km cell
* radar: SMHI (Gothenburg) or MRMS (New York) rain rate on the grid
* gauges: city gauges (Gothenburg) or ASOS (New York); used only for scoring here
* FieldSense's own maps of the same data (IDW, radar merged with links and/or stations)

Coordinates are km on the grid: cell (row i, column j) has its centre at (2j, 2i).
`window=(i0, i1, j0, j1)` keeps only rows i0:i1 and columns j0:j1 (New York's grid is 240 km
wide, its sensors fill 40 km of it); the cropped grid then starts at (0, 0).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

FIELDSENSE = Path(os.environ.get("FIELDSENSE_DIR", Path.home() / "PycharmProjects/FieldSense"))
CACHE = FIELDSENSE / "data/outputs/_multisensor_nowcasting"
EVENTS_CSV = FIELDSENSE / "projects/maps/multisensor/results/events.csv"

# FieldSense map products in the cube: R radar, C links (line IDW), P stations (IDW),
# CP links + stations, RC / RP / RCP radar adjusted by the residuals at links / stations / both
PRODUCTS = ("R", "C", "P", "CP", "RC", "RP", "RCP")
# default crop per network: New York's sensors and gauges sit inside rows and columns 40-80
WINDOWS = {"openmesh": (40, 80, 40, 80)}


class Cube:
    def __init__(self, network="openmrg", cache=CACHE, window=None):
        self.network = network
        d = Path(cache) / network
        c = d / "cube"
        self._dir = c
        self.times = pd.DatetimeIndex(np.load(c / "times.npy").astype("datetime64[ns]"))
        self.split = np.load(c / "split.npy")                      # 0 train, 1 validation, 2 test
        self.lat, self.lon = np.load(c / "lat.npy"), np.load(c / "lon.npy")
        i0, i1, j0, j1 = window or WINDOWS.get(network) or (0, self.lat.size, 0, self.lon.size)
        self._win = (slice(i0, i1), slice(j0, j1))
        self.lat, self.lon = self.lat[i0:i1], self.lon[j0:j1]
        ny, nx = self.lat.size, self.lon.size
        self.dx = 2.0
        self.x, self.y = self.dx * np.arange(nx), self.dx * np.arange(ny)
        self.area = np.load(c / "area.npy")[self._win]                        # within 10 km of a sensor
        self._maps = {p: np.load(c / f"{p}.npy", mmap_mode="r") for p in PRODUCTS}
        self._pi = np.load(c / "Pi.npy", mmap_mode="r")
        self._ps = np.load(c / "Ps.npy", mmap_mode="r")

        lr = xr.open_dataset(d / "link_rain_5min.nc").load()
        xa, ya = self.to_xy(lr.site_0_lat.values, lr.site_0_lon.values)
        xb, yb = self.to_xy(lr.site_1_lat.values, lr.site_1_lon.values)
        inside = self._inside(0.5 * (xa + xb), 0.5 * (ya + yb))
        self.links = dict(xa=xa[inside], ya=ya[inside], xb=xb[inside], yb=yb[inside])
        self.link_id = lr.link.values[inside]
        self._link_rain = (lr.rnn.isel(link=np.flatnonzero(inside)).reindex(time=self.times)
                           .transpose("time", "link").values.astype("float32"))

        # stations: the cells that ever hold a PWS, with the cell mean of their readings
        has = np.zeros((ny, nx), bool)
        for k0 in range(0, len(self.times), 2000):
            has |= np.asarray(self._pi[k0:k0 + 2000][(slice(None),) + self._win]).any(0) > 0
        self._pws_cells = np.flatnonzero(has.ravel())
        r, cidx = np.unravel_index(self._pws_cells, (ny, nx))
        self.pws = dict(x=self.x[cidx], y=self.y[r])

        g = xr.open_dataset(c / "gauges_g5.nc").load()
        gx, gy = self.to_xy(g.lat.values, g.lon.values)
        self.gauges = dict(x=gx, y=gy, id=np.array([s.replace("city_", "") for s in g.station.values]))
        self._gauge = g.g5.reindex(time=self.times).transpose("time", "station").values

    # -- geometry ---------------------------------------------------------------------
    def to_xy(self, lat, lon):
        """lat/lon to km on the grid (linear in the grid's own degree spacing)."""
        dlat, dlon = self.lat[1] - self.lat[0], self.lon[1] - self.lon[0]
        return (np.asarray(lon) - self.lon[0]) / dlon * self.dx, (np.asarray(lat) - self.lat[0]) / dlat * self.dx

    def _inside(self, x, y):
        h = self.dx / 2
        return (x >= -h) & (x <= self.x[-1] + h) & (y >= -h) & (y <= self.y[-1] + h)

    # -- time windows -----------------------------------------------------------------
    def events(self):
        """FieldSense's 10 study storms for this network: list of (name, start, end)."""
        df = pd.read_csv(EVENTS_CSV)
        df = df[df.network == self.network]
        return [(r.event_id, pd.Timestamp(r.start), pd.Timestamp(r.end)) for r in df.itertuples()]

    def index(self, start, end):
        """Step indices of [start, end] (5-min labels, interval-ending)."""
        return np.flatnonzero((self.times >= pd.Timestamp(start)) & (self.times <= pd.Timestamp(end)))

    # -- data at steps ----------------------------------------------------------------
    def link_rain(self, idx):
        return self._link_rain[idx]

    def pws_rain(self, idx):
        ps = self._cut(self._ps, idx).reshape(len(idx), -1)[:, self._pws_cells]
        pi = self._cut(self._pi, idx).reshape(len(idx), -1)[:, self._pws_cells] > 0
        return np.where(pi, ps, np.nan)

    def radar(self, idx):
        return self._cut(self._maps["R"], idx)

    def product(self, name, idx):
        return self._cut(self._maps[name], idx)

    def _cut(self, a, idx):
        return np.asarray(a[idx][(slice(None),) + self._win])

    def _load(self, name, idx):
        """Any cube channel (products, or the coverage masks Ci / Pi) at steps idx."""
        if name not in self._maps:
            self._maps[name] = np.load(self._dir / f"{name}.npy", mmap_mode="r")
        return self._cut(self._maps[name], idx)

    def gauge_rain(self, idx):
        return self._gauge[idx]
