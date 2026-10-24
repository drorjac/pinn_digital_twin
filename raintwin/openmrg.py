"""OpenMRG (Gothenburg, July 2015) on the radar grid, in km: links, radar, gauges.

Reads the OpenSense example subset (`openmrg_*_8d.nc`). Coordinates are projected onto the
radar's own stereographic grid by a quadratic fit lat/lon -> (x, y) over the radar cells
(sub-metre residuals over this domain), so no projection library is needed.
"""

import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import xarray as xr

DATA = Path(os.environ.get("OPENMRG_DIR", Path.home() / "data/cml/openmrg/_sample_8d"))


def _design(lat, lon):
    lat, lon = np.ravel(lat), np.ravel(lon)
    return np.c_[np.ones_like(lat), lat, lon, lat * lon, lat**2, lon**2]


class OpenMRG:
    def __init__(self, data_dir=DATA, crop_km=8.0):
        self.dir = Path(data_dir)
        self.rad = xr.open_dataset(self.dir / "openmrg_rad_8d.nc")
        self.cml = xr.open_dataset(self.dir / "openmrg_cml_8d.nc")
        self.gauge = xr.open_dataset(self.dir / "openmrg_municp_gauge_8d.nc")
        # radar grid in km, flipped so that y increases with row index
        xr_km, yr_km = self.rad.x.values / 1e3, self.rad.y.values / 1e3
        self._flip = yr_km[1] < yr_km[0]
        lat, lon = self.rad.lat.values, self.rad.lon.values
        D = _design(lat, lon)
        XX, YY = np.meshgrid(xr_km, yr_km)
        self._cx = np.linalg.lstsq(D, XX.ravel(), rcond=None)[0]
        self._cy = np.linalg.lstsq(D, YY.ravel(), rcond=None)[0]
        c = self.cml
        xa, ya = self.project(c.site_0_lat.values, c.site_0_lon.values)
        xb, yb = self.project(c.site_1_lat.values, c.site_1_lon.values)
        gx, gy = self.project(self.gauge.lat.values, self.gauge.lon.values)
        # crop the radar grid to the link network plus a margin
        allx, ally = np.r_[xa, xb], np.r_[ya, yb]
        ix = np.where((xr_km >= allx.min() - crop_km) & (xr_km <= allx.max() + crop_km))[0]
        yv = yr_km[::-1] if self._flip else yr_km
        iy = np.where((yv >= ally.min() - crop_km) & (yv <= ally.max() + crop_km))[0]
        self.ix, self.iy = ix, iy
        x0, y0 = xr_km[ix[0]], yv[iy[0]]
        self.x = xr_km[ix] - x0
        self.y = yv[iy] - y0
        self.links = dict(xa=xa - x0, ya=ya - y0, xb=xb - x0, yb=yb - y0)
        self.gauges = dict(x=gx - x0, y=gy - y0, id=self.gauge.id.values)
        self.cml_id = c.cml_id.values

    def project(self, lat, lon):
        D = _design(lat, lon)
        return D @ self._cx, D @ self._cy

    def radar(self, start, end):
        """Radar rain rate (T, ny, nx) mm/h on the cropped grid, y increasing; and times."""
        R = self.rad.R.sel(time=slice(start, end))
        a = R.values
        if self._flip:
            a = a[:, ::-1, :]
        return a[:, self.iy][:, :, self.ix], R.time.values

    def link_rain(self, start, end, freq="5min"):
        """Path-averaged rain rate per link (T, n_links) mm/h: the dataset's reference
        retrieval 'R', averaged over sublinks and over each interval (labelled by its start)."""
        R = self.cml.R.sel(time=slice(start, end)).mean("sublink_id", skipna=True)
        R = R.resample(time=freq, label="left", closed="left").mean()
        return R.transpose("time", "cml_id").values, R.time.values

    def gauge_rain(self, start, end, freq="5min"):
        """Gauge rain rate (T, n_gauges) mm/h from accumulations (mm per 1 min)."""
        G = self.gauge.rainfall_amount.sel(time=slice(start, end))
        dt_h = (G.time.values[1] - G.time.values[0]) / np.timedelta64(1, "h")
        G = (G / dt_h).resample(time=freq, label="left", closed="left").mean()
        return G.transpose("time", "id").values, G.time.values

    # -- finer twin grids ----------------------------------------------------------
    def fine_grid(self, factor):
        """Cell centres of a grid `factor` times finer than the radar's, same extent."""
        d = self.x[1] - self.x[0]
        h = d / factor
        xf = self.x[0] - d / 2 + h / 2 + h * np.arange(self.x.size * factor)
        yf = self.y[0] - d / 2 + h / 2 + h * np.arange(self.y.size * factor)
        return xf, yf

    def radar_operator(self, factor):
        """Sparse H (n_radar_cells, n_fine_cells): each radar cell is the mean of its fine cells."""
        ny, nx = self.y.size, self.x.size
        nxf = nx * factor
        rows, cols = [], []
        for r in range(ny):
            for c in range(nx):
                for a in range(factor):
                    for b in range(factor):
                        rows.append(r * nx + c)
                        cols.append((r * factor + a) * nxf + c * factor + b)
        return sp.csr_matrix((np.full(len(rows), 1.0 / factor**2), (rows, cols)),
                             shape=(ny * nx, ny * nx * factor**2))
