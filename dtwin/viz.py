"""One visual language for every notebook.

Colour roles
------------
categorical  fixed order (truth, twin, sensors/data, ...), never cycled
FIELD        sequential, one hue light -> dark (magnitudes: temperature, concentration)
DIVERGING    two hues around a neutral grey (errors, signed quantities)
UNCERTAINTY  sequential violet (standard deviations)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

# categorical slots, fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

# semantic roles used across notebooks
TRUTH, TWIN, DATA, PRIOR = INK, BLUE, ORANGE, MUTED

FIELD = LinearSegmentedColormap.from_list(
    "field", ["#fbfaf7", "#f6d9a8", "#eda100", "#eb6834", "#b8322a", "#5a1530", "#1d0b1c"])
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging", ["#1d4f91", "#2a78d6", "#9cc3ee", "#ebeae6", "#f3b49a", "#eb6834", "#9c3412"])
UNCERTAINTY = LinearSegmentedColormap.from_list(
    "uncertainty", ["#fbfaf7", "#d6d0f5", "#9085e9", "#6250d6", "#2e2370"])
for _cm in (FIELD, DIVERGING, UNCERTAINTY):
    try:
        mpl.colormaps.register(_cm)
    except ValueError:
        pass


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2.0,
        "font.family": "sans-serif", "image.cmap": "field",
        "animation.embed_limit": 60, "animation.html": "jshtml",
    })


def _nogrid(ax):
    ax.grid(False)
    return ax


def spacetime(ax, U, x, t, cmap=FIELD, label="u", vmin=None, vmax=None, cbar=True, **kw):
    """Space-time diagram: x horizontal, time vertical (upwards)."""
    _nogrid(ax)
    im = ax.imshow(U, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[x[0], x[-1], t[0], t[-1]], interpolation="bilinear", **kw)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    if cbar:
        plt.colorbar(im, ax=ax, label=label, pad=0.02, fraction=0.05)
    return im


def field(ax, C, L=1.0, cmap=FIELD, vmin=None, vmax=None, cbar=True, label=None, title=None):
    """A 2D field on [0, L]^2 (rows are y)."""
    _nogrid(ax)
    im = ax.imshow(C, origin="lower", extent=[0, L, 0, L], cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="bilinear")
    ax.set_xticks([0, L / 2, L])
    ax.set_yticks([0, L / 2, L])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if cbar:
        plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.03)
    return im


def sensors_2d(ax, xs, ys, **kw):
    style = dict(s=36, marker="o", facecolor="white", edgecolor=INK, linewidth=1.2, zorder=5)
    style.update(kw)
    return ax.scatter(xs, ys, **style)


def snapshot_grid(fields, titles, L=1.0, ncols=None, cmap=FIELD, vmin=None, vmax=None,
                  label=None, size=2.6, suptitle=None):
    n = len(fields)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    vmin = np.min(fields) if vmin is None else vmin
    vmax = np.max(fields) if vmax is None else vmax
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols + 0.8, size * nrows),
                             constrained_layout=True, squeeze=False)
    for ax, f, ti in zip(axes.flat, fields, titles):
        im = field(ax, f, L, cmap, vmin, vmax, cbar=False, title=ti)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, label=label, fraction=0.025, pad=0.02)
    if suptitle:
        fig.suptitle(suptitle, x=0.02, ha="left", fontweight="bold")
    return fig, axes


def animate_1d(x, frames, t, labels=None, colors=None, styles=None, ylim=None,
               title="", xlabel="x", ylabel="u", points=None, interval=80, figsize=(7, 3.4)):
    """frames: list of arrays (n_frames, nx), one per line. points: optional
    (x_pts, Y_pts (n_frames, m)) drawn as markers (e.g. sensor readings)."""
    frames = [np.asarray(f) for f in frames]
    labels = labels or [None] * len(frames)
    colors = colors or SERIES
    styles = styles or ["-"] * len(frames)
    fig, ax = plt.subplots(figsize=figsize, dpi=80)
    lines = [ax.plot(x, f[0], styles[i], color=colors[i], label=labels[i])[0]
             for i, f in enumerate(frames)]
    sc = None
    if points is not None:
        sc = ax.plot(points[0], points[1][0], "o", color=DATA, ms=6, mec="white",
                     mew=1.2, label="sensors", zorder=5)[0]
    if ylim is None:
        lo = min(f.min() for f in frames)
        hi = max(f.max() for f in frames)
        pad = 0.08 * (hi - lo + 1e-12)
        ylim = (lo - pad, hi + pad)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if any(labels) or points is not None:
        ax.legend(loc="upper right")
    ttl = ax.set_title(title)

    def update(k):
        for ln, f in zip(lines, frames):
            ln.set_ydata(f[k])
        if sc is not None:
            sc.set_ydata(points[1][k])
        ttl.set_text(f"{title}   t = {t[k]:.2f}")
        return lines

    anim = animation.FuncAnimation(fig, update, frames=len(t), interval=interval, blit=False)
    plt.close(fig)
    return anim


def animate_2d(fields_list, t, titles, L=1.0, cmaps=None, vlims=None, interval=90,
               size=3.0, sensors=None, label=None):
    """fields_list: list of arrays (n_frames, n, n) shown side by side."""
    k = len(fields_list)
    cmaps = cmaps or [FIELD] * k
    vlims = vlims or [(np.min(f), np.max(f)) for f in fields_list]
    fig, axes = plt.subplots(1, k, figsize=(size * k + 0.6, size + 0.4), constrained_layout=True,
                             squeeze=False, dpi=80)
    axes = axes[0]
    ims = []
    for ax, F, ti, cm, (lo, hi) in zip(axes, fields_list, titles, cmaps, vlims):
        im = field(ax, F[0], L, cm, lo, hi, cbar=True, title=ti, label=label)
        if sensors is not None:
            sensors_2d(ax, *sensors, s=14)
        ims.append(im)
    sup = fig.suptitle(f"t = {t[0]:.2f}", x=0.02, ha="left", fontsize=10, color=INK2)

    def update(i):
        for im, F in zip(ims, fields_list):
            im.set_data(F[i])
        sup.set_text(f"t = {t[i]:.2f}")
        return ims

    anim = animation.FuncAnimation(fig, update, frames=len(t), interval=interval, blit=False)
    plt.close(fig)
    return anim


def band(ax, x, mean, std, color=TWIN, label=None, k=2.0, alpha=0.18):
    """Mean line with a +-k std shaded band."""
    ax.fill_between(x, mean - k * std, mean + k * std, color=color, alpha=alpha, lw=0)
    return ax.plot(x, mean, color=color, label=label)[0]


def save(fig, name):
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig.savefig(os.path.join(here, "figures", name), bbox_inches="tight")
