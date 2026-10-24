"""dtwin: a small toolkit for building digital twins of 1D and 2D spatio-temporal fields.

Modules
-------
solvers       physics models (1D heat, 1D Burgers exact solution, 2D advection-diffusion)
sensors       virtual sensors and observation operators
assimilation  Kalman filter and ensemble Kalman filter (with localization)
rom           reduced-order models: POD, DMD, optimal sensor placement
pinn          physics-informed neural networks (PyTorch)
viz           consistent plotting and animation helpers
"""

from . import solvers, sensors, assimilation, rom, viz  # noqa: F401

__version__ = "1.0.0"
