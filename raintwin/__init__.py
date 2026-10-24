"""A digital twin of a rain field observed by microwave links.

State: the rain field on a grid plus the storm's motion vector. Forecast: semi-Lagrangian
advection with stochastic growth/decay. Analysis: an ensemble Kalman filter whose
observation operator averages the field along each link path. Other sensors (radar, gauges)
can be added as further observation sets, optionally with a learned bias.
"""

from .core import (path_operator, idw_points, line_idw, advect, smooth_noise,
                   segment_distance, RainTwin, ResidualBlend, scores)
