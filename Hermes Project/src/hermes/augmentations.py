# src/hermes/augmentations.py

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import random

def jitter(series, sigma=0.05):
    noise = np.random.normal(loc=0, scale=sigma, size=series.shape)
    return series + noise

def time_warping(series, sigma=0.2, num_knots=4):
    time_points = np.arange(len(series))
    knot_positions = np.linspace(0, len(series) - 1, num_knots + 2)
    knot_values = np.random.normal(loc=1.0, scale=sigma, size=num_knots + 2)
    spline = interp1d(knot_positions, knot_values, kind='cubic')
    warping_factors = spline(time_points)
    warped_time_points = np.cumsum(warping_factors)
    f = interp1d(time_points, series, bounds_error=False, fill_value=0)
    warped_series_values = f(np.linspace(0, warped_time_points[-1], len(series)))
    return warped_series_values

def scaling(series, sigma=0.1, num_knots=4):
    time_points = np.arange(len(series))
    knot_positions = np.linspace(0, len(series) - 1, num_knots + 2)
    knot_values = np.random.normal(loc=1.0, scale=sigma, size=num_knots + 2)
    spline = interp1d(knot_positions, knot_values, kind='cubic')
    scaling_factors = spline(time_points)
    return series * scaling_factors

def permutation(series, num_segments=4):
    segments = np.array_split(series, num_segments)
    random.shuffle(segments)
    return np.concatenate(segments)
