"""
Utility functions for CSON Forge.

This module contains utility functions used across the codebase.
"""
from __future__ import annotations

import math
import warnings

import xarray as xr


def compute_timestep_from_cfl(
    grid_size_x: float,
    grid_size_y: float,
    grid_nx: int,
    grid_ny: int,
    grid_ds: xr.Dataset,
    cfl: float = 0.7,
    default_depth: float = 4000.0,
) -> float:
    """
    
    [functionally validated template: requires review for accuracy]

    Compute timestep based on CFL criterion for numerical stability.
    
    The CFL (Courant-Friedrichs-Lewy) condition ensures that the timestep
    is small enough to resolve the fastest gravity waves in the simulation.
    
    Parameters
    ----------
    grid_size_x : float
        Domain size in x-direction (kilometers).
    grid_size_y : float
        Domain size in y-direction (kilometers).
    grid_nx : int
        Number of grid points in x-direction.
    grid_ny : int
        Number of grid points in y-direction.
    grid_ds : xr.Dataset
        Grid dataset containing bathymetry ('h' variable).
    cfl : float, optional
        CFL number (typically 0.5-0.8 for stability). Default is 0.7.
    default_depth : float, optional
        Default ocean depth in meters to use if 'h' variable is not found.
        Default is 4000.0 meters.
    
    Returns
    -------
    float
        Timestep in seconds, rounded to nearest integer, with minimum of 1.0.
        The timestep is adjusted to be an even divisor of 86400 (seconds per day)
        to ensure timesteps align with daily boundaries.
    
    Notes
    -----
    The calculation follows these steps:
    1. Compute minimum grid spacing (dx, dy) from domain size and grid points
    2. Estimate fastest gravity wave speed: c = sqrt(g * H_max)
    3. Apply CFL condition: dt = CFL * dx_min / c
    4. Round to nearest integer
    5. Adjust to nearest divisor of 86400 (ensures daily alignment)
    
    The fastest gravity wave speed is the barotropic wave speed for shallow
    water waves, which depends on the maximum depth in the domain.
    
    The timestep is constrained to be an even divisor of 86400 (seconds per day)
    to ensure that model timesteps align with daily boundaries, which is important
    for forcing data interpolation and output timing.
    """
    # Compute grid spacing in kilometers
    dx_km = grid_size_x / grid_nx
    dy_km = grid_size_y / grid_ny
    dx_min_km = min(dx_km, dy_km)
    
    # Convert to meters
    dx_min_m = dx_min_km * 1000.0
    
    # Get maximum depth from grid dataset (in meters)
    # The 'h' variable is bathymetry (depth) at RHO-points
    if 'h' in grid_ds:
        H_max = float(grid_ds['h'].max().values)
    else:
        # Fallback: use a typical ocean depth if 'h' is not available
        H_max = default_depth
        warnings.warn(
            "Grid dataset does not contain 'h' variable. "
            f"Using default depth of {H_max} m for CFL calculation.",
            UserWarning,
            stacklevel=2
        )
    
    # Gravity acceleration (m/s²)
    g = 9.81
    
    # Fastest gravity wave speed (barotropic wave speed)
    # c = sqrt(g * H) for shallow water waves
    c = math.sqrt(g * H_max)
    
    # Compute timestep from CFL condition: dt = CFL * dx / c
    dt = cfl * dx_min_m / c
    
    # Round to nearest integer (timesteps are typically integers in seconds)
    dt = round(dt)
    
    # Ensure minimum timestep (avoid extremely small values)
    dt = max(dt, 1.0)
    
    # Adjust dt to be an even divisor of 86400 (seconds per day)
    # This ensures that timesteps align with daily boundaries
    # Find the nearest divisor of 86400 that's close to the current dt
    dt_original = int(round(dt))  # Ensure integer
    target_seconds_per_day = 86400
    
    # Collect all divisors of 86400
    divisors = set()
    sqrt_target = int(target_seconds_per_day**0.5) + 1
    for d in range(1, sqrt_target):
        if target_seconds_per_day % d == 0:
            divisors.add(d)
            divisors.add(target_seconds_per_day // d)
    
    # Find the closest divisor
    if divisors:
        best_dt = min(divisors, key=lambda x: abs(x - dt_original))
    else:
        # Fallback (should never happen, but be safe)
        best_dt = 1
    
    dt = int(best_dt)
    
    return dt

