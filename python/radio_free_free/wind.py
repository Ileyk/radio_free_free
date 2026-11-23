"""Analytic benchmark flows."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .flow import FlowGrid


def spherical_wind(
    resolution: Tuple[int, int, int] = (96, 96, 96),
    *,
    inner_radius: float = 0.5,
    outer_radius: float = 10.0,
) -> FlowGrid:
    """Create a fully ionized, isothermal number-density proportional to r^-2 wind.

    Coordinates are in units of the reference radius R0, and density stores
    baryon number density in units of n0, its value at R0. The central sphere
    is vacuum to avoid the singularity at r=0 and represents the stellar
    interior.
    """
    if not (0 < inner_radius < outer_radius):
        raise ValueError("require 0 < inner_radius < outer_radius")
    bounds = (
        -outer_radius, outer_radius,
        -outer_radius, outer_radius,
        -outer_radius, outer_radius,
    )

    def wind_number_density(
        x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        radius_squared = x * x + y * y + z * z
        number_density = np.zeros_like(radius_squared)
        outside_star = radius_squared >= inner_radius * inner_radius
        number_density[outside_star] = 1.0 / radius_squared[outside_star]
        return number_density

    return FlowGrid.from_functions(
        bounds,
        resolution,
        density=wind_number_density,
        temperature=lambda x, y, z: np.ones_like(x),
        electron_per_baryon=lambda x, y, z: np.ones_like(x),
        ion_charge2_per_baryon=lambda x, y, z: np.ones_like(x),
    )
