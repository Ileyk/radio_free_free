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
    ni_Zi2_per_baryon: float = 1.0,
    eff_Z_i: float = 1.0,
) -> FlowGrid:
    """Create a fully ionized, isothermal number-density proportional to r^-2 wind.

    Coordinates are in units of the reference radius R0, and density stores
    baryon number density in units of n0, its value at R0. The central sphere
    is vacuum to avoid the singularity at r=0 and represents the stellar
    interior. ``outer_radius`` defines a spherical cutoff inside the
    enclosing Cartesian grid; density is zero beyond that radius, while
    temperature and composition remain uniform throughout the grid.
    The uniform composition is specified by ``ni_Zi2_per_baryon`` and
    ``eff_Z_i``; the electron abundance is derived from their ratio.
    """
    if not (0 < inner_radius < outer_radius):
        raise ValueError("require 0 < inner_radius < outer_radius")
    if not np.isfinite(ni_Zi2_per_baryon) or ni_Zi2_per_baryon <= 0:
        raise ValueError("ni_Zi2_per_baryon must be finite and positive")
    if not np.isfinite(eff_Z_i) or eff_Z_i < 1:
        raise ValueError("eff_Z_i must be finite and at least one")
    bounds = (
        -outer_radius, outer_radius,
        -outer_radius, outer_radius,
        -outer_radius, outer_radius,
    )

    def radius_squared(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return x * x + y * y + z * z

    def wind_number_density(
        x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        radius_squared_values = radius_squared(x, y, z)
        number_density = np.zeros_like(radius_squared_values)
        in_wind = (
            (radius_squared_values >= inner_radius * inner_radius)
            & (radius_squared_values <= outer_radius * outer_radius)
        )
        number_density[in_wind] = 1.0 / radius_squared_values[in_wind]
        return number_density

    electron_per_baryon = ni_Zi2_per_baryon / eff_Z_i
    return FlowGrid.from_functions(
        bounds,
        resolution,
        density=wind_number_density,
        temperature=lambda x, y, z: np.ones_like(x),
        electron_per_baryon=lambda x, y, z: np.full_like(
            x, electron_per_baryon
        ),
        ion_charge2_per_baryon=lambda x, y, z: np.full_like(
            x, ni_Zi2_per_baryon
        ),
    )
