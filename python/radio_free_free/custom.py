"""Build Cartesian flow grids by evaluating functions from a Python file."""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from .flow import Bounds, FlowGrid, FloatArray, PHYSICAL_FIELD_NAMES


def _coordinates(
    x: FloatArray,
    y: FloatArray,
    z: FloatArray,
    coordinate_system: str,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    cylindrical_radius = np.hypot(x, y)
    azimuth = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    if coordinate_system == "spherical":
        radius = np.hypot(cylindrical_radius, z)
        polar_angle = np.arctan2(cylindrical_radius, z)
        return radius, polar_angle, azimuth
    return cylindrical_radius, azimuth, z


def flow_from_python_file(
    path: str | Path,
    bounds: Bounds,
    resolution: Tuple[int, int, int],
    coordinate_system: str,
) -> FlowGrid:
    """Evaluate the four required field functions from a user Python file.

    The functions receive arrays in spherical ``(r, theta, phi)`` or
    cylindrical ``(R, phi, z)`` coordinates. Angles are in radians.
    """
    if coordinate_system not in ("spherical", "cylindrical"):
        raise ValueError("coordinate_system must be 'spherical' or 'cylindrical'")

    source = Path(path).resolve()
    module_name = f"_radio_free_free_user_flow_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load Python flow file: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    field_functions: Dict[
        str, Callable[[FloatArray, FloatArray, FloatArray], FloatArray]
    ] = {}
    coordinate_cache: List[FloatArray] = []
    for field_name in PHYSICAL_FIELD_NAMES:
        function = getattr(module, field_name, None)
        if not callable(function):
            raise ValueError(
                f"Python flow file must define callable {field_name}()"
            )

        def evaluate(
            x: FloatArray,
            y: FloatArray,
            z: FloatArray,
            *,
            _function: Callable = function,
            _field_name: str = field_name,
            _coordinate_cache: List[FloatArray] = coordinate_cache,
        ) -> FloatArray:
            if not _coordinate_cache:
                _coordinate_cache.extend(
                    _coordinates(x, y, z, coordinate_system)
                )
            values = np.asarray(_function(*_coordinate_cache), dtype=np.float64)
            try:
                return np.broadcast_to(values, x.shape)
            except ValueError as error:
                raise ValueError(
                    f"{_field_name}() returned shape {values.shape}, "
                    f"not broadcastable to {x.shape}"
                ) from error

        field_functions[field_name] = evaluate

    return FlowGrid.from_functions(
        bounds,
        resolution,
        density=field_functions["density"],
        temperature=field_functions["temperature"],
        electron_per_baryon=field_functions["electron_per_baryon"],
        ion_charge2_per_baryon=field_functions["ion_charge2_per_baryon"],
    )
