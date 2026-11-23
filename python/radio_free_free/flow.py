"""Dimensionless Cartesian flow grids and the solver interchange format."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Bounds = Tuple[float, float, float, float, float, float]
FLOW_MAGIC = b"RFFLOW2\0"
LEGACY_FLOW_MAGIC = b"RFFLOW1\0"
FIELD_NAMES = (
    "density",
    "temperature",
    "electron_per_baryon",
    "ion_charge2_per_baryon",
)
_LEGACY_FIELD_COUNT = 5
PROTON_MASS_G = 1.67262192369e-24


@dataclass(frozen=True)
class FlowGrid:
    """Uniform cell-centered grid, with positions and fields dimensionless.

    ``density`` stores baryon number density relative to n0, and temperature
    relative to T0 supplied to the C++ postprocessor. The abundance fields
    are number ratios per baryon.
    """

    bounds: Bounds
    density: FloatArray
    temperature: FloatArray
    electron_per_baryon: FloatArray
    ion_charge2_per_baryon: FloatArray

    def __post_init__(self) -> None:
        bounds = np.asarray(self.bounds, dtype=np.float64)
        if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
            raise ValueError("bounds must contain six finite values")
        if not (bounds[0] < bounds[1] and bounds[2] < bounds[3]
                and bounds[4] < bounds[5]):
            raise ValueError("each lower grid bound must be below its upper bound")
        object.__setattr__(self, "bounds", tuple(float(value) for value in bounds))

        density = np.asarray(self.density, dtype=np.float64)
        shape = density.shape
        if len(shape) != 3 or any(n < 1 for n in shape):
            raise ValueError("flow fields must be non-empty 3D arrays")
        for name in FIELD_NAMES:
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != shape:
                raise ValueError(f"{name} has shape {values.shape}, expected {shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "density", density)

        for name in ("density", "electron_per_baryon", "ion_charge2_per_baryon"):
            if np.any(getattr(self, name) < 0):
                raise ValueError(f"{name} must be non-negative")
        if np.any(self.temperature <= 0):
            raise ValueError("temperature must be positive")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.density.shape

    @property
    def spacing(self) -> Tuple[float, float, float]:
        nx, ny, nz = self.shape
        return (
            (self.bounds[1] - self.bounds[0]) / nx,
            (self.bounds[3] - self.bounds[2]) / ny,
            (self.bounds[5] - self.bounds[4]) / nz,
        )

    @classmethod
    def from_functions(
        cls,
        bounds: Bounds,
        shape: Tuple[int, int, int],
        *,
        density: Callable[[FloatArray, FloatArray, FloatArray], FloatArray],
        temperature: Callable[[FloatArray, FloatArray, FloatArray], FloatArray],
        electron_per_baryon: Callable[[FloatArray, FloatArray, FloatArray], FloatArray],
        ion_charge2_per_baryon: Callable[[FloatArray, FloatArray, FloatArray], FloatArray],
    ) -> "FlowGrid":
        """Evaluate dimensionless prescribed fields at Cartesian cell centers."""
        if len(shape) != 3 or any(int(n) != n or n < 1 for n in shape):
            raise ValueError("shape must contain three positive integers")
        axes = [
            np.linspace(bounds[2 * axis], bounds[2 * axis + 1], shape[axis] + 1)
            for axis in range(3)
        ]
        centers = [(edge[:-1] + edge[1:]) * 0.5 for edge in axes]
        x, y, z = np.meshgrid(*centers, indexing="ij")
        functions = (
            density,
            temperature,
            electron_per_baryon,
            ion_charge2_per_baryon,
        )
        values = [np.asarray(function(x, y, z), dtype=np.float64) for function in functions]
        return cls(bounds, *values)

    def write(self, path: str | Path) -> None:
        """Write a little-endian binary grid consumed by the C++ executable."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            stream.write(FLOW_MAGIC)
            stream.write(struct.pack("<III", *self.shape))
            stream.write(struct.pack("<6d", *self.bounds))
            fields = np.stack([getattr(self, name) for name in FIELD_NAMES], axis=-1)
            stream.write(np.asarray(fields, dtype="<f8", order="C").tobytes(order="C"))

    @classmethod
    def read(cls, path: str | Path) -> "FlowGrid":
        """Read a solver interchange grid, validating its header and payload."""
        with Path(path).open("rb") as stream:
            magic = stream.read(8)
            if magic == FLOW_MAGIC:
                field_count = len(FIELD_NAMES)
            elif magic == LEGACY_FLOW_MAGIC:
                field_count = _LEGACY_FIELD_COUNT
            else:
                raise ValueError("not a radio_free_free flow file")
            dims_raw = stream.read(12)
            bounds_raw = stream.read(48)
            if len(dims_raw) != 12 or len(bounds_raw) != 48:
                raise ValueError("truncated flow header")
            shape = struct.unpack("<III", dims_raw)
            bounds = struct.unpack("<6d", bounds_raw)
            expected = int(np.prod(shape)) * field_count * 8
            payload = stream.read()
        if len(payload) != expected:
            raise ValueError(f"flow payload has {len(payload)} bytes; expected {expected}")
        values = np.frombuffer(payload, dtype="<f8").reshape((*shape, field_count))
        values = values[..., : len(FIELD_NAMES)]
        return cls(bounds, *(values[..., i].copy() for i in range(len(FIELD_NAMES))))


def resample_vtk(
    path: str | Path,
    bounds: Bounds,
    shape: Tuple[int, int, int],
    field_names: Mapping[str, str],
    *,
    number_density_scale: float = 1.0,
    temperature_scale: float = 1.0,
    density_kind: str = "mass",
) -> FlowGrid:
    """Load meshio-readable VTK data and linearly resample fields onto a grid.

    ``field_names`` maps the canonical FlowGrid field names to names in the
    VTK file. By default, the source ``density`` field is a mass density in
    g cm^-3 and is converted to baryon number density using the proton mass;
    set ``density_kind="number"`` when it is already in cm^-3. Density and
    temperature are then divided by n0 and T0. Source values outside the
    convex hull are set to zero for density and abundances, and to the
    minimum source value for temperature.
    """
    try:
        import meshio
        from scipy.interpolate import griddata
    except ImportError as error:
        raise ImportError("VTK resampling requires the 'vtk' optional dependencies") from error

    missing = set(FIELD_NAMES) - set(field_names)
    if missing:
        raise ValueError(f"field_names is missing mappings for {sorted(missing)}")
    unknown = set(field_names) - set(FIELD_NAMES)
    if unknown:
        raise ValueError(f"field_names has unknown keys: {sorted(unknown)}")
    if not np.isfinite(number_density_scale) or number_density_scale <= 0:
        raise ValueError("number_density_scale must be finite and positive")
    if density_kind not in ("mass", "number"):
        raise ValueError("density_kind must be 'mass' or 'number'")
    if not np.isfinite(temperature_scale) or temperature_scale <= 0:
        raise ValueError("temperature_scale must be finite and positive")

    mesh = meshio.read(path)
    axes = [
        np.linspace(bounds[2 * axis], bounds[2 * axis + 1], shape[axis] + 1)
        for axis in range(3)
    ]
    centers = [(edge[:-1] + edge[1:]) * 0.5 for edge in axes]
    query = np.column_stack(
        [coordinate.ravel() for coordinate in np.meshgrid(*centers, indexing="ij")]
    )
    sampled = {}

    for canonical in FIELD_NAMES:
        source_name = field_names[canonical]
        if source_name in mesh.point_data:
            points = np.asarray(mesh.points, dtype=np.float64)
            values = np.asarray(mesh.point_data[source_name], dtype=np.float64)
        elif source_name in mesh.cell_data:
            point_chunks = []
            value_chunks = []
            for block, data in zip(mesh.cells, mesh.cell_data[source_name]):
                point_chunks.append(mesh.points[block.data].mean(axis=1))
                value_chunks.append(np.asarray(data, dtype=np.float64))
            points = np.concatenate(point_chunks, axis=0)
            values = np.concatenate(value_chunks, axis=0)
        else:
            raise ValueError(f"VTK field {source_name!r} was not found")

        if values.ndim != 1:
            raise ValueError(f"VTK field {source_name!r} must be scalar")
        fill = float(np.min(values)) if canonical == "temperature" else 0.0
        sampled[canonical] = griddata(
            points, values, query, method="linear", fill_value=fill
        ).reshape(shape)

    sampled["density"] = _number_density_ratio(
        sampled["density"], number_density_scale, density_kind
    )
    sampled["temperature"] /= temperature_scale
    return FlowGrid(bounds, *(sampled[name] for name in FIELD_NAMES))


def _number_density_ratio(
    density: FloatArray, number_density_scale: float, density_kind: str
) -> FloatArray:
    conversion = (
        PROTON_MASS_G * number_density_scale
        if density_kind == "mass"
        else number_density_scale
    )
    return density / conversion
