"""Orthogonal slice viewer for radio_free_free flow grids."""

from __future__ import annotations

import argparse
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .flow import FIELD_NAMES, FlowGrid

_FIELD_LABELS = {
    "density": "Number density / n0",
    "temperature": "Temperature / T0",
    "electron_per_baryon": "Electrons per baryon",
    "ion_charge2_per_baryon": "Ion charge-squared per baryon",
    "occultation_mask": "Opaque volume mask",
}

_PLANES = (
    (0, "YZ", 1, 2),
    (1, "XZ", 0, 2),
    (2, "XY", 0, 1),
)


def orthogonal_slice(
    values: NDArray[np.float64], normal_axis: int, index: int
) -> NDArray[np.float64]:
    """Return a display-oriented slice: rows are vertical, columns horizontal."""
    if normal_axis == 0:
        return values[index, :, :].T
    if normal_axis == 1:
        return values[:, index, :].T
    if normal_axis == 2:
        return values[:, :, index].T
    raise ValueError("normal_axis must be 0 (x), 1 (y), or 2 (z)")


def slice_location(
    flow: FlowGrid, axis: int, coordinate: Optional[float] = None
) -> Tuple[int, float]:
    """Return the nearest cell-center index and coordinate for one axis."""
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0 (x), 1 (y), or 2 (z)")
    lower, upper = flow.bounds[2 * axis : 2 * axis + 2]
    centers = lower + (np.arange(flow.shape[axis]) + 0.5) * flow.spacing[axis]
    if coordinate is None:
        index = flow.shape[axis] // 2
        return index, float(centers[index])
    if not np.isfinite(coordinate):
        raise ValueError(f"{'xyz'[axis]} coordinate must be finite")
    if coordinate < lower or coordinate > upper:
        raise ValueError(
            f"{'xyz'[axis]} coordinate {coordinate} is outside [{lower}, {upper}]"
        )
    index = int(np.argmin(np.abs(centers - coordinate)))
    return index, float(centers[index])


def create_slice_windows(
    flow: FlowGrid,
    field_name: str = "density",
    slice_coordinates: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None,
) -> List[Any]:
    """Create three Matplotlib figures at the requested x, y, and z coordinates.

    Coordinates are mapped to the nearest cell center. Any omitted coordinate
    selects the middle grid cell along that axis.
    """
    if field_name not in FIELD_NAMES:
        raise ValueError(f"unknown flow field {field_name!r}")
    try:
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise ImportError(
            "The flow viewer requires Matplotlib; "
            "install with `python -m pip install -e '.[visualization]'`"
        ) from error

    if slice_coordinates is None:
        slice_coordinates = (None, None, None)
    if len(slice_coordinates) != 3:
        raise ValueError("slice_coordinates must contain x, y, and z values")

    values = getattr(flow, field_name)
    windows = []

    for normal_axis, plane_name, horizontal_axis, vertical_axis in _PLANES:
        axis_name = "xyz"[normal_axis]
        initial_index, coordinate = slice_location(
            flow, normal_axis, slice_coordinates[normal_axis]
        )
        horizontal_name = "xyz"[horizontal_axis]
        vertical_name = "xyz"[vertical_axis]
        extent = (
            flow.bounds[2 * horizontal_axis],
            flow.bounds[2 * horizontal_axis + 1],
            flow.bounds[2 * vertical_axis],
            flow.bounds[2 * vertical_axis + 1],
        )

        figure = plt.figure(figsize=(8, 7))
        figure.canvas.manager.set_window_title(f"{field_name}: {plane_name} slice")
        axis = figure.add_axes((0.1, 0.12, 0.78, 0.78))
        image = axis.imshow(
            orthogonal_slice(values, normal_axis, initial_index),
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0 if field_name == "occultation_mask" else float(np.min(values)),
            vmax=1.0 if field_name == "occultation_mask" else float(np.max(values)),
        )
        mask_slice = orthogonal_slice(
            flow.occultation_mask, normal_axis, initial_index
        )
        if (
            min(mask_slice.shape) >= 2
            and np.min(mask_slice) < 0.5
            and np.max(mask_slice) > 0.5
        ):
            horizontal_centers = (
                flow.bounds[2 * horizontal_axis]
                + (np.arange(flow.shape[horizontal_axis]) + 0.5)
                * flow.spacing[horizontal_axis]
            )
            vertical_centers = (
                flow.bounds[2 * vertical_axis]
                + (np.arange(flow.shape[vertical_axis]) + 0.5)
                * flow.spacing[vertical_axis]
            )
            axis.contour(
                horizontal_centers,
                vertical_centers,
                mask_slice,
                levels=(0.5,),
                colors=("cyan",),
                linewidths=1.5,
            )
            from matplotlib.lines import Line2D

            axis.legend(
                handles=[
                    Line2D(
                        [], [], color="cyan", linewidth=1.5,
                        label="Opaque surface",
                    )
                ],
                loc="upper right",
            )
        axis.set_xlabel(f"{horizontal_name} / R0")
        axis.set_ylabel(f"{vertical_name} / R0")
        axis.set_title(
            f"{field_name} in {plane_name} plane at "
            f"{axis_name} = {coordinate:.6g}"
        )
        figure.colorbar(image, ax=axis, label=_FIELD_LABELS[field_name])
        windows.append(figure)

    return windows


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rff-view", description="View orthogonal slices of an .rff flow"
    )
    parser.add_argument("input", help="flow grid produced by rff-flow")
    parser.add_argument("--x", type=float, help="coordinate of the YZ slice")
    parser.add_argument("--y", type=float, help="coordinate of the XZ slice")
    parser.add_argument("--z", type=float, help="coordinate of the XY slice")
    parser.add_argument(
        "--field", choices=FIELD_NAMES, default="density",
        help="scalar field to display in all three windows (default: density)",
    )
    args = parser.parse_args()

    flow = FlowGrid.read(args.input)
    create_slice_windows(flow, args.field, (args.x, args.y, args.z))
    from matplotlib import pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
