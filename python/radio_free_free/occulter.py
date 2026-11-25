"""Create a flow file with a spherical or cylindrical occultation mask."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from .flow import FlowGrid


def add_occulting_shape(
    flow: FlowGrid,
    shape: str,
    center: Tuple[float, float, float],
    radius: float,
    *,
    height: Optional[float] = None,
    axis: str = "z",
) -> FlowGrid:
    """Return a copy of ``flow`` with the requested solid shape masked.

    The sphere radius and cylinder radius/height use the flow grid's
    dimensionless coordinate units. Cylinders are centered on ``center`` and
    aligned with the selected Cartesian axis. Added shapes combine with any
    existing mask.
    """
    if shape not in ("sphere", "cylinder"):
        raise ValueError("shape must be 'sphere' or 'cylinder'")
    if len(center) != 3 or not all(np.isfinite(value) for value in center):
        raise ValueError("center must contain three finite coordinates")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y', or 'z'")
    if shape == "sphere" and height is not None:
        raise ValueError("height is only valid for a cylinder")
    if shape == "cylinder" and (
        height is None or not np.isfinite(height) or height <= 0
    ):
        raise ValueError("cylinder height must be finite and positive")

    coordinates = [
        flow.bounds[2 * index]
        + (np.arange(flow.shape[index]) + 0.5) * flow.spacing[index]
        - center[index]
        for index in range(3)
    ]
    x, y, z = np.meshgrid(*coordinates, indexing="ij")
    offsets = (x, y, z)
    radius_squared = radius * radius
    if shape == "sphere":
        inside = x * x + y * y + z * z <= radius_squared
    else:
        axial_index = "xyz".index(axis)
        radial_squared = sum(
            offset * offset
            for index, offset in enumerate(offsets)
            if index != axial_index
        )
        inside = (
            (radial_squared <= radius_squared)
            & (np.abs(offsets[axial_index]) <= height * 0.5)
        )
    mask = np.maximum(flow.occultation_mask, inside.astype(np.float64))
    return replace(flow, occultation_mask=mask)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rff-occult",
        description="Add a spherical or cylindrical opaque volume to a flow file",
    )
    parser.add_argument("input", help="flow grid produced by rff-flow")
    parser.add_argument("output", help="flow grid with the occultation mask")
    parser.add_argument("--shape", choices=("sphere", "cylinder"), required=True)
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="center in dimensionless flow coordinates",
    )
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument(
        "--height",
        type=float,
        help="full cylinder height (required for --shape cylinder)",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="z",
        help="cylinder axis (default: z)",
    )
    args = parser.parse_args()
    if args.shape == "sphere" and args.height is not None:
        parser.error("--height can only be used with --shape cylinder")
    if args.shape == "cylinder" and args.height is None:
        parser.error("--height is required with --shape cylinder")

    flow = FlowGrid.read(args.input)
    masked_flow = add_occulting_shape(
        flow,
        args.shape,
        tuple(args.center),
        args.radius,
        height=args.height,
        axis=args.axis,
    )
    masked_flow.write(args.output)
    print(
        f"Wrote {args.output}: shape={masked_flow.shape}, "
        f"occulted cells={int(np.count_nonzero(masked_flow.occultation_mask))}"
    )


if __name__ == "__main__":
    main()
