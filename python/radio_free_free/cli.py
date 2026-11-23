"""Command-line entry point for creating solver flow files."""

from __future__ import annotations

import argparse
import json

from .flow import FIELD_NAMES, resample_vtk
from .wind import spherical_wind


def main() -> None:
    parser = argparse.ArgumentParser(prog="rff-flow")
    commands = parser.add_subparsers(dest="command", required=True)

    wind = commands.add_parser("wind", help="write the analytic spherical-wind benchmark")
    wind.add_argument("output")
    wind.add_argument("--resolution", type=int, nargs=3, default=(96, 96, 96))
    wind.add_argument("--inner-radius", type=float, default=1.0)
    wind.add_argument("--outer-radius", type=float, default=10.0)

    vtk = commands.add_parser("vtk", help="resample a meshio-readable VTK dataset")
    vtk.add_argument("input")
    vtk.add_argument("output")
    vtk.add_argument("--bounds", type=float, nargs=6, required=True)
    vtk.add_argument("--resolution", type=int, nargs=3, required=True)
    vtk.add_argument(
        "--n0",
        type=float,
        default=1.0,
        help="reference baryon number density in cm^-3",
    )
    vtk.add_argument(
        "--density-kind",
        choices=("mass", "number"),
        default="mass",
        help="whether VTK density is mass density or baryon number density",
    )
    vtk.add_argument("--temperature0", type=float, default=1.0)
    vtk.add_argument(
        "--fields",
        required=True,
        help="JSON object mapping canonical field names to VTK array names",
    )

    args = parser.parse_args()
    if args.command == "wind":
        flow = spherical_wind(
            tuple(args.resolution),
            inner_radius=args.inner_radius,
            outer_radius=args.outer_radius,
        )
    else:
        fields = json.loads(args.fields)
        required_fields = set(FIELD_NAMES) - {"gaunt_factor"}
        if (not isinstance(fields, dict) or not required_fields.issubset(fields)
                or set(fields) - set(FIELD_NAMES)):
            parser.error(
                "--fields must map density, temperature, electron_per_baryon, "
                "and ion_charge2_per_baryon; gaunt_factor is optional"
            )
        flow = resample_vtk(
            args.input,
            tuple(args.bounds),
            tuple(args.resolution),
            fields,
            number_density_scale=args.n0,
            temperature_scale=args.temperature0,
            density_kind=args.density_kind,
        )
    flow.write(args.output)
    print(f"Wrote {args.output}: shape={flow.shape}, bounds={flow.bounds}")


if __name__ == "__main__":
    main()
