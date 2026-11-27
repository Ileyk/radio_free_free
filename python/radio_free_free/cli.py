"""Command-line entry point for creating solver flow files."""

from __future__ import annotations

import argparse
import json

from .custom import flow_from_python_file
from .flow import PHYSICAL_FIELD_NAMES, resample_vtk
from .wind import spherical_wind


def main() -> None:
    parser = argparse.ArgumentParser(prog="rff-flow")
    commands = parser.add_subparsers(dest="command", required=True)

    wind = commands.add_parser("wind", help="write the analytic spherical-wind benchmark")
    wind.add_argument("output")
    wind.add_argument("--resolution", type=int, nargs=3, default=(96, 96, 96))
    wind.add_argument("--inner-radius", type=float, default=1.0)
    wind.add_argument("--outer-radius", type=float, default=10.0)
    wind.add_argument(
        "--ni_Zi2_per_baryon",
        type=float,
        default=1.0,
        help="uniform sum(Z_i^2 n_i)/n_b abundance (default: 1)",
    )
    wind.add_argument(
        "--eff_Z_i",
        type=float,
        default=1.0,
        help="uniform effective ionic charge (default: 1)",
    )

    functions = commands.add_parser(
        "functions",
        help="evaluate prescribed field functions from a Python file",
    )
    functions.add_argument("input", help="Python file defining the flow fields")
    functions.add_argument("output")
    functions.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        required=True,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Cartesian output-domain bounds in R0 units",
    )
    functions.add_argument(
        "--resolution",
        type=int,
        nargs=3,
        required=True,
        metavar=("NX", "NY", "NZ"),
    )
    functions.add_argument(
        "--coordinates",
        choices=("spherical", "cylindrical"),
        required=True,
        help="coordinate system passed to the field functions",
    )

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
            ni_Zi2_per_baryon=args.ni_Zi2_per_baryon,
            eff_Z_i=args.eff_Z_i,
        )
    elif args.command == "vtk":
        fields = json.loads(args.fields)
        required_fields = set(PHYSICAL_FIELD_NAMES)
        if (not isinstance(fields, dict) or not required_fields.issubset(fields)
                or set(fields) - set(PHYSICAL_FIELD_NAMES)):
            parser.error(
                "--fields must map density, temperature, electron_per_baryon, "
                "and ion_charge2_per_baryon"
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
    else:
        flow = flow_from_python_file(
            args.input,
            tuple(args.bounds),
            tuple(args.resolution),
            args.coordinates,
        )
    vtk_output = flow.write_with_vtk(args.output)
    print(f"Wrote {args.output}: shape={flow.shape}, bounds={flow.bounds}")
    print(f"Wrote {vtk_output}")


if __name__ == "__main__":
    main()
