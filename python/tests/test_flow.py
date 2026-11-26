import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from radio_free_free import FlowGrid, spherical_wind
from radio_free_free.cli import main as flow_main
from radio_free_free.flow import (
    FLOW_MAGIC,
    LEGACY_FLOW_MAGIC,
    PREVIOUS_FLOW_MAGIC,
    PROTON_MASS_G,
    _number_density_ratio,
)


class FlowGridTests(unittest.TestCase):
    def test_function_grid_is_cell_centered_and_round_trips(self):
        flow = FlowGrid.from_functions(
            (0, 2, 0, 2, 0, 2),
            (2, 2, 2),
            density=lambda x, y, z: x + y + z,
            temperature=lambda x, y, z: np.ones_like(x) * 3,
            electron_per_baryon=lambda x, y, z: np.ones_like(x),
            ion_charge2_per_baryon=lambda x, y, z: np.ones_like(x),
        )
        self.assertEqual(flow.density.shape, (2, 2, 2))
        self.assertAlmostEqual(flow.density[0, 0, 0], 1.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flow.bin"
            flow.write(path)
            self.assertEqual(path.read_bytes()[:8], FLOW_MAGIC)
            reread = FlowGrid.read(path)
        np.testing.assert_array_equal(reread.density, flow.density)
        np.testing.assert_array_equal(
            reread.occultation_mask, np.zeros_like(reread.density)
        )
        self.assertEqual(reread.bounds, flow.bounds)

    def test_reads_previous_four_field_flow(self):
        bounds = (0, 1, 0, 1, 0, 1)
        fields = np.asarray([1.0, 2.0, 3.0, 4.0], dtype="<f8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "previous-flow.rff"
            with path.open("wb") as stream:
                stream.write(PREVIOUS_FLOW_MAGIC)
                stream.write(struct.pack("<III", 1, 1, 1))
                stream.write(struct.pack("<6d", *bounds))
                stream.write(fields.tobytes())
            flow = FlowGrid.read(path)

        self.assertEqual(flow.ion_charge2_per_baryon.item(), 4.0)
        np.testing.assert_array_equal(
            flow.occultation_mask, np.zeros_like(flow.density)
        )

    def test_writes_vtk_structured_points_with_x_fastest_cell_data(self):
        density = np.arange(8, dtype=float).reshape((2, 2, 2))
        flow = FlowGrid(
            (0, 2, 0, 4, 0, 6),
            density,
            np.ones_like(density),
            np.ones_like(density),
            np.ones_like(density),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flow.vtk"
            flow.write_vtk(path)
            lines = path.read_text().splitlines()

        self.assertIn("DATASET STRUCTURED_POINTS", lines)
        self.assertIn("DIMENSIONS 3 3 3", lines)
        self.assertIn("ORIGIN 0 0 0", lines)
        self.assertIn("SPACING 1 2 3", lines)
        self.assertIn("CELL_DATA 8", lines)
        density_header = lines.index("SCALARS density double 1")
        self.assertEqual(lines[density_header + 1], "LOOKUP_TABLE default")
        vtk_density = np.asarray(
            lines[density_header + 2 : density_header + 10], dtype=float
        )
        np.testing.assert_array_equal(
            vtk_density, density.transpose(2, 1, 0).ravel()
        )

    def test_flow_cli_writes_rff_and_same_stem_vtk(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wind.rff"
            with patch(
                "sys.argv",
                [
                    "rff-flow",
                    "wind",
                    str(output),
                    "--resolution",
                    "4",
                    "4",
                    "4",
                    "--ni_Zi2_per_baryon",
                    "1",
                    "--eff_Z_i",
                    "2",
                ],
            ):
                flow_main()

            vtk_output = output.with_suffix(".vtk")
            self.assertTrue(output.is_file())
            self.assertTrue(vtk_output.is_file())
            reread = FlowGrid.read(output)
            self.assertEqual(reread.shape, (4, 4, 4))
            self.assertIn("DATASET STRUCTURED_POINTS", vtk_output.read_text())
            in_domain = reread.temperature > 0
            np.testing.assert_allclose(
                reread.ion_charge2_per_baryon[in_domain], 1.0
            )
            np.testing.assert_allclose(reread.electron_per_baryon[in_domain], 0.5)

    def test_spherical_wind_has_expected_radial_profile_and_composition(self):
        flow = spherical_wind((20, 20, 20), inner_radius=0.2, outer_radius=2)
        dx, dy, dz = flow.spacing
        x = flow.bounds[0] + (np.arange(20) + 0.5) * dx
        y = flow.bounds[2] + (np.arange(20) + 0.5) * dy
        z = flow.bounds[4] + (np.arange(20) + 0.5) * dz
        ix, iy, iz = 13, 10, 10
        radius_squared = x[ix] ** 2 + y[iy] ** 2 + z[iz] ** 2
        self.assertAlmostEqual(flow.density[ix, iy, iz], 1.0 / radius_squared)
        grid_radius_squared = (
            x[:, np.newaxis, np.newaxis] ** 2
            + y[np.newaxis, :, np.newaxis] ** 2
            + z[np.newaxis, np.newaxis, :] ** 2
        )
        outside_sphere = grid_radius_squared > 2.0**2
        self.assertTrue(np.any(outside_sphere))
        for field in (
            flow.density,
            flow.temperature,
            flow.electron_per_baryon,
            flow.ion_charge2_per_baryon,
        ):
            np.testing.assert_array_equal(field[outside_sphere], 0.0)
        inside_sphere = ~outside_sphere
        np.testing.assert_array_equal(flow.temperature[inside_sphere], 1.0)
        np.testing.assert_array_equal(flow.electron_per_baryon[inside_sphere], 1.0)
        np.testing.assert_array_equal(
            flow.ion_charge2_per_baryon[inside_sphere], 1.0
        )

    def test_spherical_wind_supports_uniform_effective_ion_composition(self):
        flow = spherical_wind(
            (20, 20, 20),
            inner_radius=0.2,
            outer_radius=2,
            ni_Zi2_per_baryon=1.0,
            eff_Z_i=2.0,
        )
        inside_domain = flow.temperature > 0
        np.testing.assert_allclose(flow.ion_charge2_per_baryon[inside_domain], 1.0)
        np.testing.assert_allclose(flow.electron_per_baryon[inside_domain], 0.5)
        np.testing.assert_allclose(
            flow.ion_charge2_per_baryon[inside_domain]
            / flow.electron_per_baryon[inside_domain],
            2.0,
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            spherical_wind(eff_Z_i=0.5)

    def test_rejects_inconsistent_fields(self):
        with self.assertRaises(ValueError):
            FlowGrid(
                (0, 1, 0, 1, 0, 1),
                np.ones((1, 1, 1)),
                np.ones((1, 1, 2)),
                np.ones((1, 1, 1)),
                np.ones((1, 1, 1)),
            )

    def test_reads_legacy_gaunt_flow_and_discards_gaunt_field(self):
        bounds = (0, 1, 0, 1, 0, 1)
        fields = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype="<f8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-flow.rff"
            with path.open("wb") as stream:
                stream.write(LEGACY_FLOW_MAGIC)
                stream.write(struct.pack("<III", 1, 1, 1))
                stream.write(struct.pack("<6d", *bounds))
                stream.write(fields.tobytes())
            flow = FlowGrid.read(path)

        self.assertEqual(flow.density.item(), 1.0)
        self.assertEqual(flow.temperature.item(), 2.0)
        self.assertEqual(flow.electron_per_baryon.item(), 3.0)
        self.assertEqual(flow.ion_charge2_per_baryon.item(), 4.0)
        self.assertFalse(hasattr(flow, "gaunt_factor"))

    def test_vtk_mass_and_number_density_normalize_to_n0(self):
        mass_density = np.asarray([PROTON_MASS_G * 10.0])
        number_density = np.asarray([10.0])
        np.testing.assert_allclose(
            _number_density_ratio(mass_density, 5.0, "mass"), [2.0]
        )
        np.testing.assert_allclose(
            _number_density_ratio(number_density, 5.0, "number"), [2.0]
        )


if __name__ == "__main__":
    unittest.main()
