import tempfile
import unittest
from pathlib import Path

import numpy as np

from radio_free_free import FlowGrid, spherical_wind
from radio_free_free.flow import PROTON_MASS_G, _number_density_ratio


class FlowGridTests(unittest.TestCase):
    def test_function_grid_is_cell_centered_and_round_trips(self):
        flow = FlowGrid.from_functions(
            (0, 2, 0, 2, 0, 2),
            (2, 2, 2),
            density=lambda x, y, z: x + y + z,
            temperature=lambda x, y, z: np.ones_like(x) * 3,
            electron_per_baryon=lambda x, y, z: np.ones_like(x),
            ion_charge2_per_baryon=lambda x, y, z: np.ones_like(x),
            gaunt_factor=lambda x, y, z: np.ones_like(x),
        )
        self.assertEqual(flow.density.shape, (2, 2, 2))
        self.assertAlmostEqual(flow.density[0, 0, 0], 1.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flow.bin"
            flow.write(path)
            reread = FlowGrid.read(path)
        np.testing.assert_array_equal(reread.density, flow.density)
        self.assertEqual(reread.bounds, flow.bounds)

    def test_spherical_wind_has_expected_radial_profile_and_composition(self):
        flow = spherical_wind((20, 20, 20), inner_radius=0.2, outer_radius=2)
        dx, dy, dz = flow.spacing
        x = flow.bounds[0] + (np.arange(20) + 0.5) * dx
        y = flow.bounds[2] + (np.arange(20) + 0.5) * dy
        z = flow.bounds[4] + (np.arange(20) + 0.5) * dz
        ix, iy, iz = 13, 10, 10
        radius_squared = x[ix] ** 2 + y[iy] ** 2 + z[iz] ** 2
        self.assertAlmostEqual(flow.density[ix, iy, iz], 1.0 / radius_squared)
        np.testing.assert_array_equal(flow.temperature, 1.0)
        np.testing.assert_array_equal(flow.electron_per_baryon, 1.0)
        np.testing.assert_array_equal(flow.ion_charge2_per_baryon, 1.0)

    def test_rejects_inconsistent_fields(self):
        with self.assertRaises(ValueError):
            FlowGrid(
                (0, 1, 0, 1, 0, 1),
                np.ones((1, 1, 1)),
                np.ones((1, 1, 2)),
                np.ones((1, 1, 1)),
                np.ones((1, 1, 1)),
                np.ones((1, 1, 1)),
            )

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
