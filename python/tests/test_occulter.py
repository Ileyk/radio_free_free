import tempfile
import unittest
from pathlib import Path

import numpy as np

from radio_free_free import FlowGrid, spherical_wind
from radio_free_free.occulter import add_occulting_shape


class OccultingShapeTests(unittest.TestCase):
    def setUp(self):
        self.flow = spherical_wind(
            (8, 8, 8), inner_radius=0.1, outer_radius=2.0
        )

    def test_sphere_mask_uses_dimensionless_cell_centers(self):
        masked = add_occulting_shape(
            self.flow, "sphere", (0.0, 0.0, 0.0), 0.5
        )
        self.assertEqual(masked.occultation_mask[4, 4, 4], 1.0)
        self.assertEqual(masked.occultation_mask[0, 0, 0], 0.0)
        np.testing.assert_array_equal(
            self.flow.occultation_mask, np.zeros_like(self.flow.density)
        )

    def test_cylinder_mask_respects_axis_and_full_height(self):
        masked = add_occulting_shape(
            self.flow,
            "cylinder",
            (0.0, 0.0, 0.0),
            0.4,
            height=0.6,
            axis="x",
        )
        self.assertEqual(masked.occultation_mask[4, 4, 4], 1.0)
        self.assertEqual(masked.occultation_mask[6, 4, 4], 0.0)
        self.assertEqual(masked.occultation_mask[4, 6, 4], 0.0)

    def test_mask_is_persisted_with_flow_file(self):
        masked = add_occulting_shape(
            self.flow, "sphere", (0.0, 0.0, 0.0), 0.5
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "occulted.rff"
            masked.write(path)
            reread = FlowGrid.read(path)
        np.testing.assert_array_equal(
            reread.occultation_mask, masked.occultation_mask
        )

    def test_rejects_invalid_shapes_and_dimensions(self):
        with self.assertRaisesRegex(ValueError, "cylinder height"):
            add_occulting_shape(
                self.flow, "cylinder", (0.0, 0.0, 0.0), 1.0
            )
        with self.assertRaisesRegex(ValueError, "radius"):
            add_occulting_shape(
                self.flow, "sphere", (0.0, 0.0, 0.0), 0.0
            )


if __name__ == "__main__":
    unittest.main()
