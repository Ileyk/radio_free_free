import unittest

import numpy as np

from radio_free_free.visualize import (
    create_slice_windows,
    orthogonal_slice,
    slice_location,
)
from radio_free_free import FlowGrid


class OrthogonalSliceTests(unittest.TestCase):
    def setUp(self):
        x, y, z = np.meshgrid(np.arange(2), np.arange(3), np.arange(4), indexing="ij")
        self.values = 100 * x + 10 * y + z

    def test_slice_orientation_and_values(self):
        yz = orthogonal_slice(self.values, 0, 1)
        xz = orthogonal_slice(self.values, 1, 2)
        xy = orthogonal_slice(self.values, 2, 3)

        self.assertEqual(yz.shape, (4, 3))
        self.assertEqual(xz.shape, (4, 2))
        self.assertEqual(xy.shape, (3, 2))
        self.assertEqual(yz[3, 2], 123)
        self.assertEqual(xz[3, 1], 123)
        self.assertEqual(xy[2, 1], 123)

    def test_slice_coordinates_choose_nearest_cell_center(self):
        ones = np.ones_like(self.values)
        flow = FlowGrid(
            (0, 2, 0, 3, 0, 4),
            self.values,
            ones,
            ones,
            ones,
            ones,
        )
        self.assertEqual(slice_location(flow, 0, 0.1), (0, 0.5))
        self.assertEqual(slice_location(flow, 1, 2.4), (2, 2.5))
        self.assertEqual(slice_location(flow, 2), (2, 2.5))
        with self.assertRaises(ValueError):
            slice_location(flow, 0, -0.1)

    def test_three_windows_display_requested_slices(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("Matplotlib is an optional dependency")
        matplotlib.use("Agg", force=True)

        from matplotlib import pyplot as plt

        values = self.values
        ones = np.ones_like(values)
        flow = FlowGrid(
            (0, 2, 0, 3, 0, 4),
            values,
            ones,
            ones,
            ones,
            ones,
        )
        windows = create_slice_windows(flow, slice_coordinates=(0.1, 2.4, 0.6))
        try:
            self.assertEqual(len(windows), 3)
            for axis, figure in enumerate(windows):
                image = figure.axes[0].images[0]
                np.testing.assert_array_equal(
                    image.get_array(),
                    orthogonal_slice(values, axis, (0, 2, 0)[axis]),
                )
        finally:
            for figure in windows:
                plt.close(figure)

        singleton = np.ones((1, 1, 1))
        flow = FlowGrid(
            (0, 1, 0, 1, 0, 1),
            singleton,
            singleton,
            singleton,
            singleton,
            singleton,
        )
        windows = create_slice_windows(flow, "temperature")
        try:
            self.assertEqual(len(windows), 3)
            self.assertTrue(all(len(figure.axes[0].images) == 1 for figure in windows))
        finally:
            for figure in windows:
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
