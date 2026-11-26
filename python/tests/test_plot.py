import tempfile
import unittest
from pathlib import Path

import numpy as np

from radio_free_free.plot import (
    FlowGridParameters,
    PhysicalScales,
    RadioImage,
    RadioSpectrum,
    plot_radio_output,
    projected_area_to_flux_jy,
    read_radio_output,
)


class RadioOutputPlotTests(unittest.TestCase):
    def test_reads_image_csv_and_preserves_projection_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.csv"
            path.write_text(
                "# intensity units: erg s^-1 cm^-2 Hz^-1 sr^-1\n"
                "# frequency_hz: 10000000000\n"
                "# projected_pixel_area_cm2: 2e26\n"
                "# line_of_sight: 0,0,1\n"
                "u_over_R0,-1,1\n"
                "-1,1,2\n"
                "1,3,4\n"
            )
            output = read_radio_output(path)

        self.assertIsInstance(output, RadioImage)
        self.assertEqual(output.frequency_hz, 1e10)
        self.assertEqual(output.projected_pixel_area_cm2, 2e26)
        np.testing.assert_array_equal(output.intensity, [[1, 2], [3, 4]])

    def test_reads_spectrum_csv_and_detects_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectrum.csv"
            path.write_text(
                "# integrated flux density units: Jy\n"
                "frequency_hz,flux_density_jy\n"
                "1000000000,0.1\n"
                "10000000000,0.25\n"
            )
            output = read_radio_output(path)

        self.assertIsInstance(output, RadioSpectrum)
        np.testing.assert_array_equal(output.frequency_hz, [1e9, 1e10])
        np.testing.assert_allclose(output.flux_density_jy, [0.1, 0.25])

    def test_legacy_jy_spectrum_is_displayed_in_mjy(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("Matplotlib is an optional dependency")
        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        spectrum = RadioSpectrum(
            np.asarray([1e9, 1e10]),
            np.asarray([0.1, 0.25]),
            "Jy",
        )
        figure, slider = plot_radio_output(spectrum)
        try:
            self.assertIsNone(slider)
            self.assertEqual(figure.axes[0].get_ylabel(), "Flux density [mJy]")
            np.testing.assert_array_equal(
                figure.axes[0].lines[0].get_ydata(), [100.0, 250.0]
            )
        finally:
            plt.close(figure)

    def test_reads_distance_independent_spectrum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectrum.csv"
            path.write_text(
                "# projected intensity-area units: erg s^-1 Hz^-1\n"
                "# n0_cm^-3: 1000000\n"
                "# temperature0_k: 10000\n"
                "# length0_cm: 69570000000\n"
                "# flow_resolution: 4,5,6\n"
                "# flow_bounds_r0: -2,2,-3,3,-4,4\n"
                "frequency_hz,projected_intensity_area\n"
                "1000000000,1e20\n"
                "10000000000,2e20\n"
            )
            output = read_radio_output(path)

        self.assertIsInstance(output, RadioSpectrum)
        self.assertIsNone(output.flux_density_jy)
        np.testing.assert_array_equal(
            output.projected_intensity_area, [1e20, 2e20]
        )
        self.assertEqual(output.scales.number_density_cm3, 1e6)
        self.assertEqual(output.scales.temperature_k, 1e4)
        self.assertEqual(output.scales.length_solar_radii, 1.0)
        self.assertEqual(output.flow_grid.resolution, (4, 5, 6))
        self.assertEqual(output.flow_grid.box_dimensions, (4.0, 6.0, 8.0))

    def test_reads_legacy_xy_image_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-image.csv"
            path.write_text(
                "# intensity units: erg s^-1 cm^-2 Hz^-1 sr^-1\n"
                "# rows: x/R0 cell centers; columns: y/R0 cell centers\n"
                "x_over_R0,-1,1\n"
                "-1,1,2\n"
                "1,3,4\n"
            )
            output = read_radio_output(path)

        self.assertIsInstance(output, RadioImage)
        self.assertEqual(output.axis_names, ("x", "y"))
        np.testing.assert_array_equal(output.intensity, [[1, 2], [3, 4]])

    def test_rejects_non_monotonic_image_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.csv"
            path.write_text(
                "u_over_R0,1,0\n"
                "0,1,2\n"
            )
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                read_radio_output(path)

    def test_plots_image_and_spectrum(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("Matplotlib is an optional dependency")
        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        image = RadioImage(
            np.asarray([0.0, 1.0]),
            np.asarray([-1.0, 1.0]),
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            1e10,
            (0.0, 0.0, 1.0),
            "erg s^-1 cm^-2 Hz^-1 sr^-1",
            2.0,
        )
        spectrum = RadioSpectrum(
            np.asarray([1e9, 1e10]),
            None,
            "erg s^-1 Hz^-1",
            np.asarray([1e20, 2e20]),
            PhysicalScales(1e6, 1e4, 1.5e13),
            FlowGridParameters((32, 24, 16), (-10, 10, -8, 8, -6, 6)),
        )
        image_figure, image_slider = plot_radio_output(image, 100)
        spectrum_figure, spectrum_slider = plot_radio_output(spectrum, 100)
        figures = [image_figure, spectrum_figure]
        try:
            self.assertIsNotNone(image_slider)
            self.assertIsNotNone(spectrum_slider)
            np.testing.assert_array_equal(
                image_figure.axes[0].images[0].get_array(), image.intensity.T
            )
            self.assertEqual(spectrum_figure.axes[0].get_xscale(), "log")
            self.assertEqual(spectrum_figure.axes[0].get_yscale(), "log")
            self.assertEqual(
                spectrum_figure.axes[0].get_xlabel(), "Frequency [GHz]"
            )
            self.assertEqual(
                spectrum_figure.axes[0].get_ylabel(), "Flux density [mJy]"
            )
            np.testing.assert_array_equal(
                spectrum_figure.axes[0].lines[0].get_xdata(), [1.0, 10.0]
            )
            points = spectrum_figure.axes[0].lines[1]
            check = spectrum_figure._rff_linford_checkbutton
            self.assertFalse(points.get_visible())
            self.assertEqual(check.get_status(), [False])
            check.set_active(0)
            self.assertTrue(points.get_visible())
            self.assertGreater(spectrum_figure.axes[0].get_xlim()[1], 34.74)
            np.testing.assert_array_equal(
                points.get_xdata(),
                [5.0, 6.96, 8.99, 10.99, 13.42, 16.36, 29.58, 34.74],
            )
            np.testing.assert_array_equal(
                points.get_ydata(),
                [0.075, 0.097, 0.147, 0.198, 0.229, 0.290, 0.450, 0.517],
            )
            check.set_active(0)
            self.assertFalse(points.get_visible())
            self.assertLess(spectrum_figure.axes[0].get_xlim()[1], 34.74)
            np.testing.assert_allclose(
                spectrum_figure.axes[0].lines[0].get_ydata(),
                projected_area_to_flux_jy(
                    spectrum.projected_intensity_area, 100
                )
                * 1e3,
            )
            scale_legend = spectrum_figure.axes[0].texts[0].get_text()
            self.assertIn("n_0", scale_legend)
            self.assertIn("T_0", scale_legend)
            self.assertIn("R_\\odot", scale_legend)
            self.assertIn("N = 32 \\times 24 \\times 16", scale_legend)
            self.assertIn("\\mathrm{Box}", scale_legend)
            self.assertIn("20 \\times 16 \\times 12", scale_legend)
            initial_flux = spectrum_figure.axes[0].lines[0].get_ydata().copy()
            spectrum_slider.set_val(spectrum_slider.valmax)
            np.testing.assert_allclose(
                spectrum_figure.axes[0].lines[0].get_ydata(),
                initial_flux / 1e2,
                rtol=2e-2,
            )
            initial_image_flux = image_figure.texts[0].get_text()
            self.assertIn("mJy", initial_image_flux)
            image_slider.set_val(image_slider.valmax)
            self.assertNotEqual(
                image_figure.texts[0].get_text(), initial_image_flux
            )
        finally:
            for figure in figures:
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
