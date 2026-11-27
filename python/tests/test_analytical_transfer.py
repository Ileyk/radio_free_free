import csv
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from radio_free_free import FlowGrid
from radio_free_free.plot import read_radio_output


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRANSFER_BINARY = REPOSITORY_ROOT / "build" / "rff-transfer"


@unittest.skipUnless(TRANSFER_BINARY.is_file(), "build rff-transfer first")
class AnalyticalTransferTests(unittest.TestCase):
    def write_model(self, directory, coordinates="cartesian", opaque_objects="[]"):
        model_path = directory / "model.py"
        model_path.write_text(
            "\n".join(
                [
                    "import numpy as np",
                    "bounds = (-1, 1, -1, 1, -1, 1)",
                    f"coordinates = {coordinates!r}",
                    f"opaque_objects = {opaque_objects}",
                    "def density(x1, x2, x3):",
                    "    return np.ones_like(x1)",
                    "def temperature(x1, x2, x3):",
                    "    return 1.0",
                    "def electron_per_baryon(x1, x2, x3):",
                    "    return 1.0",
                    "def ion_charge2_per_baryon(x1, x2, x3):",
                    "    return 1.0",
                    "",
                ]
            )
        )
        return model_path

    def run_transfer(
        self,
        input_path,
        output_path,
        image_resolution,
        ray_samples,
        extra_options=(),
    ):
        subprocess.run(
            [
                str(TRANSFER_BINARY),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--n0",
                "1e5",
                "--temperature0",
                "1e4",
                "--length0",
                "1",
                "--frequency-min-hz",
                "1e9",
                "--frequency-max-hz",
                "1e10",
                "--frequency-count",
                "3",
                "--view-theta-deg",
                "0",
                *extra_options,
                "--image-resolution",
                str(image_resolution),
                str(image_resolution),
                "--ray-samples",
                str(ray_samples),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [
            row
            for row in output_path.read_text().splitlines()
            if not row.startswith("#")
        ]
        values = np.asarray(
            [
                float(row["projected_intensity_area"])
                for row in csv.DictReader(rows)
            ]
        )
        return values, output_path.read_text()

    def test_uniform_analytical_transfer_converges_with_sampling(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            model_path = self.write_model(directory)
            coarse, _ = self.run_transfer(
                model_path, directory / "coarse.csv", 1, 1
            )
            fine, metadata = self.run_transfer(
                model_path, directory / "fine.csv", 8, 24
            )
            parsed = read_radio_output(directory / "fine.csv")

        np.testing.assert_allclose(fine, coarse, rtol=2e-12, atol=0)
        self.assertIn("# image_resolution: 8,8", metadata)
        self.assertIn("# ray_samples: 24", metadata)
        self.assertIn("# analytical_input:", metadata)
        self.assertIn("# line_of_sight:", metadata)
        self.assertIsNone(parsed.flow_grid)
        np.testing.assert_allclose(parsed.projected_intensity_area, fine)

    def test_spherical_wind_spectrum_converges_with_image_and_ray_sampling(self):
        model_path = REPOSITORY_ROOT / "runs" / "1" / "model.py"
        physical_options = (
            "--n0",
            "1e9",
            "--temperature0",
            "3e5",
            "--length0",
            "70",
            "--frequency-min-hz",
            "1e9",
            "--frequency-max-hz",
            "1e12",
            "--frequency-count",
            "8",
            "--view-theta-deg",
            "53",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            coarse, _ = self.run_transfer(
                model_path, directory / "coarse.csv", 48, 128, physical_options
            )
            ray_refined, _ = self.run_transfer(
                model_path, directory / "ray-refined.csv", 48, 256,
                physical_options,
            )
            image_refined, _ = self.run_transfer(
                model_path, directory / "image-refined.csv", 96, 128,
                physical_options,
            )
            fully_refined, _ = self.run_transfer(
                model_path, directory / "fully-refined.csv", 96, 256,
                physical_options,
            )

        np.testing.assert_allclose(ray_refined, coarse, rtol=5e-4, atol=0)
        np.testing.assert_allclose(image_refined, coarse, rtol=1e-2, atol=0)
        np.testing.assert_allclose(fully_refined, coarse, rtol=1e-2, atol=0)

    def test_spherical_cylindrical_and_cartesian_coordinates_agree(self):
        profiles = {
            "cartesian": "1 / (1 + x1**2 + x2**2 + x3**2)",
            "cylindrical": "1 / (1 + x1**2 + x3**2)",
            "spherical": "1 / (1 + x1**2)",
        }
        results = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for coordinates, density_expression in profiles.items():
                model_path = directory / f"{coordinates}.py"
                model_path.write_text(
                    "\n".join(
                        [
                            "import numpy as np",
                            "bounds = (-1, 1, -1, 1, -1, 1)",
                            f"coordinates = {coordinates!r}",
                            "opaque_objects = []",
                            "def density(x1, x2, x3):",
                            f"    return {density_expression}",
                            "def temperature(x1, x2, x3):",
                            "    return 1.0",
                            "def electron_per_baryon(x1, x2, x3):",
                            "    return 1.0",
                            "def ion_charge2_per_baryon(x1, x2, x3):",
                            "    return 1.0",
                            "",
                        ]
                    )
                )
                values, _ = self.run_transfer(
                    model_path, directory / f"{coordinates}.csv", 4, 12
                )
                results.append(values)

        np.testing.assert_allclose(results[1], results[0], rtol=1e-12, atol=0)
        np.testing.assert_allclose(results[2], results[0], rtol=1e-12, atol=0)

    def test_analytic_sphere_and_cylinder_clip_the_visible_ray(self):
        sphere = (
            '[{"shape": "sphere", "center": (0, 0, 0), "radius": 0.5}]'
        )
        cylinder = (
            '[{"shape": "cylinder", "center": (0, 0, 0), "radius": 0.5, '
            '"height": 1.0, "axis": (0, 0, 1)}]'
        )
        boundary_sphere = (
            '[{"shape": "sphere", "center": (0, 0, 0), "radius": 1.0}]'
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            spectra = []
            for name, objects, theta in (
                ("clear", "[]", "0"),
                ("sphere", sphere, "0"),
                ("cylinder_caps", cylinder, "0"),
                ("cylinder_lateral", cylinder, "90"),
                ("boundary", boundary_sphere, "0"),
            ):
                model_path = self.write_model(
                    directory, opaque_objects=objects
                )
                values, _ = self.run_transfer(
                    model_path,
                    directory / f"{name}.csv",
                    1,
                    16,
                    ("--view-theta-deg", theta),
                )
                spectra.append(values)

        clear, sphere_values, cap_values, lateral_values, boundary_values = (
            spectra
        )
        self.assertTrue(np.all(sphere_values > 0))
        self.assertTrue(np.all(sphere_values < clear))
        self.assertTrue(np.all(cap_values > 0))
        self.assertTrue(np.all(cap_values < clear))
        self.assertTrue(np.all(lateral_values > 0))
        self.assertTrue(np.all(lateral_values < clear))
        np.testing.assert_array_equal(boundary_values, np.zeros_like(clear))

    def test_direct_and_cartesian_workflows_agree_for_uniform_flow(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            model_path = self.write_model(directory)
            direct, _ = self.run_transfer(
                model_path, directory / "direct.csv", 1, 12
            )
            values = np.ones((1, 1, 1))
            flow = FlowGrid(
                (-1, 1, -1, 1, -1, 1), values, values, values, values
            )
            flow_path = directory / "uniform.rff"
            flow.write(flow_path)
            subprocess.run(
                [
                    str(TRANSFER_BINARY),
                    "--input",
                    str(flow_path),
                    "--output",
                    str(directory / "cartesian.csv"),
                    "--n0",
                    "1e5",
                    "--temperature0",
                    "1e4",
                    "--length0",
                    "1",
                    "--frequency-min-hz",
                    "1e9",
                    "--frequency-max-hz",
                    "1e10",
                    "--frequency-count",
                    "3",
                    "--view-theta-deg",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            cartesian_rows = [
                row
                for row in (directory / "cartesian.csv").read_text().splitlines()
                if not row.startswith("#")
            ]
            cartesian = np.asarray(
                [
                    float(row["projected_intensity_area"])
                    for row in csv.DictReader(cartesian_rows)
                ]
            )

        np.testing.assert_allclose(direct, cartesian, rtol=1e-11, atol=0)


if __name__ == "__main__":
    unittest.main()
