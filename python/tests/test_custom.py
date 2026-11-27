import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from radio_free_free import FlowGrid
from radio_free_free.cli import main as flow_main


class CustomFlowTests(unittest.TestCase):
    def test_functions_cli_samples_spherical_and_cylindrical_coordinates(self):
        module_source = """\
def density(a, b, c):
    return a + 10 * b + 100 * c

def temperature(a, b, c):
    return 2.0

def electron_per_baryon(a, b, c):
    return 0.5

def ion_charge2_per_baryon(a, b, c):
    return 1.0
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "custom_flow.py"
            input_path.write_text(module_source)

            for coordinates in ("spherical", "cylindrical"):
                output_path = root / f"{coordinates}.rff"
                with patch(
                    "sys.argv",
                    [
                        "rff-flow",
                        "functions",
                        str(input_path),
                        str(output_path),
                        "--coordinates",
                        coordinates,
                        "--bounds",
                        "0",
                        "1",
                        "0",
                        "1",
                        "0",
                        "1",
                        "--resolution",
                        "1",
                        "1",
                        "1",
                    ],
                ):
                    flow_main()

                flow = FlowGrid.read(output_path)
                self.assertEqual(flow.shape, (1, 1, 1))
                self.assertTrue(output_path.with_suffix(".vtk").is_file())
                self.assertEqual(flow.temperature.item(), 2.0)
                self.assertEqual(flow.electron_per_baryon.item(), 0.5)
                self.assertEqual(flow.ion_charge2_per_baryon.item(), 1.0)

                radius = math.hypot(math.hypot(0.5, 0.5), 0.5)
                polar_angle = math.atan2(math.hypot(0.5, 0.5), 0.5)
                azimuth = math.pi / 4
                if coordinates == "spherical":
                    expected_density = radius + 10 * polar_angle + 100 * azimuth
                else:
                    expected_density = math.hypot(0.5, 0.5) + 10 * azimuth + 100 * 0.5
                self.assertAlmostEqual(flow.density.item(), expected_density)


if __name__ == "__main__":
    unittest.main()
