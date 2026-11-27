import csv
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from radio_free_free import FlowGrid


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRANSFER_BINARY = REPOSITORY_ROOT / "build" / "rff-transfer"


@unittest.skipUnless(TRANSFER_BINARY.is_file(), "build rff-transfer first")
class AdaptiveTransferTests(unittest.TestCase):
    def run_transfer(self, flow, directory, name, extra_options=()):
        input_path = directory / f"{name}.rff"
        output_path = directory / f"{name}.csv"
        flow.write(input_path)
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
                "4",
                "--view-theta-deg",
                "0",
                *extra_options,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [
            line
            for line in output_path.read_text().splitlines()
            if not line.startswith("#")
        ]
        values = np.asarray(
            [float(row["projected_intensity_area"]) for row in csv.DictReader(lines)]
        )
        return values, output_path.read_text()

    def test_uniform_flow_adaptive_matches_fixed_grid(self):
        shape = (1, 1, 2)
        ones = np.ones(shape)
        flow = FlowGrid((0, 1, 0, 1, -1, 1), ones, ones, ones, ones)

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            fixed, _ = self.run_transfer(flow, directory, "fixed")
            adaptive, metadata = self.run_transfer(
                flow, directory, "adaptive", ("--adaptive-longitudinal",)
            )

        np.testing.assert_allclose(adaptive, fixed, rtol=1e-9, atol=0)
        self.assertIn("# longitudinal_integration: adaptive", metadata)
        self.assertIn("# integration_tolerance: 0.001", metadata)

    def test_tighter_tolerance_converges_for_varying_flow(self):
        shape = (1, 1, 2)
        ones = np.ones(shape)
        flow = FlowGrid(
            (0, 1, 0, 1, -1, 1),
            np.array([[[0.5, 2.0]]]),
            np.array([[[0.8, 2.5]]]),
            ones,
            ones,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            loose, _ = self.run_transfer(
                flow,
                directory,
                "loose",
                ("--adaptive-longitudinal", "--integration-tolerance", "1e-3"),
            )
            tight, _ = self.run_transfer(
                flow,
                directory,
                "tight",
                ("--adaptive-longitudinal", "--integration-tolerance", "1e-5"),
            )

        np.testing.assert_allclose(loose, tight, rtol=2e-3, atol=0)

    def test_adaptive_ray_stops_at_opaque_surface_and_keeps_foreground(self):
        shape = (1, 1, 2)
        ones = np.ones(shape)
        unocculted = FlowGrid((0, 1, 0, 1, -1, 1), ones, ones, ones, ones)
        far_surface = FlowGrid(
            (0, 1, 0, 1, -1, 1),
            ones,
            ones,
            ones,
            ones,
            occultation_mask=np.array([[[1.0, 0.0]]]),
        )
        near_surface = FlowGrid(
            (0, 1, 0, 1, -1, 1),
            ones,
            ones,
            ones,
            ones,
            occultation_mask=np.array([[[0.0, 1.0]]]),
        )
        adaptive = ("--adaptive-longitudinal",)

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            clear_values, _ = self.run_transfer(
                unocculted, directory, "clear", adaptive
            )
            far_values, _ = self.run_transfer(
                far_surface, directory, "far", adaptive
            )
            near_values, _ = self.run_transfer(
                near_surface, directory, "near", adaptive
            )

        self.assertTrue(np.all(far_values > 0))
        self.assertTrue(np.all(near_values == 0))
        self.assertTrue(np.all(far_values < clear_values))

    def test_early_termination_uses_accumulated_intensity(self):
        shape = (1, 1, 2)
        ones = np.ones(shape)
        flow = FlowGrid((0, 1, 0, 1, -1, 1), ones, ones, ones, ones)

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for mode in ((), ("--adaptive-longitudinal",)):
                reference, _ = self.run_transfer(
                    flow,
                    directory,
                    f"reference-{len(mode)}",
                    (
                        "--n0",
                        "1e6",
                        "--length0",
                        "2",
                        *mode,
                        "--relative-intensity-threshold",
                        "1e-12",
                    ),
                )
                early_stop, _ = self.run_transfer(
                    flow,
                    directory,
                    f"early-stop-{len(mode)}",
                    (
                        "--n0",
                        "1e6",
                        "--length0",
                        "2",
                        *mode,
                        "--relative-intensity-threshold",
                        "0.99",
                    ),
                )

                np.testing.assert_allclose(early_stop, reference, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
