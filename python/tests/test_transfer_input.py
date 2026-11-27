import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRANSFER_BINARY = REPOSITORY_ROOT / "build" / "rff-transfer"


@unittest.skipUnless(TRANSFER_BINARY.is_file(), "build rff-transfer first")
class TransferInputTests(unittest.TestCase):
    def test_rff_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "flow.rff"
            input_path.write_bytes(b"not a valid flow file")
            result = subprocess.run(
                [
                    str(TRANSFER_BINARY),
                    "--input",
                    str(input_path),
                    "--output",
                    str(Path(temporary_directory) / "spectrum.csv"),
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
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(".rff input is not supported", result.stderr)


if __name__ == "__main__":
    unittest.main()
