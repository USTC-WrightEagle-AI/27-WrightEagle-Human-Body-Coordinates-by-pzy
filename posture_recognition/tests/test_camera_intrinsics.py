import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "posture_recognition/src"))
from camera_intrinsics import validate_snapshot_intrinsics


class CameraIntrinsicsTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = json.loads(
            (ROOT / "calibration/head_d455/intrinsics.json").read_text())
        self.msg = SimpleNamespace(
            width=640, height=480,
            k=[v for row in self.snapshot["camera_matrix"] for v in row],
            d=copy.copy(self.snapshot["distortion_coefficients"]),
            distortion_model="plumb_bob")

    def validate(self):
        validate_snapshot_intrinsics(self.msg, self.snapshot, 640, 480)

    def test_matching_snapshot_and_small_focal_drift_pass(self):
        self.validate()
        self.msg.k[0] *= 1.0006
        self.msg.k[4] *= 1.0006
        self.validate()

    def test_each_matrix_mismatch_names_parameter_and_values(self):
        for index, name in ((0, "fx"), (4, "fy"), (2, "cx"), (5, "cy")):
            with self.subTest(name=name):
                previous = self.msg.k[index]
                self.msg.k[index] += 2.0
                with self.assertRaisesRegex(ValueError, name + ": 实时=") as error:
                    self.validate()
                self.assertIn("差值=+2.000000000 px", str(error.exception))
                self.msg.k[index] = previous

    def test_robot_reported_focal_lengths_pass(self):
        self.msg.k[0] = 387.34613037109375
        self.msg.k[4] = 386.80743408203125
        self.validate()

    def test_focal_drift_outside_half_percent_is_rejected(self):
        for index, name in ((0, "fx"), (4, "fy")):
            for scale in (0.994, 1.006):
                with self.subTest(name=name, scale=scale):
                    previous = self.msg.k[index]
                    self.msg.k[index] *= scale
                    with self.assertRaisesRegex(ValueError, name + ": 实时="):
                        self.validate()
                    self.msg.k[index] = previous

    def test_resolution_mismatch(self):
        self.msg.width = 1280
        with self.assertRaisesRegex(ValueError, "分辨率: 实时=1280x480"):
            self.validate()

    def test_distortion_mismatch_includes_models(self):
        self.msg.d = [0.0] * 5
        with self.assertRaisesRegex(ValueError, "D: 实时=") as error:
            self.validate()
        self.assertIn("plumb_bob", str(error.exception))
        self.assertIn(self.snapshot["lens_model"], str(error.exception))

    def test_distortion_lengths_report_clear_error_without_broadcasting(self):
        for coefficients in ([], [0.0], [0.0] * 8):
            with self.subTest(coefficients=coefficients):
                self.msg.d = coefficients
                with self.assertRaisesRegex(ValueError, "D: 实时="):
                    self.validate()

    def test_reports_all_mismatches_together(self):
        self.msg.width = 1280
        self.msg.k[0] *= 1.02
        self.msg.d = []
        with self.assertRaises(ValueError) as error:
            self.validate()
        for field in ("分辨率:", "fx:", "D:"):
            self.assertIn(field, str(error.exception))


if __name__ == "__main__":
    unittest.main()
