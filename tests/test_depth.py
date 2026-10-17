from __future__ import annotations

import unittest

import numpy as np

from face_robot.depth import (
    FaceDepth,
    estimate_face_depth,
    measure_face_depths,
    select_nearest_face,
)


class FaceDepthTest(unittest.TestCase):
    def test_uses_median_of_inner_face_region(self) -> None:
        depth = np.full((100, 100), 3.0, dtype=np.float32)
        depth[30:70, 30:70] = 1.25
        depth[40, 40] = 0.0

        distance = estimate_face_depth(depth, (20, 20, 80, 80))

        self.assertAlmostEqual(distance, 1.25, places=3)

    def test_invalid_depth_returns_none(self) -> None:
        depth = np.zeros((100, 100), dtype=np.float32)
        self.assertIsNone(estimate_face_depth(depth, (20, 20, 80, 80)))

    def test_face_box_outside_image_is_safely_clipped(self) -> None:
        depth = np.full((100, 100), 1.2, dtype=np.float32)
        self.assertAlmostEqual(
            estimate_face_depth(depth, (-20, -20, 60, 60)), 1.2, places=3
        )
        self.assertIsNone(estimate_face_depth(depth, (120, 120, 180, 180)))

    def test_measurement_keeps_face_order(self) -> None:
        depth = np.zeros((100, 200), dtype=np.float32)
        depth[:, :100] = 0.8
        depth[:, 100:] = 1.6

        measurements = measure_face_depths(
            depth, [(10, 10, 90, 90), (110, 10, 190, 90)]
        )

        self.assertAlmostEqual(measurements[0].distance_m, 0.8, places=3)
        self.assertAlmostEqual(measurements[1].distance_m, 1.6, places=3)

    def test_selects_nearest_face(self) -> None:
        measurements = [
            FaceDepth((0, 0, 10, 10), 1.4),
            FaceDepth((20, 0, 30, 10), 0.7),
        ]
        selection = select_nearest_face(measurements, min_depth_gap_m=0.15)
        self.assertEqual(selection.index, 1)
        self.assertIsNone(selection.reason)

    def test_rejects_ambiguous_or_invalid_depth(self) -> None:
        ambiguous = select_nearest_face(
            [FaceDepth((0, 0, 10, 10), 0.70), FaceDepth((20, 0, 30, 10), 0.78)],
            min_depth_gap_m=0.15,
        )
        invalid = select_nearest_face(
            [FaceDepth((0, 0, 10, 10), 0.7), FaceDepth((20, 0, 30, 10), None)]
        )
        self.assertEqual(ambiguous.reason, "ambiguous_depth")
        self.assertEqual(invalid.reason, "invalid_depth")
        self.assertIsNone(ambiguous.index)
        self.assertIsNone(invalid.index)


if __name__ == "__main__":
    unittest.main()
