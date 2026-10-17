from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_robot.camera import parse_camera_source
from face_robot.database import FaceDatabase, l2_normalize
from remember_person import build_parser as build_enrollment_parser


class FaceDatabaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "faces.npz"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_enroll_reload_and_match(self) -> None:
        database = FaceDatabase(self.database_path)
        database.upsert("Alice", np.array([1.0, 0.0, 0.0]))
        database.upsert("Bob", np.array([0.0, 1.0, 0.0]))

        reloaded = FaceDatabase(self.database_path)
        result = reloaded.match(np.array([0.9, 0.1, 0.0]), threshold=0.8)

        self.assertEqual(result.name, "Alice")
        self.assertTrue(result.known)
        self.assertGreater(result.score, 0.9)

    def test_unknown_below_threshold(self) -> None:
        database = FaceDatabase(self.database_path)
        database.upsert("Alice", np.array([1.0, 0.0]))

        result = database.match(np.array([0.8, 0.6]), threshold=0.9)

        self.assertEqual(result.name, "Unknown")
        self.assertFalse(result.known)
        self.assertAlmostEqual(result.score, 0.8, places=5)

    def test_replace_requires_explicit_flag(self) -> None:
        database = FaceDatabase(self.database_path)
        database.upsert("Alice", np.array([1.0, 0.0]))

        with self.assertRaisesRegex(ValueError, "--replace"):
            database.upsert("Alice", np.array([0.0, 1.0]))

        database.upsert("Alice", np.array([0.0, 1.0]), replace=True)
        result = database.match(np.array([0.0, 1.0]), threshold=0.99)
        self.assertEqual(result.name, "Alice")

    def test_remove_person_and_reload(self) -> None:
        database = FaceDatabase(self.database_path)
        database.upsert("Alice", np.array([1.0, 0.0]))
        database.upsert("Bob", np.array([0.0, 1.0]))

        database.remove("Alice")
        reloaded = FaceDatabase(self.database_path)

        self.assertEqual(reloaded.names, ["Bob"])
        with self.assertRaises(KeyError):
            reloaded.remove("Alice")

    def test_unicode_name_round_trip(self) -> None:
        database = FaceDatabase(self.database_path)
        database.upsert("张三", np.array([1.0, 2.0]))
        self.assertEqual(FaceDatabase(self.database_path).names, ["张三"])

    def test_zero_embedding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            l2_normalize(np.zeros(4))


class CameraSourceTest(unittest.TestCase):
    def test_numeric_source(self) -> None:
        self.assertEqual(parse_camera_source("2"), 2)

    def test_device_path_source(self) -> None:
        self.assertEqual(parse_camera_source("/dev/video4"), "/dev/video4")


class EnrollmentArgumentsTest(unittest.TestCase):
    def test_default_minimum_face_size_supports_longer_distance(self) -> None:
        arguments = build_enrollment_parser().parse_args(["Test Person"])
        self.assertEqual(arguments.min_face_size, 80)
        self.assertEqual(arguments.camera, "head")
        self.assertTrue(arguments.depth_filter)


if __name__ == "__main__":
    unittest.main()
