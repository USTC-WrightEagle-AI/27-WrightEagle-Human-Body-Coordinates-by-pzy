#!/usr/bin/env python3

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
for path in (PROJECT_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from face_identity_worker import (  # noqa: E402
    associate_faces, recognize_tracks, virtual_face_box)
from face_robot.engine import DetectedFace  # noqa: E402


def person_keypoints(offset_x):
    points = np.zeros((17, 3), dtype=np.float32)
    points[0] = [150 + offset_x, 116, 0.9]
    points[1] = [135 + offset_x, 105, 0.9]
    points[2] = [165 + offset_x, 105, 0.9]
    points[3] = [120 + offset_x, 115, 0.9]
    points[4] = [180 + offset_x, 115, 0.9]
    points[5] = [100 + offset_x, 190, 0.9]
    points[6] = [200 + offset_x, 190, 0.9]
    return points


def face(box, value):
    return DetectedFace(
        bbox=box, detection_score=0.95,
        embedding=np.full(4, value, dtype=np.float32))


class VirtualFaceBoxTest(unittest.TestCase):

    def test_uses_ear_keypoints_to_build_face_sized_box(self):
        result = virtual_face_box(
            person_keypoints(0), [50, 40, 250, 460], (480, 640, 3))
        np.testing.assert_allclose(result, [105, 61, 195, 169], atol=1)

    def test_falls_back_to_shoulders_when_face_keypoints_are_missing(self):
        points = person_keypoints(0)
        points[:5] = 0
        result = virtual_face_box(
            points, [50, 40, 250, 460], (480, 640, 3))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(150, (result[0] + result[2]) * 0.5)
        self.assertLess(result[3], points[5, 1])


class FaceAssociationTest(unittest.TestCase):

    def setUp(self):
        self.track_ids = [11, 22]
        self.boxes = [[50, 40, 250, 460], [300, 40, 500, 460]]
        self.keypoints = np.stack([
            person_keypoints(0), person_keypoints(250)])

    def test_global_assignment_is_one_to_one_and_order_independent(self):
        faces = [face((355, 75, 445, 165), 2),
                 face((105, 75, 195, 165), 1)]
        assignments = associate_faces(
            faces, self.track_ids, self.boxes, self.keypoints,
            (480, 640, 3))
        assigned = {track_id: detected.embedding[0]
                    for track_id, detected, _, _ in assignments}
        self.assertEqual({11: 1.0, 22: 2.0}, assigned)

    def test_rejects_face_outside_all_virtual_boxes(self):
        assignments = associate_faces(
            [face((550, 350, 620, 430), 1)], self.track_ids,
            self.boxes, self.keypoints, (480, 640, 3))
        self.assertEqual([], assignments)

    def test_recognition_detects_full_frame_once(self):
        engine = SimpleNamespace()
        calls = []

        def detect(frame, max_faces):
            calls.append((frame.shape, max_faces))
            return [face((105, 75, 195, 165), 1),
                    face((355, 75, 445, 165), 2)]

        engine.detect = detect
        database = SimpleNamespace(match=lambda embedding, threshold:
                                   SimpleNamespace(
                                       name=str(int(embedding[0])),
                                       known=True, score=0.8))
        matches = recognize_tracks(
            engine, database, np.zeros((480, 640, 3), dtype=np.uint8),
            self.track_ids, self.boxes, self.keypoints, 0.45)
        self.assertEqual([((480, 640, 3), 0)], calls)
        self.assertEqual({11, 22}, {item["track_id"] for item in matches})
        self.assertTrue(all("virtual_face_bbox" in item for item in matches))


if __name__ == "__main__":
    unittest.main()
