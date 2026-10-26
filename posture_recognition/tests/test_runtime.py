#!/usr/bin/env python3

import os
import sys
import unittest
import json

import numpy as np
import torch
import yaml


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from depth_backproject import (
    person_distance_3d, person_position_3d, select_nearest_detection)
from person_identity import NameBinder, transform_point
from pose_features import INPUT_DIM, features_from_coco
from tcn_classifier import TCNPoseClassifier, TCNFeatureTracker
from tcn_model import PoseTCN, model_from_checkpoint


def standing_keypoints():
    points = np.zeros((17, 4), dtype=np.float32)
    coordinates = {
        0: (0.0, -0.45, 2.0),
        5: (-0.18, 0.0, 2.0), 6: (0.18, 0.0, 2.0),
        7: (-0.25, 0.35, 2.0), 8: (0.25, 0.35, 2.0),
        9: (-0.25, 0.65, 2.0), 10: (0.25, 0.65, 2.0),
        11: (-0.14, 0.55, 2.0), 12: (0.14, 0.55, 2.0),
        13: (-0.14, 1.05, 2.0), 14: (0.14, 1.05, 2.0),
        15: (-0.14, 1.55, 2.0), 16: (0.14, 1.55, 2.0),
    }
    for index, coordinate in coordinates.items():
        points[index, :3] = coordinate
        points[index, 3] = 1.0
    return points


class FeatureTest(unittest.TestCase):

    def test_coco_3d_feature_has_expected_dimension(self):
        feature = features_from_coco(
            standing_keypoints(), np.array([0.0, 1.0, 0.0]))
        self.assertIsNotNone(feature)
        self.assertEqual((INPUT_DIM,), feature.shape)
        self.assertTrue(np.all(np.isfinite(feature)))


class TCNModelTest(unittest.TestCase):

    def test_output_shape(self):
        model = PoseTCN(channels=32, dropout=0.0).eval()
        inputs = torch.randn(2, 24, INPUT_DIM)
        self.assertEqual((2, 5), tuple(model(inputs).shape))

    def test_model_is_causal(self):
        torch.manual_seed(7)
        model = PoseTCN(channels=32, dropout=0.0).eval()
        original = torch.randn(1, 24, INPUT_DIM)
        modified = original.clone()
        modified[:, 12:] = torch.randn_like(modified[:, 12:])
        with torch.inference_mode():
            first = model.forward_sequence(original)
            second = model.forward_sequence(modified)
        torch.testing.assert_close(first[:, :12], second[:, :12])

    def test_packaged_checkpoint_loads(self):
        path = os.path.join(ROOT_DIR, "models", "pose_tcn_ntu_fukinect.pth")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = model_from_checkpoint(checkpoint).eval()
        self.assertEqual("gravity_aligned_3d_tcn_v1",
                         checkpoint["feature_type"])
        self.assertEqual(24, checkpoint["window_size"])
        with torch.inference_mode():
            output = model(torch.zeros(1, 24, INPUT_DIM))
        self.assertEqual((1, 5), tuple(output.shape))


class TrackerTest(unittest.TestCase):

    def test_history_survives_small_bbox_motion(self):
        tracker = TCNFeatureTracker(window_size=4)
        first = tracker.update(
            [[10, 10, 110, 210]], [np.ones(INPUT_DIM)])[0]
        second = tracker.update(
            [[15, 12, 115, 212]], [np.full(INPUT_DIM, 2.0)])[0]
        self.assertEqual(first["track_id"], second["track_id"])
        self.assertEqual(2, second["history_size"])
        self.assertEqual((4, INPUT_DIM), second["window"].shape)

    def test_classifier_loads_checkpoint_on_cpu(self):
        path = os.path.join(ROOT_DIR, "models", "pose_tcn_ntu_fukinect.pth")
        classifier = TCNPoseClassifier(
            path, device="cpu", min_history=1,
            deep_squat_guard={"enabled": False})
        detection = {
            "bbox": [0, 0, 100, 200],
            "kpts_3d": standing_keypoints(),
        }
        result = classifier.classify_batch(
            [detection], down_axis=np.array([0.0, 1.0, 0.0]))[0]
        self.assertIn(result["label_name"], classifier.pose_classes + ["unknown"])

    def test_external_bytetrack_id_owns_tcn_history(self):
        tracker = TCNFeatureTracker(window_size=4)
        first = tracker.update(
            [[10, 10, 110, 210]], [np.ones(INPUT_DIM)],
            external_track_ids=[42])[0]
        second = tracker.update(
            [[20, 12, 120, 212]], [np.full(INPUT_DIM, 2.0)],
            external_track_ids=[42])[0]
        self.assertEqual(42, first["track_id"])
        self.assertEqual(42, second["track_id"])
        self.assertEqual(2, second["history_size"])


class NearestPersonTest(unittest.TestCase):

    @staticmethod
    def detection(x, z, bbox):
        points = np.zeros((17, 4), dtype=np.float32)
        for index in (5, 6, 11, 12):
            points[index] = [x, 0.0, z, 1.0]
        return {"bbox": bbox, "kpts_3d": points}

    def test_selects_smallest_euclidean_distance(self):
        side = self.detection(2.0, 1.5, [0, 0, 100, 200])
        front = self.detection(0.0, 2.0, [200, 0, 300, 200])
        selected = select_nearest_detection([side, front])
        self.assertIs(front, selected)
        self.assertAlmostEqual(2.0, selected["distance_m"])
        self.assertAlmostEqual(2.0, person_distance_3d(front["kpts_3d"]))

    def test_invalid_depth_falls_back_to_largest_box(self):
        empty = np.zeros((17, 4), dtype=np.float32)
        small = {"bbox": [0, 0, 50, 100], "kpts_3d": empty.copy()}
        large = {"bbox": [0, 0, 100, 200], "kpts_3d": empty.copy()}
        self.assertIs(large, select_nearest_detection([small, large]))

    def test_person_position_uses_torso_median(self):
        points = np.zeros((17, 4), dtype=np.float32)
        points[5] = [-0.2, 0.0, 2.0, 1.0]
        points[6] = [0.2, 0.0, 2.0, 1.0]
        points[11] = [-0.1, 0.5, 2.1, 1.0]
        points[12] = [0.1, 0.5, 2.1, 1.0]
        np.testing.assert_allclose(
            person_position_3d(points), [0.0, 0.25, 2.05], atol=1e-6)


class IdentityBindingTest(unittest.TestCase):

    def test_name_requires_repeated_face_confirmation(self):
        binder = NameBinder(confirmations=2, history_size=3)
        observation = {
            "track_id": 7, "name": "张三", "known": True, "score": 0.8}
        binder.update([7], [observation])
        self.assertFalse(binder.identity(7)["known"])
        binder.update([7], [observation])
        self.assertEqual("张三", binder.identity(7)["name"])

    def test_transform_point_uses_homogeneous_matrix(self):
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(
            transform_point(matrix, [0.5, 0.5, 0.5]), [1.5, 2.5, 3.5])


class PackageConfigTest(unittest.TestCase):

    def test_model_paths_are_relative_and_exist(self):
        config_path = os.path.join(ROOT_DIR, "config", "pose_3d_config.yaml")
        with open(config_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        for section in ("model", "classifier"):
            value = config[section]["weights_path"]
            self.assertFalse(os.path.isabs(value))
            resolved = os.path.abspath(os.path.join(
                os.path.dirname(config_path), value))
            self.assertTrue(os.path.isfile(resolved), resolved)
        self.assertFalse(config["person_selection"]["nearest_only"])
        self.assertEqual("bytetrack", config["tracking"]["algorithm"])
        self.assertTrue(config["face_recognition"]["enabled"])
        self.assertEqual(0.35, config["face_recognition"][
            "keypoint_confidence"])
        self.assertEqual(0.08, config["face_recognition"][
            "min_assignment_iou"])
        self.assertEqual(0.25, config["face_recognition"][
            "min_assignment_score"])

    def test_config_targets_calibrated_head_camera(self):
        config_path = os.path.join(ROOT_DIR, "config", "pose_3d_config.yaml")
        with open(config_path, "r", encoding="utf-8") as stream:
            camera = yaml.safe_load(stream)["camera"]
        self.assertEqual("260722304986", camera["serial_no"])
        self.assertEqual((1280, 720, 30), (
            camera["width"], camera["height"], camera["fps"]))
        self.assertEqual(
            "/head/camera/color/image_raw", camera["rgb_topic"])
        self.assertEqual(
            "/head/camera/aligned_depth_to_color/image_raw",
            camera["depth_topic"])
        np.testing.assert_allclose(
            camera["down_axis"], [-0.004230, 0.997762, 0.066737])
        calibration_dir = os.path.abspath(os.path.join(
            os.path.dirname(config_path), camera["calibration_dir"]))
        with open(os.path.join(calibration_dir, "session.json"),
                  "r", encoding="utf-8") as stream:
            session = json.load(stream)
        self.assertEqual(camera["serial_no"], session["serial"])
        self.assertEqual((camera["width"], camera["height"], camera["fps"]), (
            session["width"], session["height"], session["fps"]))

    def test_portable_yolo_model_is_selected(self):
        config_path = os.path.join(ROOT_DIR, "config", "pose_3d_config.yaml")
        with open(config_path, "r", encoding="utf-8") as stream:
            model = yaml.safe_load(stream)["model"]
        self.assertTrue(model["weights_path"].endswith(".pt"))

    def test_checkpoint_class_order_matches_runtime_mapping(self):
        path = os.path.join(ROOT_DIR, "models", "pose_tcn_ntu_fukinect.pth")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(
            ["standing", "sitting", "squatting", "lying", "bending"],
            checkpoint["pose_classes"])


if __name__ == "__main__":
    unittest.main()
