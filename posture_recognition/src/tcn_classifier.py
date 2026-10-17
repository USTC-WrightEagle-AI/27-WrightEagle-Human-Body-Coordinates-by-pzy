#!/usr/bin/env python3
"""Streaming TCN inference for five static human postures."""

import math

import numpy as np
import torch

from pose_features import features_from_coco
from tcn_model import model_from_checkpoint


POSE_CLASSES = ["standing", "sitting", "squatting", "lying", "bending"]
CN_LABELS = {
    "standing": "站立",
    "sitting": "坐下",
    "squatting": "蹲",
    "lying": "躺",
    "bending": "弯腰",
    "unknown": "未知",
}


def sanitize_torso_depth(kpts_3d):
    """Replace RealSense torso depth outliers with the torso median."""
    key_indices = (5, 6, 11, 12)
    z_values = [kpts_3d[index, 2] for index in key_indices
                if kpts_3d[index, 3] > 0]
    if len(z_values) < 3:
        return kpts_3d
    median_z = float(np.median(z_values))
    if max(abs(value - median_z) for value in z_values) <= 0.20:
        return kpts_3d
    corrected = np.copy(kpts_3d)
    for index in key_indices:
        if (corrected[index, 3] > 0 and
                abs(corrected[index, 2] - median_z) > 0.20):
            corrected[index, 2] = median_z
    return corrected


class TCNFeatureTracker:
    """Associate detections and keep one causal feature window per track."""

    def __init__(self, window_size, max_track_age=15):
        self.window_size = int(window_size)
        self.max_track_age = int(max_track_age)
        self.tracks = {}
        self.next_track_id = 0
        self.frame_index = 0

    @staticmethod
    def _center_and_diagonal(bbox):
        width = max(1.0, float(bbox[2] - bbox[0]))
        height = max(1.0, float(bbox[3] - bbox[1]))
        center = np.array([
            (bbox[0] + bbox[2]) * 0.5,
            (bbox[1] + bbox[3]) * 0.5,
        ], dtype=np.float32)
        return center, math.hypot(width, height)

    @staticmethod
    def _iou(first, second):
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(
            0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(
            0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    def _match(self, boxes):
        candidates = []
        for detection_index, bbox in enumerate(boxes):
            center, diagonal = self._center_and_diagonal(bbox)
            for track_id, track in self.tracks.items():
                track_center, track_diagonal = self._center_and_diagonal(
                    track["bbox"])
                normalized_distance = float(np.linalg.norm(
                    center - track_center)) / max(
                        50.0, 0.5 * (diagonal + track_diagonal))
                iou = self._iou(bbox, track["bbox"])
                if iou >= 0.05 or normalized_distance <= 1.0:
                    candidates.append((
                        2.0 * iou - normalized_distance,
                        detection_index,
                        track_id,
                    ))
        candidates.sort(reverse=True)
        matches = {}
        used_tracks = set()
        for _, detection_index, track_id in candidates:
            if detection_index in matches or track_id in used_tracks:
                continue
            matches[detection_index] = track_id
            used_tracks.add(track_id)
        return matches

    def update(self, boxes, features, external_track_ids=None):
        if len(boxes) != len(features):
            raise ValueError("boxes and features must have equal length")
        if (external_track_ids is not None and
                len(external_track_ids) != len(boxes)):
            raise ValueError(
                "external_track_ids and boxes must have equal length")
        self.frame_index += 1
        stale_tracks = [
            track_id for track_id, track in self.tracks.items()
            if self.frame_index - track["last_seen"] > self.max_track_age
        ]
        for track_id in stale_tracks:
            del self.tracks[track_id]

        if external_track_ids is None:
            matches = self._match(boxes)
        else:
            matches = {
                index: int(track_id)
                for index, track_id in enumerate(external_track_ids)
            }
        outputs = []
        for detection_index, (bbox, feature) in enumerate(zip(boxes, features)):
            track_id = matches.get(detection_index)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    "bbox": list(bbox),
                    "features": [],
                    "last_seen": self.frame_index,
                    "stale_count": 0,
                }
            track = self.tracks[track_id]
            track["bbox"] = list(bbox)
            track["last_seen"] = self.frame_index
            stale_feature = feature is None
            if feature is not None:
                track["stale_count"] = 0
                track["features"].append(
                    np.asarray(feature, dtype=np.float32))
                track["features"] = track["features"][-self.window_size:]
            else:
                track["stale_count"] += 1

            history = track["features"]
            if history:
                padded = [history[0]] * (self.window_size - len(history)) + history
                window = np.asarray(padded, dtype=np.float32)
            else:
                window = None
            outputs.append({
                "track_id": track_id,
                "history_size": len(history),
                "window": window,
                "current_feature": feature,
                "stale_feature": stale_feature,
                "stale_count": track["stale_count"],
            })
        return outputs


class TCNPoseClassifier:
    """24-frame TCN trained with NTU RGB+D 120 and FUKinect-Fall."""

    def __init__(self, weights_path, device="cuda", confidence_threshold=0.4,
                 min_history=6, max_track_age=15, max_stale_features=3,
                 deep_squat_guard=None):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("TCN requested CUDA but CUDA is unavailable")
        self.device = str(device)
        self.confidence_threshold = float(confidence_threshold)
        checkpoint = torch.load(
            weights_path, map_location=self.device, weights_only=False)
        if checkpoint.get("feature_type") != "gravity_aligned_3d_tcn_v1":
            raise ValueError("weights are not a gravity_aligned_3d_tcn_v1 model")
        self.pose_classes = checkpoint.get("pose_classes", POSE_CLASSES)
        self.window_size = int(checkpoint.get("window_size", 24))
        self.min_history = max(1, min(int(min_history), self.window_size))
        self.max_stale_features = max(0, int(max_stale_features))

        guard = deep_squat_guard or {}
        self.deep_squat_guard_enabled = bool(guard.get("enabled", True))
        self.deep_squat_knee_max = float(guard.get("knee_angle_max", 65.0))
        self.deep_squat_hip_max = float(guard.get("hip_angle_max", 60.0))
        self.deep_squat_height_max = float(guard.get("height_ratio_max", 1.7))
        self.deep_squat_torso_max = float(guard.get("torso_angle_max", 55.0))
        self.deep_squat_valid_min = float(guard.get("valid_ratio_min", 0.85))

        self.model = model_from_checkpoint(checkpoint).to(self.device).eval()
        self.tracker = TCNFeatureTracker(
            self.window_size, max_track_age=max_track_age)

    @classmethod
    def from_config(cls, config, weights_path):
        return cls(
            weights_path=weights_path,
            device=config.get("device", "cuda"),
            confidence_threshold=config.get("confidence_threshold", 0.4),
            min_history=config.get("min_history", 6),
            max_track_age=config.get("max_track_age", 15),
            max_stale_features=config.get("max_stale_features", 3),
            deep_squat_guard=config.get("deep_squat_guard"),
        )

    @staticmethod
    def _extract_feature(kpts_3d, down_axis):
        if kpts_3d is None:
            return None
        corrected = sanitize_torso_depth(kpts_3d)
        return features_from_coco(corrected, down_axis)

    def _deep_squat_confidence(self, feature):
        if not self.deep_squat_guard_enabled or feature is None:
            return None
        geometry = feature[-10:]
        knee_angle = float(geometry[:2].mean()) * 180.0
        hip_angle = float(geometry[2:4].mean()) * 180.0
        torso_angle = float(geometry[4]) * 90.0
        height_ratio = float(geometry[5]) * 5.0
        valid_ratio = float(geometry[8])
        if not (
                knee_angle <= self.deep_squat_knee_max and
                hip_angle <= self.deep_squat_hip_max and
                height_ratio <= self.deep_squat_height_max and
                torso_angle <= self.deep_squat_torso_max and
                valid_ratio >= self.deep_squat_valid_min):
            return None
        knee_strength = np.clip(
            (self.deep_squat_knee_max - knee_angle) / 30.0, 0.0, 1.0)
        hip_strength = np.clip(
            (self.deep_squat_hip_max - hip_angle) / 35.0, 0.0, 1.0)
        height_strength = np.clip(
            (self.deep_squat_height_max - height_ratio) / 0.5, 0.0, 1.0)
        return float(0.65 + 0.25 * min(
            knee_strength, hip_strength, height_strength))

    def classify_batch(self, detections, down_axis=None, debug=False):
        if down_axis is None:
            down_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        boxes = [item["bbox"] for item in detections]
        features = [
            self._extract_feature(item.get("kpts_3d"), down_axis)
            for item in detections
        ]
        external_track_ids = (
            [item["track_id"] for item in detections]
            if detections and all("track_id" in item for item in detections)
            else None
        )
        tracked = self.tracker.update(
            boxes, features, external_track_ids=external_track_ids)
        ready_indices = [
            index for index, item in enumerate(tracked)
            if item["window"] is not None and
            item["history_size"] >= self.min_history and
            item["stale_count"] <= self.max_stale_features
        ]

        probability_rows = {}
        if ready_indices:
            windows = np.stack([
                tracked[index]["window"] for index in ready_indices])
            tensor = torch.from_numpy(windows).to(self.device)
            with torch.inference_mode():
                probabilities = torch.softmax(self.model(tensor), dim=1)
            for row, detection_index in enumerate(ready_indices):
                probability_rows[detection_index] = probabilities[row]

        results = []
        for index, tracked_item in enumerate(tracked):
            probabilities = probability_rows.get(index)
            feature = tracked_item["current_feature"]
            if feature is None and tracked_item["window"] is not None:
                feature = tracked_item["window"][-1]
            raw_label_name = None
            raw_score = 0.0
            geometry_override = None

            if probabilities is None:
                label_id, label_name, score = 5, "unknown", 0.0
            else:
                top_index = int(probabilities.argmax())
                top_score = float(probabilities[top_index])
                raw_label_name = self.pose_classes[top_index]
                raw_score = top_score
                if top_score < self.confidence_threshold:
                    label_id, label_name, score = 5, "unknown", top_score
                else:
                    label_id = top_index
                    label_name = raw_label_name
                    score = top_score

                squat_confidence = self._deep_squat_confidence(feature)
                if squat_confidence is not None and label_name != "squatting":
                    label_id = self.pose_classes.index("squatting")
                    label_name = "squatting"
                    score = squat_confidence
                    geometry_override = "deep_squat"

            details = {
                "track_id": tracked_item["track_id"],
                "history_size": tracked_item["history_size"],
                "warmup": tracked_item["history_size"] < self.min_history,
                "stale_feature": tracked_item["stale_feature"],
                "stale_count": tracked_item["stale_count"],
            }
            if geometry_override is not None:
                details.update({
                    "geometry_override": geometry_override,
                    "raw_label_name": raw_label_name,
                    "raw_score": round(raw_score, 4),
                })
            if debug and feature is not None:
                geometry = feature[-10:]
                details.update({
                    "knee_angle": round(float(geometry[:2].mean()) * 180.0, 1),
                    "hip_angle": round(float(geometry[2:4].mean()) * 180.0, 1),
                    "torso_angle": round(float(geometry[4]) * 90.0, 1),
                    "height_ratio": round(float(geometry[5]) * 5.0, 2),
                    "width_ratio": round(float(geometry[6]) * 3.0, 2),
                    "valid_ratio": round(float(geometry[8]), 2),
                })
            results.append({
                "label_id": label_id,
                "label_name": label_name,
                "score": score,
                "track_id": tracked_item["track_id"],
                "features": details,
            })
        return results
