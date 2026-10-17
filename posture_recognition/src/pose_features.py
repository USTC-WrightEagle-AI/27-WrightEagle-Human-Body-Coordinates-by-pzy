#!/usr/bin/env python3
"""COCO17 3D keypoints to the 62-dimensional TCN feature vector."""

import math

import numpy as np


COCO_JOINT_INDICES = np.array([
    0,
    5, 6,
    7, 8,
    9, 10,
    11, 12,
    13, 14,
    15, 16,
], dtype=np.int64)

HEAD = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 1, 2
LEFT_ELBOW, RIGHT_ELBOW = 3, 4
LEFT_WRIST, RIGHT_WRIST = 5, 6
LEFT_HIP, RIGHT_HIP = 7, 8
LEFT_KNEE, RIGHT_KNEE = 9, 10
LEFT_ANKLE, RIGHT_ANKLE = 11, 12

NUM_JOINTS = 13
GEOMETRY_DIM = 10
INPUT_DIM = NUM_JOINTS * 4 + GEOMETRY_DIM


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return np.asarray(vector, dtype=np.float32) / norm


def angle_3d(p1, p2, p3):
    first = np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32)
    second = np.asarray(p3, dtype=np.float32) - np.asarray(p2, dtype=np.float32)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-8:
        return None
    cosine = float(np.dot(first, second) / denominator)
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))


def extract_3d_features(points, valid, down_axis):
    points = np.asarray(points, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.float32).reshape(NUM_JOINTS)
    if points.shape != (NUM_JOINTS, 3):
        raise ValueError(f"points must have shape ({NUM_JOINTS}, 3)")
    required = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    if any(valid[index] <= 0 for index in required):
        return None

    down = _unit(down_axis)
    if down is None:
        raise ValueError("down_axis cannot be zero")
    up = -down
    shoulder_mid = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2.0
    hip_mid = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2.0
    torso = shoulder_mid - hip_mid
    torso_len = float(np.linalg.norm(torso))
    if torso_len < 1e-5:
        return None

    right = points[RIGHT_SHOULDER] - points[LEFT_SHOULDER]
    right = right - np.dot(right, up) * up
    right = _unit(right)
    if right is None:
        return None
    forward = _unit(np.cross(right, up))
    if forward is None:
        return None
    basis = np.stack([right, up, forward], axis=1)

    local = ((points - hip_mid) @ basis) / torso_len
    local[valid <= 0] = 0.0

    def joint_angle(a, b, c):
        if min(valid[a], valid[b], valid[c]) <= 0:
            return 180.0
        value = angle_3d(points[a], points[b], points[c])
        return 180.0 if value is None else value

    left_knee = joint_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    right_knee = joint_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    left_hip = joint_angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
    right_hip = joint_angle(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
    torso_cosine = abs(float(np.dot(torso / torso_len, up)))
    torso_tilt = math.degrees(math.acos(np.clip(torso_cosine, 0.0, 1.0)))

    valid_points = points[valid > 0]
    vertical = valid_points @ up
    body_height_ratio = float(vertical.max() - vertical.min()) / torso_len
    shoulder_width = float(np.linalg.norm(
        points[RIGHT_SHOULDER] - points[LEFT_SHOULDER])) / torso_len
    hip_width = float(np.linalg.norm(
        points[RIGHT_HIP] - points[LEFT_HIP])) / torso_len
    body_width_ratio = max(shoulder_width, hip_width)

    knee_indices = [index for index in (LEFT_KNEE, RIGHT_KNEE)
                    if valid[index] > 0]
    if knee_indices:
        knee_mid = np.mean(points[knee_indices], axis=0)
        hip_above_knee = float(np.dot(hip_mid - knee_mid, up)) / torso_len
    else:
        hip_above_knee = 0.0
    if valid[LEFT_ANKLE] > 0 and valid[RIGHT_ANKLE] > 0:
        ankle_height_difference = abs(float(np.dot(
            points[LEFT_ANKLE] - points[RIGHT_ANKLE], up))) / torso_len
    else:
        ankle_height_difference = 0.0

    geometry = np.array([
        left_knee / 180.0,
        right_knee / 180.0,
        left_hip / 180.0,
        right_hip / 180.0,
        torso_tilt / 90.0,
        np.clip(body_height_ratio / 5.0, 0.0, 1.5),
        np.clip(body_width_ratio / 3.0, 0.0, 1.5),
        np.clip(hip_above_knee / 2.0, -1.0, 1.0),
        float(valid.sum()) / NUM_JOINTS,
        np.clip(ankle_height_difference / 2.0, 0.0, 1.0),
    ], dtype=np.float32)
    return np.concatenate([
        local.reshape(-1).astype(np.float32),
        valid.astype(np.float32),
        geometry,
    ])


def features_from_coco(kpts_3d, down_axis):
    selected = np.asarray(kpts_3d, dtype=np.float32)[COCO_JOINT_INDICES]
    return extract_3d_features(
        selected[:, :3], selected[:, 3] > 0, down_axis)
