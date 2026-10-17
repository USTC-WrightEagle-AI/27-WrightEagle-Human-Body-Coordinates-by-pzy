#!/usr/bin/env python3
"""Length-prefixed InsightFace worker for the posture process."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from face_robot.database import FaceDatabase
from face_robot.engine import FaceEngine


def exact(stream, size):
    data = bytearray()
    while len(data) < size:
        part = stream.read(size - len(data))
        if not part:
            raise EOFError
        data.extend(part)
    return bytes(data)


def _valid_keypoint(keypoints, index, confidence_threshold):
    if keypoints.ndim != 2 or keypoints.shape[0] <= index or keypoints.shape[1] < 3:
        return False
    point = keypoints[index]
    return bool(np.all(np.isfinite(point[:3])) and
                point[2] >= confidence_threshold and
                not np.allclose(point[:2], 0.0))


def virtual_face_box(keypoints, person_box, image_shape,
                     confidence_threshold=0.35):
    """Estimate a face-sized box from COCO keypoints, with robust fallbacks."""
    keypoints = np.asarray(keypoints, dtype=float)
    x1, y1, x2, y2 = [float(value) for value in person_box]
    person_width = max(1.0, x2 - x1)
    person_height = max(1.0, y2 - y1)

    valid = lambda index: _valid_keypoint(  # noqa: E731
        keypoints, index, confidence_threshold)
    if valid(3) and valid(4):
        center = (keypoints[3, :2] + keypoints[4, :2]) * 0.5
        face_width = 1.5 * float(np.linalg.norm(
            keypoints[3, :2] - keypoints[4, :2]))
        face_height = 1.2 * face_width
    elif valid(1) and valid(2):
        eye_center = (keypoints[1, :2] + keypoints[2, :2]) * 0.5
        face_width = 2.7 * float(np.linalg.norm(
            keypoints[1, :2] - keypoints[2, :2]))
        face_height = 1.2 * face_width
        center = eye_center + np.array([0.0, 0.16 * face_height])
    elif valid(5) and valid(6):
        shoulder_center = (keypoints[5, :2] + keypoints[6, :2]) * 0.5
        shoulder_width = float(np.linalg.norm(
            keypoints[5, :2] - keypoints[6, :2]))
        face_width = 0.42 * shoulder_width
        face_height = 1.2 * face_width
        center_x = keypoints[0, 0] if valid(0) else shoulder_center[0]
        center_y = (keypoints[0, 1] - 0.04 * face_height
                    if valid(0) else y1 + 0.5 * face_height)
        center = np.array([center_x, center_y])
    else:
        face_width = 0.36 * person_width
        face_height = max(1.2 * face_width, 0.16 * person_height)
        face_height = min(face_height, 0.28 * person_height)
        center = np.array([(x1 + x2) * 0.5, y1 + 0.5 * face_height])

    face_width = float(np.clip(face_width, 12.0, person_width * 0.8))
    face_height = float(np.clip(face_height, 14.0, person_height * 0.35))
    image_height, image_width = image_shape[:2]
    virtual = np.array([
        center[0] - face_width * 0.5,
        center[1] - face_height * 0.5,
        center[0] + face_width * 0.5,
        center[1] + face_height * 0.5,
    ])
    virtual[[0, 2]] = np.clip(virtual[[0, 2]], 0, image_width)
    virtual[[1, 3]] = np.clip(virtual[[1, 3]], 0, image_height)
    if virtual[2] - virtual[0] < 4 or virtual[3] - virtual[1] < 4:
        return None
    return virtual


def _box_metrics(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    intersection_width = max(0.0, min(first[2], second[2]) -
                             max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) -
                              max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    iou = intersection / union if union > 0 else 0.0
    modified_iou = intersection / min(first_area, second_area) \
        if min(first_area, second_area) > 0 else 0.0
    return iou, modified_iou, first_area, second_area


def _association_score(face_box, virtual_box, person_box):
    face_box = np.asarray(face_box, dtype=float)
    virtual_box = np.asarray(virtual_box, dtype=float)
    x1, y1, x2, y2 = [float(value) for value in person_box]
    person_width = max(1.0, x2 - x1)
    person_height = max(1.0, y2 - y1)
    face_center = np.array([
        (face_box[0] + face_box[2]) * 0.5,
        (face_box[1] + face_box[3]) * 0.5,
    ])
    if not (x1 - 0.08 * person_width <= face_center[0] <=
            x2 + 0.08 * person_width and
            y1 - 0.05 * person_height <= face_center[1] <=
            y1 + 0.45 * person_height):
        return 0.0, 0.0

    iou, modified_iou, face_area, virtual_area = _box_metrics(
        face_box, virtual_box)
    if iou <= 0.0:
        return 0.0, iou
    virtual_center = np.array([
        (virtual_box[0] + virtual_box[2]) * 0.5,
        (virtual_box[1] + virtual_box[3]) * 0.5,
    ])
    diagonal = max(1.0, float(np.linalg.norm(
        virtual_box[2:] - virtual_box[:2])))
    center_similarity = max(
        0.0, 1.0 - float(np.linalg.norm(face_center - virtual_center)) / diagonal)
    scale_similarity = min(face_area, virtual_area) / max(face_area, virtual_area)
    score = (0.55 * iou + 0.15 * modified_iou +
             0.20 * center_similarity + 0.10 * scale_similarity)
    return float(score), float(iou)


def associate_faces(faces, track_ids, boxes, keypoints, image_shape,
                    confidence_threshold=0.35, min_iou=0.08,
                    min_score=0.25):
    """Globally associate detected faces with pose tracks one-to-one."""
    if not faces or not track_ids:
        return []
    virtual_boxes = [
        virtual_face_box(points, box, image_shape, confidence_threshold)
        for points, box in zip(keypoints, boxes)
    ]
    scores = np.zeros((len(faces), len(track_ids)), dtype=np.float32)
    ious = np.zeros_like(scores)
    for face_index, face in enumerate(faces):
        for track_index, virtual_box in enumerate(virtual_boxes):
            if virtual_box is None:
                continue
            scores[face_index, track_index], ious[face_index, track_index] = \
                _association_score(face.bbox, virtual_box, boxes[track_index])

    face_indices, track_indices = linear_sum_assignment(-scores)
    assignments = []
    for face_index, track_index in zip(face_indices, track_indices):
        if (ious[face_index, track_index] < min_iou or
                scores[face_index, track_index] < min_score):
            continue
        assignments.append((
            int(track_ids[track_index]), faces[face_index],
            virtual_boxes[track_index], float(scores[face_index, track_index]),
        ))
    return assignments


def recognize_tracks(engine, database, frame, track_ids, boxes, keypoints,
                     threshold, keypoint_confidence=0.35,
                     min_assignment_iou=0.08, min_assignment_score=0.25):
    """Detect faces once and bind them globally to YOLO pose tracks."""
    with contextlib.redirect_stdout(sys.stderr):
        faces = engine.detect(frame, max_faces=0)
    assignments = associate_faces(
        faces, track_ids, boxes, keypoints, frame.shape,
        confidence_threshold=keypoint_confidence,
        min_iou=min_assignment_iou, min_score=min_assignment_score)
    matches = []
    for track_id, face, virtual_box, assignment_score in assignments:
        match = database.match(face.embedding, threshold)
        matches.append({
            "track_id": track_id,
            "name": match.name,
            "known": match.known,
            "score": round(match.score, 4),
            "face_bbox": [int(value) for value in face.bbox],
            "virtual_face_bbox": [round(float(value), 1)
                                  for value in virtual_box],
            "assignment_score": round(assignment_score, 4),
        })
    return matches


def emit(payload):
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("!I", len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"),
                        default="auto")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--keypoint-confidence", type=float, default=0.35)
    parser.add_argument("--min-assignment-iou", type=float, default=0.08)
    parser.add_argument("--min-assignment-score", type=float, default=0.25)
    args = parser.parse_args()

    database = FaceDatabase(args.database)
    with contextlib.redirect_stdout(sys.stderr):
        engine = FaceEngine(
            provider=args.provider,
            detection_size=(args.det_size, args.det_size))

    while True:
        try:
            size = struct.unpack("!I", exact(sys.stdin.buffer, 4))[0]
        except EOFError:
            return
        if not 0 < size < 64 * 1024 * 1024:
            raise ValueError("invalid face-recognition request size")
        with np.load(io.BytesIO(exact(sys.stdin.buffer, size)),
                     allow_pickle=False) as packet:
            sequence = int(packet["sequence"])
            frame = packet["frame"]
            track_ids = packet["track_ids"].tolist()
            boxes = packet["boxes"].tolist()
            keypoints = packet["keypoints"]
        matches = recognize_tracks(
            engine, database, frame, track_ids, boxes, keypoints,
            args.threshold, keypoint_confidence=args.keypoint_confidence,
            min_assignment_iou=args.min_assignment_iou,
            min_assignment_score=args.min_assignment_score)
        emit({"sequence": sequence, "matches": matches})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
