#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from face_robot.camera import draw_face
from face_robot.head_camera import add_camera_arguments, open_camera
from face_robot.database import FaceDatabase, l2_normalize
from face_robot.depth import measure_face_depths, select_nearest_face
from face_robot.engine import FaceEngine


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "faces.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enroll one person using RetinaFace and ArcFace-R50."
    )
    parser.add_argument("name", help="Name to store in the local face database")
    add_camera_arguments(parser)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=3, metavar="FRAMES")
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=65,
        metavar="PIXELS",
        help="Minimum face-box side length accepted for enrollment (default: 65)",
    )
    parser.add_argument("--timeout", type=float, default=90.0, metavar="SECONDS")
    parser.add_argument("--det-size", type=int, default=1280,
                        help="Face detector input size (default: 1280; use 640 for lower compute cost)")
    parser.add_argument(
        "--depth-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enroll the nearest face using aligned RealSense depth (default: enabled)",
    )
    parser.add_argument("--min-depth", type=float, default=0.15, metavar="METERS")
    parser.add_argument("--max-depth", type=float, default=4.0, metavar="METERS")
    parser.add_argument(
        "--min-depth-gap",
        type=float,
        default=0.15,
        metavar="METERS",
        help="Required gap between the two nearest faces (default: 0.15)",
    )
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if args.sample_every < 1:
        raise ValueError("--sample-every must be at least 1")
    if args.min_face_size < 1:
        raise ValueError("--min-face-size must be at least 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.min_depth <= 0 or args.max_depth <= args.min_depth:
        raise ValueError("Depth range must satisfy 0 < --min-depth < --max-depth")
    if args.min_depth_gap < 0:
        raise ValueError("--min-depth-gap must not be negative")


def enroll(args: argparse.Namespace) -> None:
    validate_args(args)
    camera_input = open_camera(args, with_depth=args.depth_filter)
    database = FaceDatabase(args.database)
    if database.contains(args.name) and not args.replace:
        raise ValueError(
            f"{args.name!r} is already enrolled; use --replace to update it"
        )

    print("Loading RetinaFace and ArcFace-R50 models...")
    engine = FaceEngine(
        model_root=args.model_root,
        provider=args.provider,
        detection_size=(args.det_size, args.det_size),
    )
    print(f"Inference provider: {engine.provider}")
    if args.depth_filter:
        print("Depth filtering enabled: the nearest detected person will be enrolled.")
    else:
        print("Keep one person in view and slowly turn the head.")
    print("Press q to cancel.")

    samples: list[np.ndarray] = []
    frame_number = 0
    previous_status = ""
    started_at = time.monotonic()
    try:
        with camera_input as camera:
            while len(samples) < args.samples:
                if time.monotonic() - started_at > args.timeout:
                    raise TimeoutError(
                        f"Enrollment timed out after {args.timeout:.0f} seconds"
                    )

                camera_frame = camera.read()
                if args.depth_filter:
                    frame, depth_m = camera_frame
                else:
                    frame = camera_frame
                    depth_m = None
                frame_number += 1
                faces = engine.detect(frame, max_faces=0)
                status = ""
                color = (0, 165, 255)

                if not faces:
                    status = "No face detected"
                elif depth_m is not None:
                    measurements = measure_face_depths(
                        depth_m,
                        [face.bbox for face in faces],
                        min_depth_m=args.min_depth,
                        max_depth_m=args.max_depth,
                    )
                    selection = select_nearest_face(
                        measurements, min_depth_gap_m=args.min_depth_gap
                    )
                    if selection.reason == "invalid_depth":
                        status = "No face with valid depth - waiting"
                        for face, measurement in zip(faces, measurements):
                            label = (
                                f"Depth {measurement.distance_m:.2f}m"
                                if measurement.distance_m is not None
                                else "Ignored: no valid depth"
                            )
                            draw_face(frame, face.bbox, label, (0, 165, 255))
                        face = None
                    else:
                        if selection.reason == "ambiguous_depth":
                            status = "Nearest faces are too close in depth"
                            face = None
                        else:
                            nearest_index = selection.index
                            face = faces[nearest_index]
                            nearest_depth = measurements[nearest_index].distance_m
                            status = f"Nearest person {nearest_depth:.2f}m"

                        for index, (candidate, measurement) in enumerate(
                            zip(faces, measurements)
                        ):
                            if measurement.distance_m is None:
                                label = "Ignored: no valid depth"
                                box_color = (100, 100, 100)
                            elif index == selection.index:
                                label = f"Nearest {measurement.distance_m:.2f}m"
                                box_color = (0, 200, 0)
                            else:
                                label = f"Ignored {measurement.distance_m:.2f}m"
                                box_color = (100, 100, 100)
                            draw_face(frame, candidate.bbox, label, box_color)

                    if face is not None:
                        if min(face.width, face.height) < args.min_face_size:
                            status = "Nearest person: move closer"
                        else:
                            status = (
                                f"Collecting nearest person "
                                f"{len(samples)}/{args.samples}"
                            )
                            if frame_number % args.sample_every == 0:
                                samples.append(face.embedding.copy())
                                status = (
                                    f"Collected nearest person "
                                    f"{len(samples)}/{args.samples}"
                                )
                elif len(faces) > 1:
                    status = "Multiple faces detected - keep only one person in view"
                    for face in faces:
                        draw_face(frame, face.bbox, "Extra face", (0, 0, 255))
                else:
                    face = faces[0]
                    if min(face.width, face.height) < args.min_face_size:
                        status = "Move closer to the camera"
                    else:
                        status = f"Collecting {len(samples)}/{args.samples}"
                        color = (0, 200, 0)
                        if frame_number % args.sample_every == 0:
                            samples.append(face.embedding.copy())
                            status = f"Collected {len(samples)}/{args.samples}"
                    draw_face(frame, face.bbox, status, color)

                if status != previous_status:
                    print(status)
                    previous_status = status

                if not args.headless:
                    cv2.putText(
                        frame,
                        status,
                        (16, 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        "Press q to cancel",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Remember person", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt
    finally:
        if not args.headless:
            cv2.destroyAllWindows()

    prototype = l2_normalize(np.mean(np.stack(samples), axis=0))
    database.upsert(args.name, prototype, replace=args.replace)
    print(
        f"Saved {args.name!r} from {len(samples)} samples to "
        f"{args.database.expanduser()}"
    )


def main() -> int:
    args = build_parser().parse_args()
    try:
        enroll(args)
        return 0
    except KeyboardInterrupt:
        print("Enrollment cancelled.")
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
