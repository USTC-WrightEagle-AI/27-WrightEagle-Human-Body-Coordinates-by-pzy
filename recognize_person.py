#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2

from face_robot.camera import draw_face
from face_robot.head_camera import add_camera_arguments, open_camera
from face_robot.database import FaceDatabase, MatchResult
from face_robot.engine import DetectedFace, FaceEngine


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "faces.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize people using RetinaFace and ArcFace-R50."
    )
    add_camera_arguments(parser)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--history", type=int, default=5, metavar="FRAMES")
    parser.add_argument("--confirm-frames", type=int, default=3)
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--det-size", type=int, default=1280,
                        help="Face detector input size (default: 1280; use 640 for lower compute cost)")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit recognition changes as JSON lines for robot integration",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.history < 1:
        raise ValueError("--history must be at least 1")
    if not 1 <= args.confirm_frames <= args.history:
        raise ValueError("--confirm-frames must be between 1 and --history")
    if args.max_faces < 1:
        raise ValueError("--max-faces must be at least 1")


def emit_result(result: MatchResult, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "name": result.name,
                    "known": result.known,
                    "score": round(result.score, 4),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        print(f"Person: {result.name} (similarity={result.score:.3f})", flush=True)


def recognize(args: argparse.Namespace) -> None:
    validate_args(args)
    camera_input = open_camera(args)
    database = FaceDatabase(args.database)
    if not database.names:
        raise ValueError(
            f"Face database is empty: {args.database}. Run remember_person.py first."
        )

    print("Loading RetinaFace and ArcFace-R50 models...", file=sys.stderr)
    engine = FaceEngine(
        model_root=args.model_root,
        provider=args.provider,
        detection_size=(args.det_size, args.det_size),
    )
    print(
        f"Inference provider: {engine.provider}; enrolled people: "
        f"{', '.join(database.names)}",
        file=sys.stderr,
    )
    print("Press q to stop.", file=sys.stderr)

    history: deque[tuple[str, float]] = deque(maxlen=args.history)
    last_emitted: str | None = None

    try:
        with camera_input as camera:
            while True:
                frame = camera.read()
                faces = engine.detect(frame, max_faces=args.max_faces)
                matches: list[tuple[DetectedFace, MatchResult]] = []

                for face in faces:
                    match = database.match(face.embedding, args.threshold)
                    matches.append((face, match))
                    color = (0, 180, 0) if match.known else (0, 0, 255)
                    draw_face(
                        frame,
                        face.bbox,
                        f"{match.name} {match.score:.2f}",
                        color,
                    )

                if matches:
                    primary_face, primary_match = max(
                        matches, key=lambda item: item[0].area
                    )
                    history.append((primary_match.name, primary_match.score))
                    counts = Counter(name for name, _ in history)
                    stable_name, stable_count = counts.most_common(1)[0]
                    if stable_count >= args.confirm_frames and stable_name != last_emitted:
                        stable_scores = [
                            score for name, score in history if name == stable_name
                        ]
                        stable_score = sum(stable_scores) / len(stable_scores)
                        stable_result = MatchResult(
                            name=stable_name,
                            score=stable_score,
                            known=stable_name != "Unknown",
                        )
                        emit_result(stable_result, args.json)
                        last_emitted = stable_name
                else:
                    history.clear()
                    last_emitted = None
                    cv2.putText(
                        frame,
                        "No face",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if not args.headless:
                    cv2.imshow("Recognize person", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        if not args.headless:
            cv2.destroyAllWindows()


def main() -> int:
    args = build_parser().parse_args()
    try:
        recognize(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
