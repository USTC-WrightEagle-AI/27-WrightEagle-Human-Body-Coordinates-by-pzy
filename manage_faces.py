#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from face_robot.database import FaceDatabase


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "faces.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the local face database.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List enrolled people")

    delete_parser = subparsers.add_parser("delete", help="Delete one enrolled person")
    delete_parser.add_argument("name", help="Name to delete")
    delete_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        database = FaceDatabase(args.database)
        if args.command == "list":
            if database.names:
                for name in database.names:
                    print(name)
            else:
                print("Face database is empty.")
            return 0

        if args.command == "delete":
            if args.name.strip() not in database.names:
                raise KeyError(f"Person is not enrolled: {args.name!r}")
            if not args.yes:
                answer = input(
                    f"Delete face information for {args.name!r}? [y/N] "
                )
                if answer.strip().lower() not in {"y", "yes"}:
                    print("Deletion cancelled.")
                    return 0
            database.remove(args.name)
            print(f"Deleted face information for {args.name!r}.")
            return 0

        raise RuntimeError(f"Unknown command: {args.command}")
    except (KeyError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
