from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("Face embedding must not be a zero vector")
    return value / norm


@dataclass(frozen=True)
class MatchResult:
    name: str
    score: float
    known: bool


class FaceDatabase:
    """Small, atomic NPZ-backed database with one prototype per person."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.names: list[str] = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.names = []
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            return

        with np.load(self.path, allow_pickle=False) as data:
            if "names" not in data or "embeddings" not in data:
                raise ValueError(f"Invalid face database: {self.path}")
            names = np.asarray(data["names"]).astype(str).tolist()
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(names):
            raise ValueError(f"Inconsistent face database: {self.path}")
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate names in face database: {self.path}")

        if len(names):
            embeddings = np.stack([l2_normalize(row) for row in embeddings])
        self.names = names
        self.embeddings = embeddings

    def contains(self, name: str) -> bool:
        return name.strip() in self.names

    def remove(self, name: str) -> None:
        clean_name = name.strip()
        if clean_name not in self.names:
            raise KeyError(f"Person is not enrolled: {clean_name!r}")

        index = self.names.index(clean_name)
        self.names.pop(index)
        self.embeddings = np.delete(self.embeddings, index, axis=0)
        self._save_atomic()

    def upsert(self, name: str, embedding: np.ndarray, replace: bool = False) -> None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Person name must not be empty")
        if "\n" in clean_name or "\r" in clean_name:
            raise ValueError("Person name must be a single line")

        normalized = l2_normalize(embedding)
        if self.embeddings.size and normalized.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Embedding size {normalized.shape[0]} does not match database size "
                f"{self.embeddings.shape[1]}"
            )

        if clean_name in self.names:
            if not replace:
                raise ValueError(
                    f"{clean_name!r} is already enrolled; pass --replace to update it"
                )
            index = self.names.index(clean_name)
            self.embeddings[index] = normalized
        else:
            self.names.append(clean_name)
            if self.embeddings.size:
                self.embeddings = np.vstack((self.embeddings, normalized))
            else:
                self.embeddings = normalized.reshape(1, -1)

        self._save_atomic()

    def match(self, embedding: np.ndarray, threshold: float) -> MatchResult:
        if not self.names:
            return MatchResult(name="Unknown", score=0.0, known=False)

        normalized = l2_normalize(embedding)
        if normalized.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Embedding size {normalized.shape[0]} does not match database size "
                f"{self.embeddings.shape[1]}"
            )

        scores = self.embeddings @ normalized
        index = int(np.argmax(scores))
        score = float(scores[index])
        known = score >= threshold
        return MatchResult(
            name=self.names[index] if known else "Unknown",
            score=score,
            known=known,
        )

    def _save_atomic(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(
                    temporary,
                    names=np.asarray(self.names, dtype=np.str_),
                    embeddings=self.embeddings.astype(np.float32),
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
