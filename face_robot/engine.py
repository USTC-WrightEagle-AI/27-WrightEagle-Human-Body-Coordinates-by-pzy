from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .database import l2_normalize


@dataclass(frozen=True)
class DetectedFace:
    bbox: tuple[int, int, int, int]
    detection_score: float
    embedding: np.ndarray

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        return self.width * self.height


class FaceEngine:
    """InsightFace buffalo_l: RetinaFace detection plus ArcFace-R50."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        model_root: str | Path | None = None,
        provider: str = "auto",
        detection_size: tuple[int, int] = (640, 640),
        detection_threshold: float = 0.6,
    ) -> None:
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise RuntimeError(
                "Missing inference dependencies. Install requirements.txt in a "
                "Python 3.10 environment."
            ) from error

        preload_dlls = getattr(ort, "preload_dlls", None)
        if preload_dlls is not None:
            preload_dlls()

        provider = provider.lower()
        if provider not in {"auto", "cuda", "cpu"}:
            raise ValueError("provider must be one of: auto, cuda, cpu")

        available = set(ort.get_available_providers())
        cuda_available = "CUDAExecutionProvider" in available
        if provider == "cuda" and not cuda_available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. Check onnxruntime-gpu and "
                "run this process outside a GPU-isolated container."
            )

        use_cuda = cuda_available and provider != "cpu"
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda
            else ["CPUExecutionProvider"]
        )

        options: dict[str, object] = {
            "name": model_name,
            "allowed_modules": ["detection", "recognition"],
            "providers": providers,
        }
        if model_root is not None:
            options["root"] = str(Path(model_root).expanduser())

        self._analysis = FaceAnalysis(**options)
        self._analysis.prepare(
            ctx_id=0 if use_cuda else -1,
            det_thresh=detection_threshold,
            det_size=detection_size,
        )
        self.provider = "cuda" if use_cuda else "cpu"

    def detect(self, frame: np.ndarray, max_faces: int = 0) -> list[DetectedFace]:
        raw_faces = self._analysis.get(frame, max_num=max_faces)
        detected: list[DetectedFace] = []
        for face in raw_faces:
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue

            bbox = np.asarray(face.bbox, dtype=np.float32).round().astype(int)
            detected.append(
                DetectedFace(
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    detection_score=float(face.det_score),
                    embedding=l2_normalize(np.asarray(embedding)),
                )
            )
        return detected
