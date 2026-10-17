from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def parse_camera_source(value: str) -> int | str:
    stripped = value.strip()
    return int(stripped) if stripped.lstrip("-").isdigit() else stripped


@dataclass
class Camera:
    source: int | str
    width: int = 1280
    height: int = 720
    fps: int = 30

    def __post_init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "Camera":
        self._capture = cv2.VideoCapture(self.source)
        if self.width > 0:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"Cannot open camera source: {self.source!r}")
        return self

    def read(self):
        if self._capture is None:
            raise RuntimeError("Camera is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read camera source: {self.source!r}")
        return frame

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def draw_face(frame, bbox: tuple[int, int, int, int], label: str, color) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if not label.isascii():
        _draw_unicode_label(frame, label, (x1, y1), color)
        return

    text_size, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    )
    text_y = max(y1, text_size[1] + baseline + 4)
    cv2.rectangle(
        frame,
        (x1, text_y - text_size[1] - baseline - 4),
        (x1 + text_size[0] + 6, text_y),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 3, text_y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


@lru_cache(maxsize=1)
def _unicode_font():
    try:
        from PIL import ImageFont
    except ImportError as error:
        raise RuntimeError("Pillow is required to display non-ASCII names") from error

    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), 22)
    return ImageFont.load_default()


def _draw_unicode_label(frame, label: str, origin: tuple[int, int], color) -> None:
    from PIL import Image, ImageDraw

    x, y = origin
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = _unicode_font()
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    top = max(0, y - text_height - 10)
    draw.rectangle(
        (x, top, x + text_width + 8, top + text_height + 8),
        fill=tuple(int(channel) for channel in color),
    )
    draw.text((x + 4, top + 2), label, font=font, fill=(255, 255, 255))
    frame[:] = np.asarray(image)
