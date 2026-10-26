from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FaceDepth:
    bbox: tuple[int, int, int, int]
    distance_m: float | None


@dataclass(frozen=True)
class NearestFaceSelection:
    index: int | None
    reason: str | None


def estimate_face_depth(
    depth_m: np.ndarray,
    bbox: tuple[int, int, int, int],
    min_depth_m: float = 0.15,
    max_depth_m: float = 4.0,
    min_valid_pixels: int = 20,
) -> float | None:
    """Return the median depth in the central face region."""
    if depth_m.ndim != 2:
        raise ValueError("Aligned depth image must be a two-dimensional array")

    image_height, image_width = depth_m.shape
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted(
        (int(np.clip(x1, 0, image_width)), int(np.clip(x2, 0, image_width)))
    )
    y1, y2 = sorted(
        (int(np.clip(y1, 0, image_height)), int(np.clip(y2, 0, image_height)))
    )
    if x2 <= x1 or y2 <= y1:
        return None

    # The central region avoids background pixels around the face-box edges.
    width = x2 - x1
    height = y2 - y1
    inner_x1 = x1 + int(width * 0.25)
    inner_x2 = x2 - int(width * 0.25)
    inner_y1 = y1 + int(height * 0.25)
    inner_y2 = y2 - int(height * 0.20)
    region = depth_m[inner_y1:inner_y2, inner_x1:inner_x2]
    valid = region[
        np.isfinite(region)
        & (region >= min_depth_m)
        & (region <= max_depth_m)
    ]
    if valid.size < min_valid_pixels:
        return None
    return float(np.median(valid))


def measure_face_depths(
    depth_m: np.ndarray,
    bboxes: Iterable[tuple[int, int, int, int]],
    min_depth_m: float = 0.15,
    max_depth_m: float = 4.0,
) -> list[FaceDepth]:
    return [
        FaceDepth(
            bbox=bbox,
            distance_m=estimate_face_depth(
                depth_m,
                bbox,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            ),
        )
        for bbox in bboxes
    ]


def select_nearest_face(
    measurements: list[FaceDepth], min_depth_gap_m: float = 0.15
) -> NearestFaceSelection:
    if min_depth_gap_m < 0:
        raise ValueError("Minimum depth gap must not be negative")
    if not measurements:
        return NearestFaceSelection(index=None, reason="no_faces")
    ordered = sorted(
        (float(measurement.distance_m), index)
        for index, measurement in enumerate(measurements)
        if measurement.distance_m is not None
    )
    if not ordered:
        return NearestFaceSelection(index=None, reason="invalid_depth")
    if len(ordered) > 1 and ordered[1][0] - ordered[0][0] < min_depth_gap_m:
        return NearestFaceSelection(index=None, reason="ambiguous_depth")
    return NearestFaceSelection(index=ordered[0][1], reason=None)


class RealSenseCamera:
    """Synchronized RealSense color and depth frames aligned to color."""

    def __init__(
        self,
        source: int | str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        serial: str | None = None,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self._rs = None
        self._pipeline = None
        self._align = None
        self._depth_scale = 0.001
        self.color_intrinsics = None

    def __enter__(self) -> "RealSenseCamera":
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError(
                "Depth filtering requires pyrealsense2. Install the updated "
                "requirements.txt in the robot-face environment."
            ) from error

        serial = self.serial or self._resolve_serial(rs)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps,
        )
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )

        try:
            profile = pipeline.start(config)
        except Exception as error:
            raise RuntimeError(
                f"Cannot start synchronized color/depth streams for RealSense "
                f"{serial}. Check that no ROS or V4L2 process is using it and "
                f"that {self.width}x{self.height}@{self.fps} is supported."
            ) from error

        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        self._rs = rs
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)

        try:
            self.color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
            # Discard initial auto-exposure frames.
            for _ in range(10):
                self._read_frames(timeout_ms=5000)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        color_frame, depth_frame = self._read_frames(timeout_ms=5000)
        color = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_m *= self._depth_scale
        return color, depth_m

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._align = None
        self._rs = None

    def _read_frames(self, timeout_ms: int):
        if self._pipeline is None or self._align is None:
            raise RuntimeError("RealSense camera is not open")
        frames = self._pipeline.wait_for_frames(timeout_ms)
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense returned incomplete color/depth frames")
        return color_frame, depth_frame

    def _resolve_serial(self, rs) -> str:
        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError("No RealSense camera was found")

        source_serial, source_usb_port = _video_source_identity(self.source)
        candidates: list[tuple[str, str, str]] = []
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            physical_port = (
                device.get_info(rs.camera_info.physical_port)
                if device.supports(rs.camera_info.physical_port)
                else ""
            )
            candidates.append((serial, name, physical_port))
            if source_serial and serial == source_serial:
                return serial
            if source_usb_port and source_usb_port in physical_port:
                return serial

        if len(candidates) == 1:
            return candidates[0][0]

        available = ", ".join(f"{name} ({serial})" for serial, name, _ in candidates)
        raise RuntimeError(
            f"Cannot map {self.source!r} to one RealSense device. Pass "
            f"--realsense-serial explicitly. Available devices: {available}"
        )


def _video_source_identity(source: int | str) -> tuple[str | None, str | None]:
    if not isinstance(source, str) or not source.startswith("/dev/video"):
        return None, None

    sysfs_device = Path("/sys/class/video4linux") / Path(source).name / "device"
    try:
        resolved = sysfs_device.resolve(strict=True)
    except OSError:
        return None, None

    for parent in (resolved, *resolved.parents):
        vendor_path = parent / "idVendor"
        if not vendor_path.exists():
            continue
        try:
            if vendor_path.read_text().strip().lower() != "8086":
                continue
            serial_path = parent / "serial"
            serial = serial_path.read_text().strip() if serial_path.exists() else None
            return serial, parent.name
        except OSError:
            return None, parent.name
    return None, None
