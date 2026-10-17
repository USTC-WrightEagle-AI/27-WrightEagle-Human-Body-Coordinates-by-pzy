"""Asynchronous face recognition and stable name-to-track binding."""
from __future__ import annotations

import io
import json
import queue
import struct
import subprocess
import threading
from collections import Counter, deque

import numpy as np


def transform_point(matrix, point):
    matrix = np.asarray(matrix, dtype=float)
    point = np.asarray(point, dtype=float).reshape(3)
    if matrix.shape != (4, 4):
        raise ValueError("transform matrix must be 4x4")
    result = matrix @ np.append(point, 1.0)
    if abs(result[3]) < 1e-12:
        raise ValueError("transformed homogeneous point has zero scale")
    return result[:3] / result[3]


class NameBinder:
    """Bind a confirmed face name to a ByteTrack ID until the track expires."""

    def __init__(self, confirmations=2, history_size=5, max_track_age=45):
        self.confirmations = max(1, int(confirmations))
        self.history_size = max(self.confirmations, int(history_size))
        self.max_track_age = max(1, int(max_track_age))
        self.frame_index = 0
        self.states = {}

    def update(self, active_track_ids, observations=()):
        self.frame_index += 1
        active = {int(track_id) for track_id in active_track_ids}
        for track_id in active:
            state = self.states.setdefault(track_id, {
                "history": deque(maxlen=self.history_size),
                "name": "Unknown",
                "known": False,
                "score": 0.0,
                "last_seen": self.frame_index,
            })
            state["last_seen"] = self.frame_index

        for observation in observations:
            track_id = int(observation["track_id"])
            if track_id not in self.states or not observation.get("known"):
                continue
            state = self.states[track_id]
            name = str(observation["name"])
            score = float(observation.get("score", 0.0))
            state["history"].append((name, score))
            counts = Counter(item[0] for item in state["history"])
            candidate, count = counts.most_common(1)[0]
            if count >= self.confirmations:
                conflicting = any(
                    other_id != track_id and other.get("known") and
                    other.get("name") == candidate and
                    self.frame_index - other["last_seen"] <= self.max_track_age
                    for other_id, other in self.states.items()
                )
                if not conflicting:
                    scores = [
                        item_score for item_name, item_score in state["history"]
                        if item_name == candidate
                    ]
                    state.update({
                        "name": candidate,
                        "known": True,
                        "score": float(sum(scores) / len(scores)),
                    })

        stale = [
            track_id for track_id, state in self.states.items()
            if self.frame_index - state["last_seen"] > self.max_track_age
        ]
        for track_id in stale:
            del self.states[track_id]

    def identity(self, track_id):
        state = self.states.get(int(track_id))
        if state is None:
            return {"name": "Unknown", "known": False, "score": 0.0}
        return {
            "name": state["name"],
            "known": bool(state["known"]),
            "score": float(state["score"]),
        }


class FaceRecognitionClient:
    """Run InsightFace in its own Conda environment without blocking frames."""

    def __init__(self, python, worker, database, threshold=0.45,
                 provider="auto", detection_size=640,
                 keypoint_confidence=0.35, min_assignment_iou=0.08,
                 min_assignment_score=0.25):
        command = [
            str(python), str(worker), "--database", str(database),
            "--threshold", str(threshold), "--provider", str(provider),
            "--det-size", str(detection_size),
            "--keypoint-confidence", str(keypoint_confidence),
            "--min-assignment-iou", str(min_assignment_iou),
            "--min-assignment-score", str(min_assignment_score),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.pending = queue.Queue(maxsize=1)
        self.results = deque(maxlen=1)
        self.errors = deque(maxlen=4)
        self.lock = threading.Lock()
        self.closed = False
        self.writer = threading.Thread(target=self._writer, daemon=True)
        self.reader = threading.Thread(target=self._reader, daemon=True)
        self.writer.start()
        self.reader.start()

    @staticmethod
    def _exact(stream, size):
        data = bytearray()
        while len(data) < size:
            part = stream.read(size - len(data))
            if not part:
                raise EOFError("face-recognition worker stream closed")
            data.extend(part)
        return bytes(data)

    def _writer(self):
        try:
            while True:
                packet = self.pending.get()
                if packet is None:
                    return
                sequence, frame, track_ids, boxes, keypoints = packet
                data = io.BytesIO()
                np.savez_compressed(
                    data, sequence=np.asarray(sequence, dtype=np.int64),
                    frame=frame, track_ids=np.asarray(track_ids, dtype=np.int64),
                    boxes=np.asarray(boxes, dtype=np.int32),
                    keypoints=np.asarray(keypoints, dtype=np.float32))
                payload = data.getvalue()
                self.process.stdin.write(struct.pack("!I", len(payload)))
                self.process.stdin.write(payload)
                self.process.stdin.flush()
        except Exception as exc:
            with self.lock:
                self.errors.append(str(exc))

    def _reader(self):
        try:
            while True:
                size = struct.unpack(
                    "!I", self._exact(self.process.stdout, 4))[0]
                if not 0 < size < 8 * 1024 * 1024:
                    raise ValueError("invalid face-recognition response size")
                response = json.loads(
                    self._exact(self.process.stdout, size).decode("utf-8"))
                with self.lock:
                    self.results.append(response)
        except Exception as exc:
            if not self.closed:
                with self.lock:
                    self.errors.append(str(exc))

    def submit(self, sequence, frame, tracks):
        if self.closed or self.process.poll() is not None or not tracks:
            return
        item = (
            int(sequence), np.ascontiguousarray(frame.copy()),
            [int(track["track_id"]) for track in tracks],
            [track["bbox"] for track in tracks],
            [track["kpts_2d"] for track in tracks],
        )
        try:
            self.pending.put_nowait(item)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            try:
                self.pending.put_nowait(item)
            except queue.Full:
                pass

    def poll(self):
        with self.lock:
            if not self.results:
                return None
            return self.results.pop()

    def pop_error(self):
        with self.lock:
            return self.errors.popleft() if self.errors else None

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.pending.put_nowait(None)
        except queue.Full:
            try:
                self.pending.get_nowait()
                self.pending.put_nowait(None)
            except queue.Empty:
                pass
        self.writer.join(timeout=1.0)
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        self.reader.join(timeout=1.0)
