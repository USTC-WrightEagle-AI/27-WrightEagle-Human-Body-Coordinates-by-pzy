#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS 2 head-D455 3D static posture recognition node."""

import argparse
import cv2
import numpy as np
import json
import time
import yaml
import os
import hashlib
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import message_filters
from ultralytics import YOLO
from PIL import Image as PILImage, ImageDraw, ImageFont

from depth_backproject import (
    DepthBackProjector,
    person_distance_3d,
    person_position_3d,
    select_nearest_detection,
)
from person_identity import FaceRecognitionClient, NameBinder, transform_point
from tcn_classifier import CN_LABELS, TCNPoseClassifier


def resolve_path(value, config_path):
    """将配置中的相对路径按配置文件所在目录解析。"""
    value = os.path.expanduser(str(value))
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(os.path.dirname(config_path), value))


class PoseClassifier3DNode(Node):

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"{type(value).__name__} 不支持 JSON 序列化")

    def __init__(self, config_path=None, debug=False, publish_image=True,
                 frame_skip=0):
        super().__init__("pose_classifier_3d_node")
        config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "pose_3d_config.yaml")
        config_path = os.path.abspath(os.path.expanduser(config_path))
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置不存在: {config_path}")
        with open(config_path, "r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream) or {}
        self.get_logger().info(f"已加载配置: {config_path}")

        cfg_model = self.config.get("model", {})
        cfg_cam = self.config.get("camera", {})
        cfg_depth = cfg_cam.get("depth", {})
        cfg_selection = self.config.get("person_selection", {})
        cfg_tracking = self.config.get("tracking", {})
        cfg_face = self.config.get("face_recognition", {})
        cfg_localization = self.config.get("localization", {})

        self.model_path = resolve_path(
            cfg_model.get("weights_path", "../models/yolov8n-pose.pt"),
            config_path)
        self.rgb_topic = cfg_cam.get(
            "rgb_topic", "/head/camera/color/image_raw")
        self.depth_topic = cfg_cam.get(
            "depth_topic", "/head/camera/aligned_depth_to_color/image_raw")
        self.camera_info_topic = cfg_cam.get(
            "camera_info_topic", "/head/camera/color/camera_info")
        self.camera_info_timeout = float(cfg_cam.get("camera_info_timeout", 15.0))
        self.frame_id = cfg_cam.get(
            "frame_id", "head_camera_color_optical_frame")
        self.conf_threshold = float(cfg_model.get("conf_threshold", 0.5))
        self.input_size = int(cfg_model.get("input_size", 640))
        self.device = cfg_model.get("device", 0)
        self.publish_image = bool(publish_image)
        self.frame_skip = max(0, int(frame_skip))
        self.debug = bool(debug)
        self.nearest_only = bool(cfg_selection.get("nearest_only", True))
        self.tracker_config = resolve_path(
            cfg_tracking.get("config", "bytetrack.yaml"), config_path)
        self.font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        self.down_axis = np.asarray(
            cfg_cam.get("down_axis", [0.0, 1.0, 0.0]), dtype=float)
        if self.down_axis.shape != (3,) or np.linalg.norm(self.down_axis) < 1e-9:
            raise ValueError("camera.down_axis 必须是非零的三元向量")
        self.down_axis /= np.linalg.norm(self.down_axis)
        self._last_logs = {}
        self.frame_sequence = 0

        self._validate_calibration_snapshot(cfg_cam, config_path)
        self._validate_camera_node(cfg_cam)

        self.get_logger().info(f"等待相机内参: {self.camera_info_topic}")
        camera_info_messages = []
        initial_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, camera_info_messages.append,
            qos_profile_sensor_data)
        deadline = time.monotonic() + self.camera_info_timeout
        while not camera_info_messages and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(initial_info_sub)
        if not camera_info_messages:
            raise RuntimeError(
                f"在 {self.camera_info_timeout:.1f}s 内未收到 CameraInfo: "
                f"{self.camera_info_topic}")
        camera_info = camera_info_messages[-1]
        self._validate_camera_info(camera_info)
        self._validate_live_intrinsics(camera_info, cfg_cam, config_path)
        self.backprojector = DepthBackProjector(
            fx=camera_info.k[0], fy=camera_info.k[4],
            cx=camera_info.k[2], cy=camera_info.k[5],
            depth_unit="mm",
            min_depth=cfg_depth.get("min_valid", 0.3),
            max_depth=cfg_depth.get("max_valid", 10.0),
            hole_radius=cfg_depth.get("hole_search_radius", 5),
        )
        self.image_width = camera_info.width
        self.image_height = camera_info.height
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_callback,
            qos_profile_sensor_data)
        self.get_logger().info(
            "实时内参: "
            f"{self.image_width}x{self.image_height}, "
            f"fx={camera_info.k[0]:.6f}, fy={camera_info.k[4]:.6f}, "
            f"cx={camera_info.k[2]:.6f}, cy={camera_info.k[5]:.6f}")

        # ---- 检测器 ----
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        self.get_logger().info(f"加载模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        cls_config = self.config.get("classifier", {})
        classifier_weights = resolve_path(
            cls_config.get("weights_path", "../models/pose_tcn_ntu_fukinect.pth"),
            config_path)
        if not os.path.isfile(classifier_weights):
            raise FileNotFoundError(f"TCN 权重不存在: {classifier_weights}")
        self.classifier = TCNPoseClassifier.from_config(
            cls_config, classifier_weights)
        self.get_logger().info(f"加载 TCN: {classifier_weights}")
        self.base_transform = None
        self.base_transform_validated = False
        if cfg_localization.get("publish_candidate_left_base", True):
            transform_path = resolve_path(
                cfg_localization["candidate_base_transform"], config_path)
            with open(transform_path, "r", encoding="utf-8") as stream:
                transform_data = json.load(stream)
            self.base_transform = np.asarray(
                transform_data["transforms"]["T_base_camera"]["matrix_4x4"],
                dtype=float)
            self.base_transform_validated = bool(
                transform_data.get("validation", {}).get("passed", False))

        self.face_interval = max(1, int(cfg_face.get("every_n_frames", 5)))
        self.name_binder = NameBinder(
            confirmations=cfg_face.get("confirmations", 2),
            history_size=cfg_face.get("history_size", 5),
            max_track_age=cfg_face.get("max_track_age", 45))
        self.face_client = None
        if cfg_face.get("enabled", True):
            self.face_client = FaceRecognitionClient(
                python=cfg_face["python"],
                worker=resolve_path(cfg_face["worker"], config_path),
                database=resolve_path(cfg_face["database"], config_path),
                threshold=cfg_face.get("threshold", 0.45),
                provider=cfg_face.get("provider", "auto"),
                detection_size=cfg_face.get("det_size", 640),
                keypoint_confidence=cfg_face.get(
                    "keypoint_confidence", 0.35),
                min_assignment_iou=cfg_face.get(
                    "min_assignment_iou", 0.08),
                min_assignment_score=cfg_face.get(
                    "min_assignment_score", 0.25))
            self.get_logger().info("人脸识别子进程已启动")
        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_callback_time = None
        self.realtime_fps = 0.0
        if self.publish_image:
            if not os.path.isfile(self.font_path):
                raise FileNotFoundError(f"中文字体不存在: {self.font_path}")
            self.label_font = ImageFont.truetype(self.font_path, 22)
            self.stats_font = ImageFont.truetype(self.font_path, 24)

        # ---- 发布器 ----
        self.results_pub = self.create_publisher(
            String, "/pose_classifier_3d/results", 1)
        self.identity_locations_pub = self.create_publisher(
            String, "/person_identity_locations", 1)
        if self.publish_image:
            self.image_pub = self.create_publisher(
                Image, "/pose_classifier_3d/image", 1)

        # ---- RGB+深度 同步订阅 ----
        rgb_sub = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.get_logger().info("节点启动完成")
        self.get_logger().info(f"RGB: {self.rgb_topic}")
        self.get_logger().info(f"深度: {self.depth_topic}")
        self.get_logger().info(f"内参: {self.camera_info_topic}")
        self.get_logger().info(f"向下轴: {self.down_axis.tolist()}")
        self.get_logger().info(
            f"  人员筛选: {'仅识别最近人员' if self.nearest_only else '识别所有人员'}")
        self.get_logger().info("发布: /pose_classifier_3d/results (JSON)")
        self.get_logger().info("发布: /person_identity_locations (JSON)")
        if self.publish_image:
            self.get_logger().info("发布: /pose_classifier_3d/image (可视化)")

    @staticmethod
    def _calibration_files(cfg_cam, config_path):
        directory = resolve_path(cfg_cam["calibration_dir"], config_path)
        with open(os.path.join(directory, "session.json"), "r",
                  encoding="utf-8") as stream:
            session = json.load(stream)
        intrinsics_path = os.path.join(directory, "intrinsics.json")
        with open(intrinsics_path, "rb") as stream:
            raw = stream.read()
        return session, json.loads(raw), raw

    def _validate_calibration_snapshot(self, cfg_cam, config_path):
        session, intrinsics, raw = self._calibration_files(cfg_cam, config_path)
        expected = (
            str(cfg_cam.get("serial_no")), int(cfg_cam.get("width")),
            int(cfg_cam.get("height")), int(cfg_cam.get("fps")))
        actual = (
            session.get("serial"), session.get("width"),
            session.get("height"), session.get("fps"))
        if actual != expected:
            raise ValueError(
                f"头部相机配置与标定 session 不一致: {expected} != {actual}")
        if hashlib.sha256(raw).hexdigest() != session.get("intrinsics_sha256"):
            raise ValueError("头部相机内参文件摘要与标定 session 不一致")
        if (intrinsics.get("report", {}).get("serial") != expected[0] or
                intrinsics.get("image_size_px") != {
                    "width": expected[1], "height": expected[2]}):
            raise ValueError("头部相机内参不属于配置的设备或分辨率")

    def _validate_camera_node(self, cfg_cam):
        service = cfg_cam.get(
            "parameter_service", "/head/camera/get_parameters")
        client = self.create_client(GetParameters, service)
        deadline = time.monotonic() + self.camera_info_timeout
        while not client.wait_for_service(timeout_sec=0.2):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"头部相机参数服务不可用: {service}")
        request = GetParameters.Request()
        request.names = [
            "serial_no", "rgb_camera.color_profile",
            "depth_module.depth_profile"]
        future = client.call_async(request)
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"读取头部相机参数超时: {service}")
        values = future.result().values
        expected_profile = (
            f"{cfg_cam['width']},{cfg_cam['height']},{cfg_cam['fps']}")
        serial = values[0].string_value.lstrip("_") if values else ""
        profiles = [value.string_value.replace(" ", "") for value in values[1:]]
        if serial != str(cfg_cam["serial_no"]):
            raise ValueError(
                f"运行中的头部相机序列号不匹配: {serial}")
        if len(profiles) != 2 or any(
                profile != expected_profile for profile in profiles):
            raise ValueError(
                f"运行中的头部相机流配置不匹配: 期望 {expected_profile}, 实际 {profiles}")

    def _validate_live_intrinsics(self, msg, cfg_cam, config_path):
        _, intrinsics, _ = self._calibration_files(cfg_cam, config_path)
        matrix = np.asarray(intrinsics["camera_matrix"], dtype=float)
        values = np.asarray([msg.k[0], msg.k[4], msg.k[2], msg.k[5]])
        expected = np.asarray(
            [matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]])
        focal_ok = np.allclose(values[:2], expected[:2], rtol=0.005, atol=1e-5)
        principal_ok = np.allclose(
            values[2:], expected[2:], rtol=0.0, atol=1e-4)
        distortion_ok = np.allclose(
            msg.d, intrinsics["distortion_coefficients"],
            rtol=0.0, atol=1e-7)
        if (msg.width != int(cfg_cam["width"]) or
                msg.height != int(cfg_cam["height"]) or
                not focal_ok or not principal_ok or not distortion_ok):
            raise ValueError("运行中的头部相机内参与配置的标定快照不一致")

    def _log_throttled(self, level, key, period, message):
        now = time.monotonic()
        if now - self._last_logs.get(key, float("-inf")) < period:
            return
        self._last_logs[key] = now
        getattr(self.get_logger(), level)(message)

    @staticmethod
    def _validate_camera_info(msg):
        if msg.width <= 0 or msg.height <= 0 or len(msg.k) != 9:
            raise ValueError("CameraInfo 尺寸或 K 矩阵无效")
        if msg.k[0] <= 0 or msg.k[4] <= 0:
            raise ValueError("CameraInfo 中 fx/fy 无效")

    def _camera_info_callback(self, msg):
        try:
            self._validate_camera_info(msg)
            self.backprojector.update_intrinsics(
                msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.image_width = msg.width
            self.image_height = msg.height
        except ValueError as exc:
            self._log_throttled(
                "warning", "camera_info", 5.0,
                f"忽略无效 CameraInfo: {exc}")

    def _detect_and_classify(self, frame, depth_image, debug=False):
        """ByteTrack + pose + 3D localization + asynchronous identity."""
        outputs = self.model.track(
            frame, conf=self.conf_threshold, imgsz=self.input_size,
            device=self.device, verbose=False, persist=True,
            tracker=self.tracker_config)
        results = []
        if (len(outputs) == 0 or outputs[0].keypoints is None or
                outputs[0].boxes is None or outputs[0].boxes.id is None):
            self.classifier.classify_batch(
                [], down_axis=self.down_axis, debug=debug)
            face_response = self.face_client.poll() if self.face_client else None
            self.name_binder.update(
                [], face_response.get("matches", []) if face_response else [])
            return results

        kp_list = outputs[0].keypoints
        box_list = outputs[0].boxes
        keypoints_xy = kp_list.xy.cpu().numpy()
        keypoints_conf = kp_list.conf.cpu().numpy()
        boxes_xyxy = box_list.xyxy.cpu().numpy()
        track_ids = box_list.id.int().cpu().numpy()

        detections = []
        for i in range(len(kp_list)):
            kpts_2d = np.concatenate([
                keypoints_xy[i],
                keypoints_conf[i, :, np.newaxis],
            ], axis=1)

            bbox = boxes_xyxy[i].astype(int).tolist()
            kpts_3d = self.backprojector.backproject(kpts_2d, depth_image)
            position = person_position_3d(kpts_3d)
            detections.append({
                "bbox": bbox,
                "track_id": int(track_ids[i]),
                "kpts_2d": kpts_2d,
                "kpts_3d": kpts_3d,
                "position_3d": position,
                "distance_m": person_distance_3d(kpts_3d),
            })

        if self.nearest_only and detections:
            nearest = select_nearest_detection(detections)
            detections = [nearest]

        predictions = self.classifier.classify_batch(
            detections, down_axis=self.down_axis, debug=debug)

        face_response = self.face_client.poll() if self.face_client else None
        observations = face_response.get("matches", []) if face_response else []
        active_track_ids = [item["track_id"] for item in detections]
        self.name_binder.update(active_track_ids, observations)
        if (self.face_client is not None and
                self.frame_sequence % self.face_interval == 0):
            self.face_client.submit(self.frame_sequence, frame, detections)
        if self.face_client is not None:
            error = self.face_client.pop_error()
            if error:
                self._log_throttled(
                    "error", "face_worker", 5.0,
                    f"人脸识别子进程异常: {error}")

        for detection, prediction in zip(detections, predictions):
            lname = prediction["label_name"]
            track_id = int(detection["track_id"])
            identity = self.name_binder.identity(track_id)
            entry = {
                "bbox": detection["bbox"],
                "person_id": track_id,
                "name": identity["name"],
                "name_known": identity["known"],
                "face_score": round(identity["score"], 4),
                "pose": lname,
                "pose_cn": CN_LABELS[lname],
                "conf": round(prediction["score"], 2),
                # 仅供当前帧可视化使用，发布 JSON 前会剔除。
                "_keypoints_2d": detection["kpts_2d"].copy(),
            }
            if prediction.get("features"):
                entry["features"] = prediction["features"]
            distance_m = detection.get("distance_m", float("inf"))
            if np.isfinite(distance_m):
                entry["distance_m"] = round(float(distance_m), 2)
            position = detection.get("position_3d")
            if position is not None:
                entry["camera_xyz_m"] = [
                    round(float(value), 3) for value in position]
                if self.base_transform is not None:
                    base_position = transform_point(
                        self.base_transform, position)
                    entry["left_base_xyz_m"] = [
                        round(float(value), 3) for value in base_position]
                    entry["left_base_transform_validated"] = bool(
                        self.base_transform_validated)
            results.append(entry)

        return results

    def _draw(self, frame, results):
        """绘制可视化结果"""
        COLORS = {
            "standing": (0, 255, 0), "sitting": (255, 165, 0),
            "squatting": (0, 255, 255), "lying": (255, 0, 0),
            "bending": (255, 0, 255), "unknown": (128, 128, 128),
        }
        SKELETON = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
        ]

        labels = []
        for person in results:
            bbox = person["bbox"]
            pose = person["pose"]
            conf = person["conf"]
            color = COLORS.get(pose, (128, 128, 128))

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            kpts = person.get("_keypoints_2d")
            if kpts is not None:
                for i, j in SKELETON:
                    if kpts[i, 2] >= 0.3 and kpts[j, 2] >= 0.3:
                        p1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                        p2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
                for x, y, score in kpts:
                    if score >= 0.3:
                        cv2.circle(frame, (int(x), int(y)), 3,
                                   (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (int(x), int(y)), 4,
                                   color, 1, cv2.LINE_AA)

            distance_text = ""
            if "distance_m" in person:
                distance_text = f" {person['distance_m']:.2f}m"
            position_text = ""
            if "camera_xyz_m" in person:
                x_value, y_value, z_value = person["camera_xyz_m"]
                position_text = (
                    f"\ncam [{x_value:.2f}, {y_value:.2f}, {z_value:.2f}]m")
            labels.append((bbox[0], bbox[1],
                           f"ID {person['person_id']} {person['name']} "
                           f"{person['pose_cn']} {conf:.0%}{distance_text}"
                           f"{position_text}",
                           color))

        counts = {}
        for p in results:
            cn = p["pose_cn"]
            counts[cn] = counts.get(cn, 0) + 1
        # OpenCV Hershey 字体不包含中文，使用本机 Noto CJK 字体统一绘制。
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)
        for x, y, label, color in labels:
            text_box = draw.multiline_textbbox(
                (0, 0), label, font=self.label_font, spacing=2)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            top = max(0, y - text_h - 8)
            rgb_color = (color[2], color[1], color[0])
            draw.rectangle((x, top, x + text_w + 8, top + text_h + 8),
                           fill=rgb_color)
            draw.multiline_text(
                (x + 4, top + 2), label, font=self.label_font,
                fill=(255, 255, 255), spacing=2)

        y = 8
        for cn, n in counts.items():
            text = f"{cn}: {n}人"
            draw.text((10, y), text, font=self.stats_font,
                      fill=(255, 255, 0), stroke_width=1,
                      stroke_fill=(0, 0, 0))
            y += 32
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    def callback(self, rgb_msg, depth_msg):
        """RGB + 深度 同步回调"""
        if self.frame_skip > 0 and self.frame_count % (self.frame_skip + 1) != 0:
            self.frame_count += 1
            return

        callback_time = time.monotonic()
        if self.last_callback_time is not None:
            interval = callback_time - self.last_callback_time
            if interval > 0:
                instant_fps = 1.0 / interval
                if self.realtime_fps <= 0:
                    self.realtime_fps = instant_fps
                else:
                    self.realtime_fps = (
                        0.15 * instant_fps + 0.85 * self.realtime_fps)
        self.last_callback_time = callback_time

        t0 = time.perf_counter()

        # 转换 RGB
        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB 转换错误: {e}")
            return

        # 按 ROS 编码确定深度单位，不对浮点深度做破坏性强转。
        try:
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg, depth_msg.encoding)
            if depth_msg.encoding in ("16UC1", "mono16"):
                self.backprojector.set_depth_unit("mm")
            elif depth_msg.encoding == "32FC1":
                self.backprojector.set_depth_unit("m")
            else:
                self._log_throttled(
                    "error", "depth_encoding", 5.0,
                    f"不支持的深度编码: {depth_msg.encoding}")
                return
        except CvBridgeError as e:
            self.get_logger().error(f"深度转换错误: {e}")
            return

        if frame.shape[:2] != depth_image.shape[:2]:
            self._log_throttled(
                "error", "image_size", 5.0, "RGB 与深度图尺寸不一致: "
                f"RGB={frame.shape[:2]}, depth={depth_image.shape[:2]}")
            return
        if (frame.shape[1] != self.image_width or
                frame.shape[0] != self.image_height):
            self._log_throttled(
                "error", "camera_info_size", 5.0,
                "RGB 尺寸与 CameraInfo 不一致: "
                f"RGB={frame.shape[1]}x{frame.shape[0]}, "
                f"CameraInfo={self.image_width}x{self.image_height}")
            return

        # 检测 + 分类
        self.frame_sequence += 1
        results = self._detect_and_classify(frame, depth_image, debug=self.debug)

        dt = time.perf_counter() - t0
        process_fps = 1.0 / dt if dt > 0 else 0

        # 发布 JSON
        # 可视化关键点属于内部数据，不改变现有 JSON 接口。
        public_results = [
            {key: value for key, value in person.items()
             if not key.startswith("_")}
            for person in results
        ]
        output = {
            "header": {
                "stamp": (rgb_msg.header.stamp.sec +
                          rgb_msg.header.stamp.nanosec * 1e-9),
                "frame_id": rgb_msg.header.frame_id,
            },
            "fps": round(process_fps, 1),
            "realtime_fps": round(self.realtime_fps, 1),
            "tracking": "ByteTrack",
            "camera_coordinate_frame": "head_camera_color_optical_frame",
            "camera_coordinate_convention": "+X right, +Y down, +Z forward",
            "left_base_transform_validated": bool(
                self.base_transform_validated),
            "num_persons": len(public_results),
            "persons": public_results,
        }
        self.results_pub.publish(
            String(data=json.dumps(
                output, ensure_ascii=False, default=self._json_default)))
        identity_locations = {
            "header": output["header"],
            "camera_coordinate_frame": output["camera_coordinate_frame"],
            "camera_coordinate_convention": output[
                "camera_coordinate_convention"],
            "left_base_transform_validated": bool(
                self.base_transform_validated),
            "persons": [
                {
                    key: person[key]
                    for key in (
                        "person_id", "name", "name_known", "face_score",
                        "camera_xyz_m", "left_base_xyz_m",
                        "left_base_transform_validated")
                    if key in person
                }
                for person in public_results
            ],
        }
        self.identity_locations_pub.publish(String(data=json.dumps(
            identity_locations, ensure_ascii=False,
            default=self._json_default)))

        # 发布可视化
        if self.publish_image:
            vis = self._draw(frame.copy(), results)
            cv2.putText(
                        vis,
                        f"Realtime FPS: {self.realtime_fps:.1f} | "
                        f"Process FPS: {process_fps:.1f}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                        cv2.LINE_AA)
            try:
                vis_msg = self.bridge.cv2_to_imgmsg(vis, "bgr8")
                vis_msg.header = rgb_msg.header
                self.image_pub.publish(vis_msg)
            except CvBridgeError as e:
                self.get_logger().error(f"图像发布错误: {e}")

        # 日志
        if results:
            persons = ", ".join(
                [f"ID={r['person_id']} {r['name']} "
                 f"{r['pose_cn']}({r['conf']:.0%}) "
                 f"坐标={r.get('camera_xyz_m', '-')}"
                 for r in results])
            self._log_throttled(
                "info", "results", 1.0,
                f"{persons} | "
                f"实时={self.realtime_fps:.1f} FPS "
                f"处理={process_fps:.1f} FPS")

            if self.debug:
                for i, r in enumerate(results):
                    feat = r.get("features", {})
                    self._log_throttled(
                        "info", f"debug_{i}", 1.0,
                        f"  人{i+1}: 膝角={feat.get('knee_angle', '-')}° "
                        f"髋角={feat.get('hip_angle', '-')}° "
                        f"躯干倾角={feat.get('torso_angle', '-')}° "
                        f"身高比={feat.get('height_ratio', '-')} "
                        f"3D点={feat.get('valid_3d_keypoints', '-')}")
        self.frame_count += 1

    def run(self):
        rclpy.spin(self)

    def close(self):
        if self.face_client is not None:
            self.face_client.close()
            self.face_client = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=0)
    args = parser.parse_args()
    rclpy.init(args=None)
    node = None
    try:
        node = PoseClassifier3DNode(
            config_path=args.config_path,
            debug=args.debug,
            publish_image=not args.no_image,
            frame_skip=args.frame_skip)
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
