#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度图反投影模块
将 YOLOv8-pose 检测的 2D 关键点 + 深度图 → 3D 空间坐标

反投影公式:
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth(u, v)

适配环境: Jetson AGX Orin + Intel RealSense D455
"""

import numpy as np


class DepthBackProjector:
    """2D 关键点 + 深度图 → 3D 坐标"""

    def __init__(self, fx, fy, cx, cy,
                 depth_unit="mm",
                 min_depth=0.3,
                 max_depth=10.0,
                 hole_radius=5):
        """
        参数:
            fx, fy, cx, cy: 相机内参
            depth_unit: 深度图单位 "mm" 或 "m"
            min_depth: 最小有效深度（米）
            max_depth: 最大有效深度（米）
            hole_radius: 深度空洞邻域搜索半径（像素）
        """
        self.update_intrinsics(fx, fy, cx, cy)
        self.set_depth_unit(depth_unit)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.hole_radius = hole_radius

    def update_intrinsics(self, fx, fy, cx, cy):
        """更新彩色相机内参（来自 ROS CameraInfo）。"""
        values = tuple(float(v) for v in (fx, fy, cx, cy))
        if values[0] <= 0 or values[1] <= 0:
            raise ValueError("fx 和 fy 必须大于 0")
        self.fx, self.fy, self.cx, self.cy = values

    def set_depth_unit(self, depth_unit):
        """设置深度图单位，支持 RealSense 16UC1 和 32FC1 输出。"""
        if depth_unit not in ("mm", "m"):
            raise ValueError(f"不支持的深度单位: {depth_unit}")
        self.depth_unit = depth_unit
        self.depth_scale_m = 0.001 if depth_unit == "mm" else 1.0

    def set_depth_scale(self, depth_scale_m):
        """设置深度原始数值到米的转换系数（由设备运行时查询）。"""
        depth_scale_m = float(depth_scale_m)
        if depth_scale_m <= 0:
            raise ValueError("depth_scale_m 必须大于 0")
        self.depth_unit = "device"
        self.depth_scale_m = depth_scale_m

    def _get_depth(self, depth_image, u, v):
        """
        获取 (u, v) 处的深度值（米）
        如果中心点无效，在邻域内搜索最近的有效值
        """
        h, w = depth_image.shape

        # 边界检查
        if u < 0 or u >= w or v < 0 or v >= h:
            return 0.0

        # 中心有效时，用相近深度的 3x3 中值抑制飞点，同时避免把人体边缘
        # 后方的背景深度混入关节点。
        z_raw = float(depth_image[v, u])
        if z_raw > 0:
            z = z_raw * self.depth_scale_m
            if self.min_depth <= z <= self.max_depth:
                y0, y1 = max(0, v - 1), min(h, v + 2)
                x0, x1 = max(0, u - 1), min(w, u + 2)
                values = (depth_image[y0:y1, x0:x1].astype(float) *
                          self.depth_scale_m)
                valid = values[
                    (values >= self.min_depth) &
                    (values <= self.max_depth) &
                    (np.abs(values - z) <= max(0.08, 0.03 * z))]
                return float(np.median(valid)) if valid.size else z

        # 中心为空时，逐圈取最近一圈有效值的中值，不再依赖扫描顺序取
        # “第一个像素”，避免同一关节在人体和背景深度之间随机跳变。
        for r in range(1, self.hole_radius + 1):
            ring_values = []
            for dv in range(-r, r + 1):
                for du in range(-r, r + 1):
                    if abs(du) != r and abs(dv) != r:
                        continue  # 只搜外圈
                    nu, nv = u + du, v + dv
                    if 0 <= nu < w and 0 <= nv < h:
                        z_raw = float(depth_image[nv, nu])
                        if z_raw > 0:
                            z = z_raw * self.depth_scale_m
                            if self.min_depth <= z <= self.max_depth:
                                ring_values.append(z)
            if ring_values:
                return float(np.median(ring_values))
        return 0.0

    def backproject(self, kpts_2d, depth_image):
        """
        将单个人的 2D 关键点反投影到 3D 空间

        参数:
            kpts_2d: np.ndarray (17, 3) [x, y, conf]
            depth_image: np.ndarray (H, W) 深度图

        返回:
            kpts_3d: np.ndarray (17, 4) [X, Y, Z, valid]
                     valid=1.0 表示该点有有效 3D 坐标
        """
        kpts_3d = np.zeros((17, 4), dtype=float)

        for i in range(17):
            x, y, conf = kpts_2d[i]

            if conf < 0.3:
                continue

            u, v = int(round(x)), int(round(y))

            z = self._get_depth(depth_image, u, v)
            if z <= 0:
                continue

            kpts_3d[i, 0] = (u - self.cx) * z / self.fx  # X
            kpts_3d[i, 1] = (v - self.cy) * z / self.fy  # Y
            kpts_3d[i, 2] = z                              # Z
            kpts_3d[i, 3] = 1.0                           # valid

        return kpts_3d

    def backproject_batch(self, kpts_list, depth_image):
        """
        批量反投影多个人的关键点

        参数:
            kpts_list: list of (17, 3) arrays
            depth_image: (H, W) 深度图

        返回:
            list of (17, 4) arrays
        """
        return [self.backproject(kpts, depth_image) for kpts in kpts_list]

def person_position_3d(kpts_3d):
    """Return a robust person center in the color optical frame."""
    points = np.asarray(kpts_3d, dtype=float)
    if points.ndim != 2 or points.shape[0] < 17 or points.shape[1] < 4:
        return None

    valid = ((points[:, 3] > 0) &
             np.all(np.isfinite(points[:, :3]), axis=1) &
             (points[:, 2] > 0))
    groups = (
        (5, 6, 11, 12),
        (11, 12, 13, 14),
        (13, 14, 15, 16),
        tuple(range(17)),
    )
    for indices in groups:
        selected = [points[index, :3] for index in indices if valid[index]]
        if len(selected) >= 2:
            center = np.median(np.asarray(selected), axis=0)
            if np.all(np.isfinite(center)) and center[2] > 0:
                return center.astype(float)
    return None


def person_distance_3d(kpts_3d):
    """
    估计人体中心到相机光心的欧氏距离（米）。

    优先使用肩和髋部的中值，减少单个关键点深度跳到背景时的
    影响；躯干不可用时依次回退到腿部和所有有效点。
    无法得到有效三维点时返回 np.inf。
    """
    center = person_position_3d(kpts_3d)
    if center is None:
        return float("inf")
    distance = float(np.linalg.norm(center))
    return distance if np.isfinite(distance) and distance > 0 else float("inf")


def select_nearest_detection(detections):
    """选出三维距离最近的检测；深度全部失效时回退到最大检测框。"""
    if not detections:
        return None

    distances = [person_distance_3d(item.get("kpts_3d"))
                 for item in detections]
    valid_indices = [index for index, distance in enumerate(distances)
                     if np.isfinite(distance)]
    if valid_indices:
        selected_index = min(valid_indices, key=lambda index: distances[index])
    else:
        def bbox_area(item):
            bbox = item.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                return 0.0
            return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
                0.0, float(bbox[3]) - float(bbox[1]))

        selected_index = max(range(len(detections)),
                             key=lambda index: bbox_area(detections[index]))

    selected = detections[selected_index]
    selected["distance_m"] = distances[selected_index]
    return selected
