"""Compare live CameraInfo with a calibration snapshot without ROS imports."""

import numpy as np


def validate_snapshot_intrinsics(msg, intrinsics, width, height):
    """Allow small focal drift and report every mismatched field.

    The robot reported about +0.300% in both focal lengths with unchanged
    principal point and distortion. Use a bounded 0.5% focal tolerance;
    backprojection uses the live values, never the snapshot focal lengths.
    """
    matrix = np.asarray(intrinsics["camera_matrix"], dtype=float)
    actual = np.asarray([msg.k[0], msg.k[4], msg.k[2], msg.k[5]])
    expected = np.asarray(
        [matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]])
    differences = []
    if (msg.width, msg.height) != (width, height):
        differences.append(
            f"分辨率: 实时={msg.width}x{msg.height}, 标定={width}x{height}")
    for index, name in enumerate(("fx", "fy", "cx", "cy")):
        focal = index < 2
        if not np.isclose(actual[index], expected[index],
                          rtol=0.005 if focal else 0.0,
                          atol=1e-5 if focal else 1e-4):
            tolerance = "0.5% + 0.00001 px" if focal else "0.0001 px"
            differences.append(
                f"{name}: 实时={actual[index]:.9f}, "
                f"标定={expected[index]:.9f}, "
                f"差值={actual[index] - expected[index]:+.9f} px "
                f"(容差 {tolerance})")
    coefficients = np.asarray(msg.d, dtype=float)
    expected_coefficients = np.asarray(
        intrinsics["distortion_coefficients"], dtype=float)
    if (coefficients.shape != expected_coefficients.shape or
            not np.allclose(coefficients, expected_coefficients,
                            rtol=0.0, atol=1e-7)):
        differences.append(
            f"D: 实时={coefficients.tolist()}, "
            f"标定={expected_coefficients.tolist()} (容差 1e-7)")
    if differences:
        raise ValueError(
            "运行中的头部相机内参与标定快照不一致:\n  "
            + "\n  ".join(differences)
            + f"\n  实时畸变模型={msg.distortion_model!r}, "
            f"标定模型={intrinsics.get('lens_model', '未记录')!r}"
            + "\n请核对 CameraInfo、驱动流配置和标定快照；"
            "确认差异来源后再更新对应标定，勿直接覆盖旧文件。")
