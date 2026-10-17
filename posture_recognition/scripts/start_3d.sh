#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${POSE_3D_CONFIG:-${ROOT_DIR}/config/pose_3d_config.yaml}"
ROS_SETUP="${POSE_ROS_SETUP:-/opt/ros/humble/setup.bash}"
PYTHON="${POSE_PYTHON:-/home/rpp/miniconda3/envs/foundationpose/bin/python}"

if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "[ERROR] ROS 2 环境不存在: ${ROS_SETUP}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "[ERROR] 姿态识别 Python 不存在: ${PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] 配置文件不存在: ${CONFIG_PATH}" >&2
    exit 1
fi

# ROS 2 packages come from Humble; CUDA packages come from foundationpose.
# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! "${PYTHON}" -c 'import rclpy, cv_bridge, torch, ultralytics, yaml' >/dev/null; then
    echo "[ERROR] Python 环境缺少 ROS 2、PyTorch 或 Ultralytics 依赖" >&2
    exit 1
fi

for topic in \
    /head/camera/color/image_raw \
    /head/camera/aligned_depth_to_color/image_raw \
    /head/camera/color/camera_info; do
    if ! ros2 topic info "${topic}" 2>/dev/null | \
            grep 'Publisher count: [1-9]' >/dev/null; then
        echo "[ERROR] 头部相机话题没有发布者: ${topic}" >&2
        echo "        请先启动机器人 camera bringup。" >&2
        exit 1
    fi
done

cd "${ROOT_DIR}"
exec "${PYTHON}" "${ROOT_DIR}/src/pose_classifier_3d_ros.py" \
    --config "${CONFIG_PATH}" "$@"
