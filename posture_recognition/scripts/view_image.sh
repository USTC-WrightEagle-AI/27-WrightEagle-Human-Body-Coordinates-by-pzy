#!/usr/bin/env bash
set -eo pipefail

ROS_SETUP="${POSE_ROS_SETUP:-/opt/ros/humble/setup.bash}"
RQT_IMAGE_VIEW="/opt/ros/humble/lib/rqt_image_view/rqt_image_view"

if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "[ERROR] ROS 2 环境不存在: ${ROS_SETUP}" >&2
    exit 1
fi
if [[ ! -f "${RQT_IMAGE_VIEW}" ]]; then
    echo "[ERROR] rqt_image_view 未安装: ${RQT_IMAGE_VIEW}" >&2
    exit 1
fi

# The user's Conda base currently provides Python 3.14. ROS 2 Humble's rclpy
# extension is built for Ubuntu's Python 3.10, so bypass PATH intentionally.
# shellcheck disable=SC1090
source "${ROS_SETUP}"
exec /usr/bin/python3 "${RQT_IMAGE_VIEW}" \
    /pose_classifier_3d/image "$@"
