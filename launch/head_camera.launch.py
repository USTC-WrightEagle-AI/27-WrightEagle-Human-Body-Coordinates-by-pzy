"""Start only the head camera, using the face project's calibrated stream profile."""
import json
from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    root = Path(__file__).resolve().parents[1]
    session = json.loads((root / 'calibration/head_d455_hd/session.json').read_text())
    profile = f"{session['width']},{session['height']},{session['fps']}"
    return LaunchDescription([Node(
        package='realsense2_camera', executable='realsense2_camera_node',
        name='camera', namespace='head', output='screen',
        parameters=[{
            'serial_no': session['serial'], 'camera_name': 'head_camera',
            'publish_tf': False, 'enable_color': True, 'enable_depth': True,
            'enable_infra': False, 'enable_infra1': False, 'enable_infra2': False,
            'enable_accel': False, 'enable_gyro': False,
            'pointcloud.enable': False, 'align_depth.enable': True,
            'rgb_camera.color_profile': profile, 'depth_module.depth_profile': profile,
        }],
    )])
