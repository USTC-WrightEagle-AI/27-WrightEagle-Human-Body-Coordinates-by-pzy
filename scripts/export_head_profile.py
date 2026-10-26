"""Read factory intrinsics for a supported stream pair without starting the camera.

Run with the robot-face Python environment before starting the head ROS driver.
Writes a new profile directory; existing calibration directories are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyrealsense2 as rs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    serial = '260722304986'
    device = next((d for d in rs.context().query_devices()
                   if d.get_info(rs.camera_info.serial_number) == serial), None)
    if device is None:
        raise RuntimeError('Head D455 is not connected: ' + serial)

    def profile(stream, fmt):
        for sensor in device.query_sensors():
            for p in sensor.get_stream_profiles():
                if p.stream_type() != stream or p.format() != fmt:
                    continue
                v = p.as_video_stream_profile()
                if (v.width(), v.height(), v.fps()) == (args.width, args.height, args.fps):
                    return v
        raise RuntimeError(f'Unsupported profile: {stream} {args.width}x{args.height}@{args.fps}')

    color = profile(rs.stream.color, rs.format.rgb8)
    depth = profile(rs.stream.depth, rs.format.z16)

    def parameters(p):
        i = p.get_intrinsics()
        return {
            'image_size_px': {'width': i.width, 'height': i.height},
            'camera_matrix': [[i.fx, 0.0, i.ppx], [0.0, i.fy, i.ppy], [0.0, 0.0, 1.0]],
            'distortion_coefficients': list(i.coeffs),
            'realsense_distortion_model': str(i.model).split('.')[-1],
        }

    intr = parameters(color)
    intr.update(camera='d455', lens_model='realsense-' + intr['realsense_distortion_model'].replace('_', '-'),
                calibration_rms_px=None, report={
                    'source': 'Intel RealSense factory calibration via pyrealsense2',
                    'serial': serial, 'stream': 'color', 'format': 'rgb8', 'fps': args.fps,
                    'realsense_distortion_model': intr['realsense_distortion_model'],
                })
    intr['realsense_depth'] = parameters(depth)
    intr['realsense_depth'].update(stream='depth', format='z16', fps=args.fps)
    for name, source, target, source_name, target_name in [
        ('T_depth_color', color, depth, 'color', 'depth'),
        ('T_color_depth', depth, color, 'depth', 'color'),
    ]:
        e = source.get_extrinsics_to(target)
        matrix = np.eye(4)
        matrix[:3, :3] = np.asarray(e.rotation).reshape(3, 3, order='F')
        matrix[:3, 3] = e.translation
        intr['realsense_depth'][name] = {
            'maps_points_from': source_name, 'maps_points_to': target_name,
            'matrix_4x4': matrix.tolist(),
        }
    raw = (json.dumps(intr, indent=2) + '\n').encode()
    session = {
        'schema_version': 2, 'camera': 'd455', 'serial': serial,
        'width': args.width, 'height': args.height, 'fps': args.fps,
        'purpose': 'Face recognition RGB-D profile; factory intrinsics, no hand-eye calibration',
        'intrinsics_sha256': hashlib.sha256(raw).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'intrinsics.json').write_bytes(raw)
    (args.output / 'session.json').write_text(json.dumps(session, indent=2) + '\n')
    print(f'Exported {serial} {args.width}x{args.height}@{args.fps} to {args.output}')


if __name__ == '__main__':
    main()
