"""Calibrated head D455 input, with explicit overrides for other cameras."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .camera import Camera, parse_camera_source
from .depth import RealSenseCamera

HEAD_SERIAL = '260722304986'
DEFAULT_CALIBRATION = Path(__file__).resolve().parents[1] / 'calibration/head_d455'


def add_camera_arguments(parser):
    parser.add_argument('--camera', default='head',
                        help='head (default): calibrated head D455; or explicit video device/index/URL')
    parser.add_argument('--realsense-serial', help='Explicit RealSense serial (head mode must match calibration)')
    parser.add_argument('--calibration-dir', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--head-backend', choices=('ros','sdk'), default='ros',
                        help='Head input: existing ROS topics (default), or direct SDK when ROS camera is stopped')
    parser.add_argument('--width', type=int, help='Head: calibrated width; other cameras: 1280')
    parser.add_argument('--height', type=int, help='Head: calibrated height; other cameras: 720')
    parser.add_argument('--fps', type=int, help='Head: calibrated fps; other cameras: 30')


def load_head_configuration(directory):
    directory = Path(directory)
    session = json.loads((directory / 'session.json').read_text(encoding='utf-8'))
    raw = (directory / 'intrinsics.json').read_bytes()
    intrinsics = json.loads(raw)
    if (session.get('schema_version') != 2 or session.get('camera') != 'd455'
            or session.get('serial') != HEAD_SERIAL
            or intrinsics.get('camera') != 'd455'
            or intrinsics.get('report', {}).get('serial') != HEAD_SERIAL):
        raise ValueError('Calibration must belong to the head D455 ' + HEAD_SERIAL)
    if hashlib.sha256(raw).hexdigest() != session.get('intrinsics_sha256'):
        raise ValueError('Head intrinsics checksum differs from calibration session')
    if (intrinsics.get('image_size_px') != {'width':session['width'],'height':session['height']}
            or intrinsics.get('report', {}).get('fps') != session['fps']):
        raise ValueError('Head calibration stream configuration is inconsistent')
    return session, intrinsics


def validate_camera_parameters(width, height, values, coefficients, expected):
    k = np.asarray(expected['camera_matrix'])
    size = expected['image_size_px']
    # Match the posture node's 0.5% tolerance for small live focal drift.
    if (width != size['width'] or height != size['height']
            or not np.allclose(values[:2],[k[0,0],k[1,1]],rtol=.005,atol=1e-5)
            or not np.allclose(values[2:],[k[0,2],k[1,2]],rtol=0,atol=1e-4)
            or len(coefficients) != len(expected['distortion_coefficients'])
            or not np.allclose(coefficients,expected['distortion_coefficients'],rtol=0,atol=1e-7)):
        raise ValueError('Live head RGB intrinsics differ from calibration; check camera/stream configuration')


def validate_live_intrinsics(actual, expected):
    validate_camera_parameters(actual.width,actual.height,
                               [actual.fx,actual.fy,actual.ppx,actual.ppy],actual.coeffs,expected)
    if str(actual.model).split('.')[-1].replace('_','-') != expected['lens_model'].removeprefix('realsense-'):
        raise ValueError('Head RGB distortion model differs from calibration')


class HeadCamera:
    """Use SDK factory distortion/alignment, without altering face model pixels."""
    def __init__(self, args, with_depth=False):
        session, self.intrinsics = load_head_configuration(args.calibration_dir)
        serial = args.realsense_serial or session['serial']
        if serial != session['serial']:
            raise ValueError('Head camera serial must match calibration: ' + session['serial'])
        for key in ('width','height','fps'):
            value = getattr(args,key)
            if value is not None and value != session[key]:
                raise ValueError(f'Head --{key} must match calibrated value {session[key]}')
        self.with_depth = with_depth
        self.camera = RealSenseCamera('head',session['width'],session['height'],session['fps'],serial=serial)

    def __enter__(self):
        self.camera.__enter__()
        try:
            validate_live_intrinsics(self.camera.color_intrinsics,self.intrinsics)
        except Exception:
            self.camera.__exit__(None,None,None)
            raise
        return self

    def read(self):
        color, depth = self.camera.read()
        return (color,depth) if self.with_depth else color

    def __exit__(self, exc_type, exc_value, traceback):
        self.camera.__exit__(exc_type,exc_value,traceback)


def open_camera(args, with_depth=False):
    if args.camera == 'head':
        if args.head_backend == 'ros':
            from .ros_camera import RosHeadCamera
            return RosHeadCamera(args,with_depth=with_depth)
        return HeadCamera(args,with_depth=with_depth)
    source = parse_camera_source(args.camera)
    dimensions = [getattr(args,k) if getattr(args,k) is not None else default
                  for k,default in [('width',1280),('height',720),('fps',30)]]
    if any(v <= 0 for v in dimensions):
        raise ValueError('Camera width, height and fps must be positive')
    if with_depth:
        return RealSenseCamera(source,*dimensions,serial=args.realsense_serial)
    if args.realsense_serial:
        raise ValueError('Use --camera head for serial-selected recognition; explicit video inputs use --camera only')
    return Camera(source,*dimensions)
