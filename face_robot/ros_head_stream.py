#!/usr/bin/env python3
"""System-Python ROS reader. Emit length-prefixed NumPy packets to stdout."""
from __future__ import annotations
import argparse
import io
import json
import struct
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_robot.head_camera import load_head_configuration, validate_camera_parameters


def image_array(msg):
    enc = msg.encoding.lower()
    if enc in ('bgr8','rgb8'):
        a = np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
        return (a[:,:,::-1] if enc == 'rgb8' else a).copy()
    if enc not in ('16uc1','32fc1'):
        raise ValueError('Unsupported head image encoding: '+msg.encoding)
    dtype = np.dtype(('>' if msg.is_bigendian else '<')+('u2' if enc == '16uc1' else 'f4'))
    a = np.ndarray((msg.height,msg.width),dtype=dtype,buffer=msg.data,strides=(msg.step,dtype.itemsize)).astype(np.float32)
    return a*.001 if enc == '16uc1' else a


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--calibration-dir',required=True)
    p.add_argument('--with-depth',action='store_true')
    args=p.parse_args()
    session, intr=load_head_configuration(args.calibration_dir)
    rclpy.init(args=[])
    node=rclpy.create_node('face_head_camera_reader')
    depths=deque(maxlen=8)
    info_ok=False
    serial_ok=False
    pending=None
    error=None
    last_output=time.monotonic()

    def info(msg):
        nonlocal info_ok, error
        try:
            validate_camera_parameters(msg.width,msg.height,[msg.k[0],msg.k[4],msg.k[2],msg.k[5]],msg.d,intr)
            info_ok=True
        except ValueError as exc:error=exc

    def publish(color, depth=None):
        nonlocal last_output
        frame=image_array(color)
        if frame.shape != (session['height'],session['width'],3):
            raise ValueError('ROS head color resolution differs from calibration')
        payload={'color':frame}
        if depth is not None:
            array=image_array(depth)
            if array.shape != frame.shape[:2]:raise ValueError('Aligned depth/color size mismatch')
            payload['depth']=array
        data=io.BytesIO()
        np.savez(data,**payload)
        b=data.getvalue()
        sys.stdout.buffer.write(struct.pack('!I',len(b))+b)
        sys.stdout.buffer.flush()
        last_output=time.monotonic()

    def try_pair():
        nonlocal pending
        if pending is None or not info_ok or not serial_ok:return
        if time.monotonic()-pending[1] > .5:
            pending=None
            return
        color=pending[0]
        if not args.with_depth:
            publish(color);pending=None;return
        if not depths:return
        d=min(depths,key=lambda m:abs(stamp(m)-stamp(color)))
        if abs(stamp(d)-stamp(color)) <= .05:
            publish(color,d);pending=None

    def on_color(msg):
        nonlocal pending
        pending=(msg,time.monotonic());try_pair()

    def on_depth(msg):
        depths.append(msg);try_pair()

    subs=[node.create_subscription(CameraInfo,'/head/camera/color/camera_info',info,qos_profile_sensor_data),
          node.create_subscription(Image,'/head/camera/color/image_raw',on_color,qos_profile_sensor_data)]
    if args.with_depth:
        subs.append(node.create_subscription(Image,'/head/camera/aligned_depth_to_color/image_raw',on_depth,qos_profile_sensor_data))
    client=node.create_client(GetParameters,'/head/camera/get_parameters')
    future=None
    try:
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.05)
            if error:raise error
            if future is None and client.service_is_ready():
                req=GetParameters.Request();req.names=['serial_no','rgb_camera.color_profile']
                if args.with_depth:req.names.append('depth_module.depth_profile')
                future=client.call_async(req)
            if future is not None and future.done() and not serial_ok:
                values=future.result().values
                if len(values)<2 or values[0].string_value.lstrip('_') != session['serial']:
                    raise ValueError('ROS head camera serial_no differs from calibration')
                expected_profile=f"{session['width']},{session['height']},{session['fps']}"
                if any(v.string_value.replace(' ','') != expected_profile for v in values[1:]):
                    raise ValueError('ROS head stream profiles differ from calibration')
                serial_ok=True
            if time.monotonic()-last_output > 8:
                raise RuntimeError('No fresh calibrated head ROS frames; check /head/camera topics and ROS environment, or use --head-backend sdk when ROS camera is stopped')
    finally:
        node.destroy_node();rclpy.shutdown()


if __name__ == '__main__':
    try:main()
    except (Exception,KeyboardInterrupt) as exc:
        print(str(exc),file=sys.stderr,flush=True)
        raise SystemExit(1)
