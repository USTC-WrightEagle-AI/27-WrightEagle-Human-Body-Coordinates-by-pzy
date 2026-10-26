import argparse
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import recognize_person
import remember_person
from face_robot import head_camera as head
from face_robot.ros_camera import RosHeadCamera
from face_robot.database import MatchResult
from face_robot.engine import DetectedFace


class HeadCameraTests(unittest.TestCase):
    def args(self,values=()):
        parser=argparse.ArgumentParser()
        head.add_camera_arguments(parser)
        return parser.parse_args(values)

    def test_both_entrypoints_default_to_head_ros(self):
        for args in [recognize_person.build_parser().parse_args([]),remember_person.build_parser().parse_args(['Test'])]:
            self.assertEqual(args.camera,'head')
            self.assertEqual(args.head_backend,'ros')
            self.assertEqual(args.det_size,1280)
            camera=head.open_camera(args)
            self.assertIsInstance(camera,RosHeadCamera)
        session,intr=head.load_head_configuration(head.DEFAULT_CALIBRATION)
        self.assertEqual(session['serial'],head.HEAD_SERIAL)
        self.assertEqual((session['width'],session['height'],session['fps']),(1280,720,30))

    def test_no_silent_camera_or_stream_switch(self):
        for options in [('--width','640'),('--fps','15'),('--realsense-serial','260322273941')]:
            with self.assertRaises(ValueError):head.open_camera(self.args(options))

    def test_modified_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name in ['session.json','intrinsics.json']:
                (p/name).write_bytes((head.DEFAULT_CALIBRATION/name).read_bytes())
            (p/'intrinsics.json').write_bytes((p/'intrinsics.json').read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'checksum'):head.load_head_configuration(p)

    def test_live_focal_drift_tolerance_and_bad_camera_info(self):
        _,intr=head.load_head_configuration(head.DEFAULT_CALIBRATION)
        k=intr['camera_matrix'];values=[k[0][0],k[1][1],k[0][2],k[1][2]]
        live=values.copy();live[0]*=1.0006;live[1]*=1.0006
        head.validate_camera_parameters(1280,720,live,intr['distortion_coefficients'],intr)
        for width,v in [(640,values),(1280,[values[0]*1.02,*values[1:]])]:
            with self.assertRaises(ValueError):head.validate_camera_parameters(width,720,v,intr['distortion_coefficients'],intr)

    def test_sdk_receives_head_serial_and_returns_expected_frame_type(self):
        _,intr=head.load_head_configuration(head.DEFAULT_CALIBRATION)
        color=np.zeros((720,1280,3),dtype=np.uint8);depth=np.ones((720,1280),dtype=np.float32)
        for with_depth in [False,True]:
            with mock.patch.object(head,'RealSenseCamera') as constructor, mock.patch.object(head,'validate_live_intrinsics'):
                device=constructor.return_value
                device.read.return_value=(color,depth)
                with head.open_camera(self.args(['--head-backend','sdk']),with_depth=with_depth) as camera:
                    result=camera.read()
                constructor.assert_called_once_with('head',1280,720,30,serial=head.HEAD_SERIAL)
                self.assertIs(result[0] if with_depth else result,color)
                if with_depth:self.assertIs(result[1],depth)
                device.__exit__.assert_called_once()

    def test_sdk_releases_camera_if_calibration_check_fails(self):
        with mock.patch.object(head,'RealSenseCamera') as constructor, mock.patch.object(head,'validate_live_intrinsics',side_effect=ValueError('bad')):
            with self.assertRaises(ValueError):
                with head.open_camera(self.args(['--head-backend','sdk'])):pass
            constructor.return_value.__exit__.assert_called_once()

    def test_explicit_legacy_video_input_still_works(self):
        with mock.patch.object(head,'Camera') as camera:
            head.open_camera(self.args(['--camera','/dev/video16']))
            camera.assert_called_once_with('/dev/video16',1280,720,30)

    def test_recognition_uses_head_frames_and_existing_database(self):
        args=recognize_person.build_parser().parse_args([])
        camera=mock.MagicMock()
        camera.__enter__.return_value.read.return_value=np.zeros((720,1280,3),dtype=np.uint8)
        face=DetectedFace((10,10,150,150),.99,np.ones(512,dtype=np.float32))
        with (mock.patch.object(recognize_person,'open_camera',return_value=camera) as open_input,
              mock.patch.object(recognize_person,'FaceEngine') as engine,
              mock.patch.object(recognize_person,'FaceDatabase') as database,
              mock.patch.object(recognize_person,'emit_result') as emit,
              mock.patch.object(recognize_person,'draw_face'),
              mock.patch.object(recognize_person.cv2,'imshow'),
              mock.patch.object(recognize_person.cv2,'destroyAllWindows'),
              mock.patch.object(recognize_person.cv2,'waitKey',side_effect=[0,0,ord('q')]),
              contextlib.redirect_stderr(io.StringIO())):
            engine.return_value.detect.return_value=[face]
            database.return_value.names=['Test']
            database.return_value.match.return_value=MatchResult('Test',.8,True)
            recognize_person.recognize(args)
            open_input.assert_called_once_with(args)
            self.assertEqual(engine.return_value.detect.call_count,3)
            self.assertEqual(emit.call_count,1)
            database.return_value.upsert.assert_not_called()


if __name__=='__main__':unittest.main()
