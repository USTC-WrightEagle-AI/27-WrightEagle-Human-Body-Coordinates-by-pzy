import contextlib
import io
import unittest
from unittest import mock

import numpy as np

import remember_person
from face_robot.engine import DetectedFace


class EnrollmentDepthTests(unittest.TestCase):
    def test_mixed_depth_enrolls_valid_face_and_labels_invalid_face(self):
        self.run_depth_case(None, 1.2, should_enroll=True)

    def test_all_invalid_depth_does_not_enroll(self):
        self.run_depth_case(None, None, should_enroll=False)

    def test_mixed_depth_keeps_ambiguity_protection(self):
        self.run_depth_case(1.25, 1.2, should_enroll=False)

    def run_depth_case(self, first_depth, second_depth, should_enroll):
        args = remember_person.build_parser().parse_args([
            'Test', '--samples', '1', '--sample-every', '1', '--headless',
        ])
        color = np.zeros((120, 240, 3), dtype=np.uint8)
        depth = np.zeros((120, 240), dtype=np.float32)
        if first_depth is not None:
            depth[:, :110] = first_depth
        if second_depth is not None:
            depth[:, 120:] = second_depth
        faces = [
            DetectedFace((0, 0, 100, 100), .99, np.array([1., 0.])),
            DetectedFace((120, 0, 220, 100), .99, np.array([0., 1.])),
        ]
        camera = mock.MagicMock()
        camera.__enter__.return_value.read.side_effect = [(color, depth), KeyboardInterrupt]
        with (mock.patch.object(remember_person, 'open_camera', return_value=camera),
              mock.patch.object(remember_person, 'FaceEngine') as engine,
              mock.patch.object(remember_person, 'FaceDatabase') as database,
              mock.patch.object(remember_person, 'draw_face') as draw,
              contextlib.redirect_stdout(io.StringIO()) as output):
            engine.return_value.detect.return_value = faces
            database.return_value.contains.return_value = False
            if should_enroll:
                remember_person.enroll(args)
                database.return_value.upsert.assert_called_once()
                np.testing.assert_array_equal(
                    database.return_value.upsert.call_args.args[1], faces[1].embedding,
                )
            else:
                with self.assertRaises(KeyboardInterrupt):
                    remember_person.enroll(args)
                database.return_value.upsert.assert_not_called()
            if first_depth is None:
                labels = [call.args[2] for call in draw.call_args_list]
                self.assertIn('Ignored: no valid depth', labels)
            if first_depth is None and second_depth is None:
                self.assertIn('No face with valid depth', output.getvalue())
