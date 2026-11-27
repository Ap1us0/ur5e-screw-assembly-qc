import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from inspection.assembly_run import run_assembly
from inspection.detector import Detector
from inspection.vision_alignment import HOLE_PLANE_Z_M, FixtureShift, Pinhole, find_hole, fit_fixture


PROFILE = Path(__file__).resolve().parents[1] / 'samples' / 'simulation' / 'profile.json'


class GeometryTests(unittest.TestCase):
    def test_projected_points_round_trip_to_fixture_plane(self):
        angle = np.radians(22)
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        camera = Pinhole(np.array([-.43, .02, .25]), rotation, 640, 480, 39)
        points = np.array([[-.49, 0, HOLE_PLANE_Z_M], [-.51, .03, HOLE_PLANE_Z_M]])
        np.testing.assert_allclose(camera.intersect_plane(camera.project(points)), points, atol=1e-9)

    def test_fixture_fit_rejects_inconsistent_hole_layout(self):
        taught = np.array([[-.484, -.105], [-.51, 0], [-.474, .105]])
        truth = FixtureShift(-8, 4, -4)
        translated = truth.apply(taught)
        estimate, residual = fit_fixture(taught, translated)
        np.testing.assert_allclose(estimate.apply(taught), translated, atol=1e-9)
        self.assertLess(max(residual), 1e-8)
        translated[1, 0] += .002
        with self.assertRaises(ValueError):
            fit_fixture(taught, translated)

    def test_multiple_valid_dark_holes_do_not_produce_a_coordinate(self):
        camera = Pinhole(np.array([0, 0, HOLE_PLANE_Z_M + .17]), np.eye(3), 640, 480, 39)
        frame = np.full((480, 640, 3), 158, np.uint8)
        radius = round(.0073 * camera.focal / .17)
        for shift in (-.012, .012):
            u, v = np.round(camera.project([shift, 0, HOLE_PLANE_Z_M])).astype(int)
            cv2.circle(frame, (u, v), radius, (30, 30, 30), -1)
        observation, reason, _ = find_hole(frame, camera, np.zeros(2))
        self.assertIsNone(observation)
        self.assertIn('多个', reason)


class VisionAssemblyTests(unittest.TestCase):
    def test_shifted_fixture_recovers_from_primary_glare_and_completes(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, report = run_assembly(Detector(PROFILE), Path(temp), 'none', seed=106, alignment='vision',
                                          fixture_shift=FixtureShift(-8, 4, -4), alignment_fault='glare')
            self.assertEqual(report['status'], 'pass')
            self.assertEqual(report['inspection_verdict'], 'pass')
            self.assertLess(max(report['registration']['actual_center_errors_mm'].values()), .20)
            self.assertEqual([(a['site'], a['angle'], bool(a['observation'])) for a in report['alignment_attempts']],
                             [('S1', 'primary', True), ('S2', 'primary', False),
                              ('S2', 'alternate', True), ('S3', 'primary', True)])
            self.assertGreater(report['events'][0]['sim_time_s'], report['alignment_attempts'][-1]['at_s'])
            for attempt in report['alignment_attempts']:
                self.assertTrue((folder / attempt['photo']).is_file())
                self.assertTrue((folder / attempt['marked_photo']).is_file())

    def test_unreadable_hole_stops_before_any_pickup(self):
        with tempfile.TemporaryDirectory() as temp:
            _, report = run_assembly(Detector(PROFILE), Path(temp), 'none', alignment='vision',
                                     fixture_shift=FixtureShift(7, -4, 3), alignment_fault='occluded')
            self.assertEqual(report['status'], 'aborted')
            self.assertEqual(report['stop_code'], 'registration_image')
            self.assertIsNone(report['inspection_report'])
            self.assertIsNone(report['registration'])
            self.assertFalse(any(event['event'] == 'pickup_contact' for event in report['events']))
            self.assertTrue(all(p['status'] == 'pending' for p in report['positions'].values()))


if __name__ == '__main__':
    unittest.main()
