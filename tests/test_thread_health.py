import math
import tempfile
import unittest
from pathlib import Path

from inspection.assembly_run import run_assembly
from inspection.detector import Detector
from inspection.thread_health import SlipMonitor


PROFILE = Path(__file__).resolve().parents[1] / 'samples' / 'simulation' / 'profile.json'


class SlipMonitorTests(unittest.TestCase):
    def test_requires_low_torque_and_missing_depth_over_a_rotation_window(self):
        slipping, advancing, loaded = SlipMonitor(), SlipMonitor(), SlipMonitor()
        slip = None
        for i in range(70):
            turns = i / 100
            angle = math.tau * turns
            slip = slipping.observe(angle, 2.4, 0.04)
            if i < 36:
                self.assertIsNone(slip)
            self.assertIsNone(advancing.observe(angle, 2 * turns, 0.04))
            self.assertIsNone(loaded.observe(angle, 2.4, 0.18))
        self.assertIsNotNone(slip)
        self.assertGreater(slip['window_pitch_deficit_mm'], 0.48)


class ThreadHealthWorkflowTests(unittest.TestCase):
    def test_two_slip_signatures_hold_part_for_mechanical_review(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, result = run_assembly(Detector(PROFILE), Path(temp), 'stripped_thread',
                                          seed=106, insertion='force')
            self.assertEqual(result['status'], 'review')
            self.assertEqual(result['mechanical_review_site'], 'S2')
            self.assertEqual(result['stop_code'], 'thread_slip_retry_limit')
            self.assertEqual([p['status'] for p in result['positions'].values()], ['installed', 'review', 'pending'])
            self.assertIsNone(result['inspection_verdict'])
            self.assertIsNone(result['inspection_report'])
            self.assertTrue((folder / result['review_photo']).is_file())
            slips = [e for e in result['events'] if e['event'] == 'thread_slip_detected']
            self.assertEqual([e['attempt'] for e in slips], [1, 2])
            self.assertFalse(result['final']['S2']['thread'])
            self.assertTrue(result['final']['S2']['magnet'])
            self.assertTrue(all(r['spindle_velocity_command_rad_s'] == 0 for r in result['final'].values()))
