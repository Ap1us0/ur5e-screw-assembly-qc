import unittest
import tempfile
import json
from pathlib import Path

from inspection.assembly import Assembly, AssemblyStopped, PITCH_MM, TORQUE_TARGET_NM
from inspection.assembly_acceptance import Case, score
from inspection.assembly_run import run_round
from inspection.detector import Detector, MISSING, PRESENT, SITES


PROFILE = Path(__file__).resolve().parents[1] / 'samples' / 'simulation' / 'profile.json'


class AssemblyTests(unittest.TestCase):
    def test_all_positions_start_empty_and_visual_gate_runs_after_three_installations(self):
        rig = Assembly()
        self.addCleanup(rig.close)
        detector = Detector(PROFILE)
        self.assertEqual(len({station.screw for station in rig.stations.values()}), 3)
        self.assertEqual(len({station.screw_nominal[0] for station in rig.stations.values()}), 3)
        for site in SITES:
            photo = rig.capture(site, 'primary')
            self.assertEqual(detector.inspect(site, 'primary', photo).state, MISSING)
        with tempfile.TemporaryDirectory() as tmp:
            folder, report = run_round(rig, detector, Path(tmp), None, False)
            self.assertEqual(report['status'], 'pass')
            self.assertEqual(tuple(report['order']), SITES)
            self.assertEqual(report['motion'], 'optimized')
            self.assertLess(report['inspection_started_at_s'], report['cycle_time_s'])
            self.assertTrue(all(move['settled_velocity_rad_s'] < .002 for move in report['moves']))
            self.assertTrue(all(move['settled_error_rad'] < (.0003 if move['stage'] == 'inspection' else .0018)
                                for move in report['moves']))
            self.assertEqual([(e['site'], e['event']) for e in report['events']], [
                (site, event) for site in SITES for event in (
                    'pickup_contact', 'screw_lifted', 'axis_alignment', 'thread_engaged', 'torque_cutoff',
                    'spindle_stopped', 'driver_released', 'retention_verified',
                )])
            self.assertGreaterEqual(report['inspection_started_at_s'], report['events'][-1]['sim_time_s'])
            self.assertEqual(len(list((folder / 'inspection').iterdir())), 1)
            inspection_path = (folder / report['inspection_report']).with_suffix('.json')
            inspection = json.loads(inspection_path.read_text(encoding='utf-8'))
            self.assertEqual([position['state'] for position in inspection['positions']], [PRESENT] * 3)
            self.assertTrue(score(Case('baseline', 0), report, folder)['correct'])
            for site, position in report['positions'].items():
                self.assertEqual(position['status'], 'installed')
                site_events = [event for event in report['events'] if event['site'] == site]
                self.assertGreater(site_events[0]['bit_contact_n'], 0.15)
                self.assertGreater(site_events[1]['screw_z_m'] - site_events[0]['screw_z_m'], 0.05)
                final = report['final'][site]
                self.assertAlmostEqual(final['insertion_depth_mm'], 6, delta=0.25)
                self.assertAlmostEqual(final['insertion_depth_mm'], PITCH_MM * final['thread_turns'], delta=0.25)
                self.assertFalse(final['magnet'])
                self.assertTrue(final['thread'])
                self.assertGreaterEqual(position['torque_cutoff']['spindle_torque_nm'], TORQUE_TARGET_NM)
                self.assertGreaterEqual(position['torque_cutoff']['seat_force_n'], 4)
            # A controller claiming pass without its resulting photos must fail acceptance.
            missing_photo = inspection_path.parent / inspection['positions'][0]['photo']
            missing_photo.unlink()
            self.assertFalse(score(Case('baseline', 0), report, folder)['correct'])
        rig.reset_cycle()
        self.assertTrue(all(not rig.snapshot(station)['thread'] for station in rig.stations.values()))
        self.assertTrue(all(result['status'] == 'pending' for result in rig.results.values()))

    def test_misaligned_screw_never_engages_or_turns(self):
        rig = Assembly('misaligned', fault_site='S3', image_fault='glare', image_site='S1')
        self.addCleanup(rig.close)
        with tempfile.TemporaryDirectory() as tmp:
            folder, report = run_round(rig, Detector(PROFILE), Path(tmp), None, False)
            self.assertEqual(report['status'], 'aborted')
            self.assertIn('对孔超差', report['error'])
            self.assertIsNone(report['inspection_started_at_s'])
            self.assertIsNone(report['inspection_report'])
            self.assertFalse((folder / 'inspection').exists())
            self.assertEqual([result['status'] for result in report['positions'].values()],
                             ['installed', 'installed', 'failed'])
            self.assertFalse(report['final']['S3']['thread'])
            self.assertEqual(report['final']['S3']['thread_turns'], 0)
            self.assertEqual(rig.data.ctrl[rig.motor], 0)
            case = Case('misaligned_S3', 0, 'aborted', 'misaligned', 'S3', 'glare', 'S1')
            self.assertTrue(score(case, report, folder)['correct'])

    def test_obstruction_ends_cycle_before_torque_acceptance(self):
        rig = Assembly('blocked', ('S2',))
        self.addCleanup(rig.close)
        with self.assertRaisesRegex(AssemblyStopped, '即触碰障碍'):
            rig.assemble()
        self.assertGreater(rig.snapshot()['obstruction_force_n'], 3)
        self.assertLess(rig.snapshot()['insertion_depth_mm'], 4)
        self.assertIsNone(rig.results['S2']['torque_cutoff'])
        self.assertEqual(rig.data.ctrl[rig.motor], 0)


if __name__ == '__main__':
    unittest.main()
