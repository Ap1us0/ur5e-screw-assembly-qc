import unittest
from contextlib import closing

from inspection.assembly import Assembly, AssemblyStopped
from inspection.force_study import retraction_verified


class ForceInsertionTests(unittest.TestCase):
    def test_lateral_contact_reduces_entering_error_before_rotation(self):
        with closing(Assembly(sites=('S2',), insertion='force', entry_offset_mm=(1.1, 0), seed=106)) as rig:
            rig.assemble()
            initial = next(e for e in rig.events if e['event'] == 'axis_alignment')
            entry = next(e for e in rig.events if e['event'] == 'probe_complete')
            corrections = [e for e in rig.events if e['event'] == 'force_correction']
            self.assertGreater(initial['radial_error_mm'], 0.65)
            self.assertLessEqual(entry['radial_error_mm'], 0.35)
            self.assertTrue(corrections)
            self.assertTrue(all(e['spindle_velocity_command_rad_s'] == 0 for e in corrections))
            self.assertEqual(rig.results['S2']['status'], 'installed')

    def test_retry_uses_same_screw_and_resets_obstruction_for_next_cycle(self):
        with closing(Assembly('retryable_jam', ('S2',), insertion='force', seed=106)) as rig:
            for _ in range(2):
                self.assertLess(rig.data.mocap_pos[rig.jam_mocap, 2], -0.03)
                rig.assemble()
                self.assertTrue(retraction_verified(rig.events, 1))
                names = [e['event'] for e in rig.events]
                self.assertEqual(names.count('pickup_contact'), 1)
                self.assertEqual(names.count('fault_fixture_extended'), 1)
                self.assertEqual(names.count('fault_fixture_withdrawn'), 1)
                self.assertEqual([attempt['status'] for attempt in rig.results['S2']['attempts']], ['jammed', 'seated'])
                rig.reset_cycle()

    def test_permanent_obstruction_retracts_twice_and_never_accepts_torque(self):
        with closing(Assembly('blocked', ('S2',), insertion='force', seed=106)) as rig:
            with self.assertRaises(AssemblyStopped) as stopped:
                rig.assemble()
            self.assertEqual(stopped.exception.code, 'jam_retry_limit')
            self.assertTrue(retraction_verified(rig.events, 1))
            self.assertTrue(retraction_verified(rig.events, 2))
            self.assertIsNone(rig.results['S2']['torque_cutoff'])
            self.assertEqual(rig.results['S2']['status'], 'review')
