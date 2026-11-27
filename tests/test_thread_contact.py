import unittest
from contextlib import closing

import numpy as np

from inspection.assembly import Assembly
from inspection.thread_study import recovered, thread_jam_evidence


class ThreadRecoveryTests(unittest.TestCase):
    def test_angled_screw_binds_on_static_threads_and_reseats_after_axis_correction(self):
        with closing(Assembly('cross_thread', fault_site='S2', seed=106, insertion='force')) as rig:
            female = rig.female_flanks
            self.assertGreater(len(female), 0)
            self.assertEqual(rig.model.nmocap, 0)
            self.assertTrue(np.all(rig.model.geom_bodyid[female] == 0))
            self.assertTrue(all(station.block == -1 for station in rig.stations.values()))
            before = rig.data.geom_xpos[female].copy()

            rig.assemble()
            events = [e for e in rig.events if e['site'] == 'S2']
            jams = [e for e in events if e['event'] == 'jam_detected']
            self.assertEqual(len(jams), 1)
            self.assertTrue(thread_jam_evidence(jams[0]))
            self.assertTrue(recovered(events, 1))
            self.assertEqual(sum(e['event'] == 'pickup_contact' for e in events), 1)
            corrected = next(e for e in events if e['event'] == 'retry_axis_corrected')
            self.assertLess(corrected['tool_tilt_deg'], 0.5)
            self.assertEqual([attempt['status'] for attempt in rig.results['S2']['attempts']], ['jammed', 'seated'])
            self.assertTrue(all(result['status'] == 'installed' for result in rig.results.values()))
            np.testing.assert_array_equal(before, rig.data.geom_xpos[female])

            rig.reset_cycle()
            np.testing.assert_array_equal(before, rig.data.geom_xpos[female])
            self.assertFalse(rig.data.eq_active[rig.stations['S2'].guide])
            self.assertEqual(rig.results['S2']['attempts'], [])
