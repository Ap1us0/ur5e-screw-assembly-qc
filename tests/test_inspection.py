import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inspection.acceptance import evaluate
from inspection.detector import Detector, MISSING, PRESENT, REVIEW, Observation, SITES
from inspection.hardware import StationConfig, StationLink
from inspection.samples import verify_samples, write_csv, write_json
from inspection.urscript import build_script
from inspection.workflow import CameraError, inspect_workpiece


ROOT = Path(__file__).resolve().parents[1]


class StubRig:
    source = 'hardware'

    def __init__(self, observations, finish_error=False):
        self.observations = iter(observations)
        self.calls = []
        self.finish_error = finish_error

    def capture(self, site, angle):
        self.calls.append((site, angle))
        observation = next(self.observations)
        if isinstance(observation, Exception):
            raise observation
        return observation

    def finish(self):
        if self.finish_error:
            raise ConnectionError('return incomplete')

    def close(self):
        self.calls.append('closed')


class WorkflowTests(unittest.TestCase):
    def execute_with_rig(self, rig):
        detector = SimpleNamespace(source='hardware', inspect=lambda _site, _angle, observation: observation)
        with tempfile.TemporaryDirectory() as tmp, patch('inspection.workflow.save_image'):
            folder, result = inspect_workpiece(detector, 'hardware', lambda: rig, tmp)
            self.assertEqual(result, json.loads((folder / 'report.json').read_text(encoding='utf-8')))
        return result

    def test_camera_timeout_retries_only_affected_position(self):
        rig = StubRig([Observation(PRESENT, ''), CameraError('timeout'), Observation(MISSING, ''),
                       Observation(PRESENT, '')])
        result = self.execute_with_rig(rig)
        self.assertEqual(result['verdict'], MISSING)
        self.assertEqual(rig.calls, [('S1', 'primary'), ('S2', 'primary'), ('S2', 'alternate'),
                                     ('S3', 'primary'), 'closed'])
        self.assertEqual(result['positions'][1]['photo'], 'S2_alternate.png')
        self.assertIsNone(result['positions'][1]['attempts'][0]['photo'])

    def test_motion_failure_stops_future_moves(self):
        rig = StubRig([Observation(MISSING, ''), ConnectionError('wrong pose')])
        result = self.execute_with_rig(rig)
        self.assertEqual(result['verdict'], MISSING)
        self.assertEqual(rig.calls, [('S1', 'primary'), ('S2', 'primary'), 'closed'])
        self.assertEqual(result['positions'][2]['state'], REVIEW)
        self.assertIn('wrong pose', result['error'])

    def test_failed_return_cannot_pass(self):
        rig = StubRig([Observation(PRESENT, '') for _ in SITES], finish_error=True)
        result = self.execute_with_rig(rig)
        self.assertEqual(result['verdict'], REVIEW)
        self.assertIn('return incomplete', result['error'])

    def test_bundled_images_cover_both_classes_and_all_views(self):
        sample_dir = ROOT / 'samples' / 'simulation'
        detector = Detector(sample_dir / 'profile.json')
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_samples(detector, sample_dir / 'holdout' / 'samples.csv', Path(tmp) / 'report.json')
        self.assertTrue(result['passed'])
        self.assertEqual(result['classes'], {PRESENT: 6, MISSING: 6, REVIEW: 2})


class StationTests(unittest.TestCase):
    def test_partial_line_requires_complete_exact_pose_acknowledgment(self):
        local, remote = socket.socketpair()
        with local, remote:
            link = StationLink(SimpleNamespace(move_timeout=2))
            link.connection = local
            seen = []

            def reply():
                with remote.makefile('rb') as reader:
                    seen.append(reader.readline())
                    remote.sendall(b'AT ')
                    remote.sendall(b'2\r\n')
                    seen.append(reader.readline())
                    remote.sendall(b'AT 2\n')

            thread = threading.Thread(target=reply)
            thread.start()
            link.move(2)
            with self.assertRaises(ConnectionError):
                link.move(3)
            thread.join(timeout=2)
            self.assertEqual(seen, [b'(2)\n', b'(3)\n'])

    def test_controller_program_binds_to_versioned_taught_routes(self):
        config = json.loads((ROOT / 'station' / 'example.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            station = Path(tmp) / 'station.json'
            station.write_text(json.dumps(config), encoding='utf-8')
            with self.assertRaises(ValueError):
                StationConfig(station)
            config['home'] = [0, 0, 0, 0, 0, 0]
            for index, (site, angle) in enumerate(((s, a) for s in SITES for a in ('primary', 'alternate')), 1):
                config['routes'][site][angle] = [[index / 10, 0, 0, 0, 0, 0]]
            station.write_text(json.dumps(config), encoding='utf-8')
            script = Path(tmp) / 'station.script'
            route = StationConfig(station)
            build_script(station, script)
            generated = script.read_text(encoding='ascii')
            self.assertIn(f'READY {route.route_token}', generated)
            for number in range(7):
                self.assertIn(f'AT {number}', generated)
            with self.assertRaises(FileExistsError):
                build_script(station, script)


class AcceptanceTests(unittest.TestCase):
    def test_single_false_pass_fails_even_with_nineteen_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'image.png').touch()
            cases = []
            for trial in range(20):
                stem = f'run{trial:02}'
                known_missing = trial >= 10
                observed_missing = trial >= 11
                positions = [dict(site=site, photo='image.png',
                                  state=MISSING if observed_missing and site == 'S2' else PRESENT) for site in SITES]
                write_json(root / f'{stem}.json', dict(version=1, source='hardware', run_id=stem,
                           verdict=MISSING if observed_missing else 'pass', error=None, positions=positions))
                cases.append(dict(missing='S2' if known_missing else '', report=f'{stem}.json'))
            manifest = root / 'trials.csv'
            write_csv(manifest, ('missing', 'report'), cases)
            result = evaluate(manifest, root / 'acceptance.json', 'hardware')
            self.assertEqual((result['correct'], result['false_pass'], result['passed']), (19, 1, False))
            with self.assertRaises(ValueError):
                evaluate(manifest, root / 'source.json', 'simulation')


if __name__ == '__main__':
    unittest.main()
