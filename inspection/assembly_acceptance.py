import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

from .assembly import DEPTH_M, PITCH_MM, TORQUE_MAX_NM, TORQUE_TARGET_NM
from .assembly_run import run_assembly
from .detector import Detector, PRESENT, REVIEW, SITES
from .samples import write_csv, write_json
from .workflow import new_run


@dataclass(frozen=True)
class Case:
    label: str
    seed: int
    outcome: str = 'pass'
    fault: str = 'none'
    fault_site: str = 'S2'
    image_fault: str = 'none'
    image_site: str = 'S2'


MATRIX = (
    Case('baseline', 0),
    *(Case(f'tolerance_{seed}', seed) for seed in (102, 103, 104)),
    *(Case(f'{fault}_{site}', seed, image_fault=fault, image_site=site)
      for fault, first_seed in (('glare', 105), ('blur', 108))
      for seed, site in enumerate(SITES, first_seed)),
    Case('all_angles_bad_S1', 111, 'review', image_fault='both_bad', image_site='S1'),
    Case('all_angles_bad_S3', 112, 'review', image_fault='both_bad', image_site='S3'),
    *(Case(f'{fault}_{site}', seed, 'aborted', fault, site)
      for fault, first_seed in (('misaligned', 113), ('blocked', 116))
      for seed, site in enumerate(SITES, first_seed)),
    Case('empty_feeder_S1', 119, 'aborted', 'empty_feeder', 'S1'),
    Case('empty_feeder_S3', 120, 'aborted', 'empty_feeder', 'S3'),
)
MIN_CORRECT = 18
STOP_CODES = {'misaligned': 'alignment', 'blocked': 'obstruction', 'empty_feeder': 'pickup_contact'}
LIFECYCLE = ('pickup_contact', 'screw_lifted', 'axis_alignment', 'thread_engaged', 'torque_cutoff',
             'spindle_stopped', 'driver_released', 'retention_verified')


def mechanical_evidence(site, result):
    events = [event for event in result['events'] if event['site'] == site]
    sequence = tuple(event['event'] for event in events if event['event'] in LIFECYCLE)
    summary = result['positions'][site]
    cutoff, final = summary['torque_cutoff'], result['final'][site]
    return bool(sequence == LIFECYCLE and cutoff and summary['after_retract']
                and events[0]['bit_contact_n'] >= 0.15
                and cutoff['seat_force_n'] >= 4.0 and cutoff['thread_rate_rad_s'] <= 0.70
                and TORQUE_TARGET_NM <= cutoff['spindle_torque_nm'] <= TORQUE_MAX_NM
                and abs(cutoff['insertion_depth_mm'] - DEPTH_M * 1000) <= 0.60
                and abs(cutoff['insertion_depth_mm'] - PITCH_MM * cutoff['thread_turns']) <= 0.25
                and abs(final['insertion_depth_mm'] - DEPTH_M * 1000) <= 0.55
                and final['radial_error_mm'] <= 0.30 and final['thread'] and not final['magnet'])


def optical_evidence(case, result, folder):
    path = (folder / result['inspection_report']).with_suffix('.json')
    vision = json.loads(path.read_text(encoding='utf-8-sig'))
    if (tuple(p['site'] for p in vision['positions']) != SITES or vision['source'] != 'simulation'
            or vision['error'] or vision['verdict'] != result['inspection_verdict']):
        return False
    for position in vision['positions']:
        targeted = case.image_fault != 'none' and position['site'] == case.image_site
        needs_review = targeted and case.image_fault == 'both_bad'
        attempts = position['attempts']
        expected_angles = ('primary', 'alternate') if targeted else ('primary',)
        if (tuple(attempt['angle'] for attempt in attempts) != expected_angles
                or position['state'] != (REVIEW if needs_review else PRESENT)
                or not all(a['photo'] and (path.parent / a['photo']).is_file() for a in attempts)):
            return False
        if targeted and attempts[0]['state'] != REVIEW:
            return False
    return True


def score(case, result, folder):
    statuses = [result['positions'][site]['status'] for site in SITES]
    cutoff_verified = all(mechanical_evidence(site, result) for site, state in zip(SITES, statuses)
                          if state == 'installed')
    motor_stopped = all(abs(row['spindle_velocity_rad_s']) <= 0.08 and row['spindle_velocity_command_rad_s'] == 0
                        for row in result['final'].values())
    if case.outcome == 'aborted':
        failed = SITES.index(case.fault_site)
        expected_statuses = ['installed'] * failed + ['failed'] + ['pending'] * (2 - failed)
        fault_events = [event['event'] for event in result['events'] if event['site'] == case.fault_site]
        evidence = (statuses == expected_statuses and cutoff_verified and motor_stopped
                    and result['stop_code'] == STOP_CODES[case.fault]
                    and result['inspection_started_at_s'] is None and result['inspection_report'] is None
                    and 'torque_cutoff' not in fault_events and 'spindle_stopped_after_abort' in fault_events)
        if case.fault != 'blocked':
            evidence = evidence and 'thread_engaged' not in fault_events
    else:
        retained = [e for e in result['events'] if e['event'] == 'retention_verified']
        evidence = (statuses == ['installed'] * 3 and cutoff_verified and motor_stopped
                    and tuple(e['site'] for e in retained) == SITES and not result['error']
                    and result['inspection_started_at_s'] is not None
                    and result['inspection_started_at_s'] >= retained[-1]['sim_time_s']
                    and bool(result['inspection_report']) and optical_evidence(case, result, folder))
    evidence = (bool(evidence) and tuple(result['order']) == SITES and result['source'] == 'simulation'
                and result['seed'] == case.seed and result['scenario'] == case.fault
                and result['fault_site'] == (case.fault_site if case.fault != 'none' else None)
                and result['image_fault'] == case.image_fault and result['image_site'] == case.image_site)
    observed = result['status']
    return dict(observed=observed, matched=observed == case.outcome, evidence_ok=bool(evidence),
                correct=observed == case.outcome and bool(evidence), false_pass=case.outcome != 'pass' and observed == 'pass',
                installed=statuses.count('installed'), stop_code=result['stop_code'],
                inspection=result['inspection_verdict'], error=result['error'])


def evaluate_case(case, profile, root, motion='optimized'):
    status = 'error'
    path = None
    cycle_time = assembly_time = settled = None
    try:
        folder, result = run_assembly(Detector(profile), root / 'cases' / case.label, case.fault,
                                      fault_site=case.fault_site, seed=case.seed,
                                      image_fault=case.image_fault, image_site=case.image_site, motion=motion)
        status = result['status']
        path = (folder / 'report.html').relative_to(root).as_posix()
        outcome = score(case, result, folder)
        if result['motion'] != motion:
            raise ValueError('运动曲线与验收设置不符')
        cycle_time = result['cycle_time_s']
        assembly_time = result['inspection_started_at_s']
        settled = sum(move['settling_s'] for move in result['moves'])
    except Exception as exc:
        outcome = dict(observed=status, matched=status == case.outcome, evidence_ok=False, correct=False,
                       false_pass=case.outcome != 'pass' and status == 'pass',
                       installed=0, stop_code=None, inspection=None, error=f'{type(exc).__name__}: {exc}')
    case_row = {**asdict(case), **outcome, 'cycle_time_s': cycle_time, 'assembly_time_s': assembly_time,
                'arm_settling_s': settled, 'report': path}
    write_json(root / 'cases' / case.label / 'case.json', case_row)
    return case_row


def write_dashboard(folder, result):
    rows = []
    for case in result['cases']:
        href = f'<a href="{escape(case["report"])}">查看</a>' if case['report'] else '—'
        duration = f'{case["cycle_time_s"]:.2f} s' if case['cycle_time_s'] is not None else '—'
        rows.append(f'<tr><td>{case["label"]}</td><td>{case["seed"]}</td><td>{case["outcome"]}</td>'
                    f'<td>{case["observed"]}</td><td>{case["installed"]}</td>'
                    f'<td>{"通过" if case["evidence_ok"] else "缺失"}</td><td>{duration}</td><td>{href}</td></tr>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>UR5e 装配仿真验收</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{background:#f3f6f6;color:#193039;font:16px/1.6 "Microsoft YaHei",sans-serif;margin:0}}
main{{max-width:1000px;margin:auto;padding:28px 18px}}h1{{margin:.15em 0}}table{{background:white;border-collapse:collapse;width:100%}}
th,td{{padding:8px;text-align:left;border-bottom:1px solid #d8e3e5}}a{{color:#165669}}</style>
<main><p>MuJoCo · 固定夹具 · UR5e · {escape(result['motion'])}</p><h1>装配与质检闭环验收</h1>
<p>正确且证据完整 {result['correct']}/{result['rounds']} · 误放行 {result['false_pass']} ·
{'通过' if result['passed'] else '未通过'}</p><p><a href="matrix.json">冻结工况</a> ·
<a href="acceptance.csv">验收汇总 CSV</a> · <a href="acceptance.json">验收汇总 JSON</a></p>
<table><tr><th>工况</th><th>种子</th><th>预期</th><th>实际</th><th>已装配</th><th>证据</th><th>仿真时长</th><th>单轮报告</th></tr>
{''.join(rows)}</table></main></html>'''
    (folder / 'report.html').write_text(page, encoding='utf-8')


def run_acceptance(profile, output, jobs, motion='optimized'):
    if not 1 <= jobs <= 8:
        raise ValueError('并发子进程数应在 1 到 8 之间')
    folder = new_run(output)
    print(f'验收批次 · {folder}', flush=True)
    for case in MATRIX:
        (folder / 'cases' / case.label).mkdir(parents=True)
    write_json(folder / 'matrix.json', dict(source='simulation', motion=motion, cases=[asdict(case) for case in MATRIX],
                                             minimum_correct=MIN_CORRECT, maximum_false_pass=0))
    if jobs == 1:
        cases = [evaluate_case(case, profile, folder, motion) for case in MATRIX]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(evaluate_case, case, profile, folder, motion) for case in MATRIX]
            cases = [future.result() for future in futures]
    correct = sum(case['correct'] for case in cases)
    false_pass = sum(case['false_pass'] for case in cases)
    result = dict(source='simulation', motion=motion, rounds=len(cases), correct=correct, false_pass=false_pass,
                  expected_counts={kind: sum(c['outcome'] == kind for c in cases) for kind in ('pass', 'review', 'aborted')},
                  observed_counts={kind: sum(c['observed'] == kind for c in cases)
                                   for kind in ('pass', 'review', 'aborted', 'error')},
                  passed=correct >= MIN_CORRECT and false_pass == 0, cases=cases)
    write_json(folder / 'acceptance.json', result)
    write_csv(folder / 'acceptance.csv', tuple(cases[0]), cases)
    write_dashboard(folder, result)
    return folder, result
