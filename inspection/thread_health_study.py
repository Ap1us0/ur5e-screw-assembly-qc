from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from html import escape

from .assembly_acceptance import Case, mechanical_evidence, optical_evidence
from .assembly_run import run_assembly
from .detector import Detector, SITES
from .samples import write_csv, write_json
from .thread_contact import MAX_ANGLED_ENTRY_DEG
from .thread_health import SLIP_MAX_TORQUE_NM, SLIP_MIN_DEFICIT_MM, SLIP_WINDOW_TURNS
from .workflow import new_run


@dataclass(frozen=True)
class Trial:
    label: str
    fault: str = 'none'
    site: str = 'S2'
    seed: int = 106
    tilt: tuple[float, float] | None = None
    strip_depth_mm: float | None = None


TRIALS = (
    Trial('normal', seed=0),
    Trial('normal_variation', seed=103),
    Trial('cross_S1', 'cross_thread', 'S1', tilt=(0, -5.45)),
    Trial('cross_S2', 'cross_thread', 'S2', tilt=(5.25, 0)),
    Trial('cross_S3', 'cross_thread', 'S3', 118, (3.6, 3.6)),
    Trial('crushed_S1', 'damaged_thread', 'S1'),
    Trial('crushed_S2', 'damaged_thread', 'S2'),
    Trial('crushed_S3', 'damaged_thread', 'S3', 118),
    Trial('crushed_with_tilt', 'damaged_thread', 'S2', tilt=(5.25, 0)),
    Trial('strip_S1', 'stripped_thread', 'S1', strip_depth_mm=1.5),
    Trial('strip_S2', 'stripped_thread', 'S2', strip_depth_mm=2.4),
    Trial('strip_S3', 'stripped_thread', 'S3', 118, strip_depth_mm=3.8),
)
REVIEW_CODES = {'damaged_thread': 'jam_retry_limit', 'stripped_thread': 'thread_slip_retry_limit'}
SCENARIOS = {'none': '正常', 'cross_thread': '可恢复斜咬牙', 'damaged_thread': '牙侧压溃', 'stripped_thread': '滑牙'}


def motion_evidence(events, attempt, reason):
    sequence = (reason, 'jam_spindle_stopped', 'unwind_reverse_measured', 'unwind_complete', 'retry_retracted')
    recovery = [e for e in events if e['attempt'] == attempt and e['event'] in sequence]
    if tuple(e['event'] for e in recovery) != sequence:
        return False
    fault, stopped, reverse, unwound, lifted = recovery
    if not (fault['thread'] and fault['magnet'] and stopped['spindle_velocity_command_rad_s'] == 0
            and abs(stopped['spindle_velocity_rad_s']) <= 0.08
            and reverse['spindle_velocity_command_rad_s'] < 0 and reverse['spindle_velocity_rad_s'] < -0.12
            and abs(unwound['insertion_depth_mm']) < 0.45 and unwound['thread']
            and lifted['magnet'] and not lifted['thread'] and lifted['insertion_depth_mm'] < -12):
        return False
    if reason == 'thread_slip_detected':
        return (fault['window_turns'] >= SLIP_WINDOW_TURNS and fault['window_mean_torque_nm'] <= SLIP_MAX_TORQUE_NM
                and fault['window_pitch_deficit_mm'] >= SLIP_MIN_DEFICIT_MM)
    stalled = (fault['jam_trigger'] == 'stall' and abs(fault['spindle_velocity_rad_s']) <= 0.8
               and abs(fault['window_turns']) <= .02 and abs(fault['window_advance_mm']) <= .13)
    angle_limit = (fault['jam_trigger'] == 'tilt_load' and fault['tilt_deg'] > MAX_ANGLED_ENTRY_DEG + .75
                   and abs(fault['spindle_velocity_rad_s']) < .5 * fault['spindle_velocity_command_rad_s'])
    return (fault['sensed_torque_nm'] >= 0.155 and (stalled or angle_limit)
            and fault['flank_force_n'] > 10 and fault['obstruction_force_n'] == 0)


def verify(trial, result, folder, events):
    reason = 'thread_slip_detected' if trial.strip_depth_mm is not None else 'jam_detected'
    states = [p['status'] for p in result['positions'].values()]
    passed = trial.fault not in REVIEW_CODES
    attempts = [a['status'] for a in result['positions'][trial.site]['attempts']]
    if (result['status'] != ('pass' if passed else 'review')
            or not all(r['spindle_velocity_command_rad_s'] == 0 and abs(r['spindle_velocity_rad_s']) <= .08
                       for r in result['final'].values())
            or not all(mechanical_evidence(site, result) for site, state in zip(SITES, states) if state == 'installed')):
        return False
    names = [e['event'] for e in events]
    if names.count('pickup_contact') != 1 or result['external_jam_fixture']:
        return False
    model = result['thread_model']
    if trial.fault == 'damaged_thread' and (
            not model or not model['unchanged'] or not model['all_world_geoms']
            or not model['damaged_flanks'] or model['obstruction_geoms'] or model['mocap_bodies']):
        return False
    if passed:
        if states != ['installed'] * 3 or not result['inspection_report'] or result['inspection_verdict'] != 'pass':
            return False
        last_installed = max(e['sim_time_s'] for e in result['events'] if e['event'] == 'retention_verified')
        if (result['inspection_started_at_s'] < last_installed
                or not optical_evidence(Case(trial.label, trial.seed), result, folder)):
            return False
        if trial.fault == 'none':
            return attempts == ['seated'] and names.count(reason) == 0 and 'thread_slip_detected' not in names
        return (attempts == ['jammed', 'seated'] and motion_evidence(events, 1, reason)
                and names.index('retry_retracted') < names.index('retry_axis_corrected') < names.index('torque_cutoff'))
    index = SITES.index(trial.site)
    repeated = 'slipped' if trial.strip_depth_mm is not None else 'jammed'
    return (states == ['installed'] * index + ['review'] + ['pending'] * (2 - index)
            and result['mechanical_review_site'] == trial.site and result['stop_code'] == REVIEW_CODES[trial.fault]
            and result['inspection_started_at_s'] is None and result['inspection_verdict'] is None
            and result['inspection_report'] is None and 'torque_cutoff' not in names
            and result['review_photo'] and (folder / result['review_photo']).is_file()
            and 'spindle_stopped_for_review' in names and attempts == [repeated, repeated]
            and all(motion_evidence(events, attempt, reason) for attempt in (1, 2))
            and (trial.tilt is None or 'retry_axis_corrected' in names))


def evaluate(trial, profile, root):
    folder, result = run_assembly(Detector(profile), root / 'cases' / trial.label, trial.fault,
                                  fault_site=trial.site, seed=trial.seed, insertion='force',
                                  entry_tilt_deg=trial.tilt, strip_depth_mm=trial.strip_depth_mm)
    events = [e for e in result['events'] if e['site'] == trial.site]
    first = next((e for e in events if e['event'] in ('thread_slip_detected', 'jam_detected')), {})
    last = result['positions'][trial.site]
    return dict(case=trial.label, fault=trial.fault, site=trial.site, seed=trial.seed,
                strip_depth_mm=trial.strip_depth_mm, actual=result['status'], stop_code=result['stop_code'],
                attempts=' / '.join(a['status'] for a in last['attempts']),
                diagnosis=last['thread_diagnosis'], fault_depth_mm=first.get('encoder_depth_mm'),
                fault_turns=first.get('window_turns'), fault_advance_mm=first.get('window_advance_mm'),
                fault_torque_nm=first.get('window_mean_torque_nm', first.get('sensed_torque_nm')),
                fault_trigger=first.get('jam_trigger', first.get('event')),
                correct=bool(verify(trial, result, folder, events)),
                false_pass=trial.fault in REVIEW_CODES and result['status'] == 'pass',
                report=(folder / 'report.html').relative_to(root).as_posix(),
                signature=(folder / 'fastening_signature.png').relative_to(root).as_posix())


def write_dashboard(folder, result):
    def display(row):
        label = SCENARIOS[row['fault']]
        if row['case'] == 'crushed_with_tilt':
            label += '并斜着入牙'
        torque = row['fault_torque_nm']
        depth = row['fault_depth_mm']
        signature = f'{torque:.3f} N·m，深度 {depth:.2f} mm' if torque is not None and depth is not None else '—'
        if row['fault_turns'] is not None:
            signature += f'，{row["fault_turns"]:.2f} 圈进给 {row["fault_advance_mm"]:.2f} mm'
        elif row['fault_trigger'] == 'tilt_load':
            signature += '，带载偏角越限'
        state = 'ok' if row['correct'] else 'bad'
        verdict = {'pass': '合格', 'review': '待复检'}.get(row['actual'], '中止')
        return (f'<tr class="{state}"><th>{label} · {row["site"]}</th><td>{signature}</td>'
                f'<td>{verdict}</td><td><a href="{escape(row["report"])}">报告</a> · '
                f'<a href="{escape(row["signature"])}">曲线</a></td></tr>')

    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>UR5e 滑牙与牙损质检对照</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{background:#f4f8f8;color:#18323d;margin:0;font:16px/1.6 'Microsoft YaHei',sans-serif}}
main{{max-width:1060px;margin:auto;padding:28px 18px}}h1{{margin:.2em 0}}a{{color:#155c6c}}
table{{width:100%;border-collapse:collapse;background:#fff}}thead{{background:#dcebed}}
th,td{{text-align:left;padding:8px;border-bottom:1px solid #dce6e7}}.bad{{background:#fbe0d8}}</style>
<main><p>MuJoCo · UR5e · 三螺丝闭环</p><h1>牙损、滑牙与斜咬牙对照</h1>
<p>符合预期且证据完整 {result['correct']}/{result['rounds']} 轮，坏牙误放行 {result['false_pass']} 轮。</p>
<p>表中的扭矩、转角和深度来自电批传感数据。牙侧持续卡滞与持续空转都先退牙复试。第二次仍异常则保持工件待复检。</p>
<p><a href="study.csv">CSV 明细</a> · <a href="study.json">JSON 与工况</a></p>
<table><thead><tr><th>工况</th><th>首次异常读数</th><th>整件判定</th><th>证据</th></tr></thead>
<tbody>{''.join(map(display, result['cases']))}</tbody></table></main></html>"""
    (folder / 'report.html').write_text(html, encoding='utf-8')


def run_health_study(profile, output, jobs):
    if jobs < 1:
        raise ValueError('并发仿真进程数必须大于零')
    folder = new_run(output)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(evaluate, trial, profile, folder) for trial in TRIALS]
        rows = [future.result() for future in futures]
    correct = sum(row['correct'] for row in rows)
    false_pass = sum(row['false_pass'] for row in rows)
    result = dict(source='simulation', rounds=len(rows), correct=correct, false_pass=false_pass,
                  passed=correct == len(rows) and not false_pass, matrix=[asdict(trial) for trial in TRIALS], cases=rows)
    write_csv(folder / 'study.csv', tuple(rows[0]), rows)
    write_json(folder / 'study.json', result)
    write_dashboard(folder, result)
    return folder, result
