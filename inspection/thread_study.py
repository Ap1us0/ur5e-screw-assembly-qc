from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from html import escape

from .assembly_acceptance import Case, mechanical_evidence, optical_evidence
from .assembly_run import run_assembly
from .detector import Detector, SITES
from .force_study import retraction_verified
from .samples import write_csv, write_json
from .thread_contact import REALIGN_TILT_DEG, RECOVERY_MODES
from .workflow import new_run


@dataclass(frozen=True)
class ThreadTrial:
    label: str
    site: str
    tilt: tuple[float, float]
    seed: int = 106
    jam: bool = True


TRIALS = (
    ThreadTrial('straight', 'S2', (0, 0), jam=False),
    ThreadTrial('pitch_S1', 'S1', (0, -5.45)),
    ThreadTrial('diagonal_S1', 'S1', (3.8, 3.8)),
    ThreadTrial('roll_S2', 'S2', (5.25, 0)),
    ThreadTrial('diagonal_S3', 'S3', (3.6, 3.6), seed=118),
)


def stopped(result):
    return all(abs(row['spindle_velocity_rad_s']) <= 0.08 and row['spindle_velocity_command_rad_s'] == 0
               for row in result['final'].values())


def thread_jam_evidence(event):
    return (event['obstruction_force_n'] == 0 and event['flank_force_n'] > 10
            and event['sensed_torque_nm'] >= 0.155 and abs(event['spindle_velocity_rad_s']) < 0.8
            and event['tool_tilt_deg'] > 2 and 0.45 < event['insertion_depth_mm'] < 5.3
            and event['spindle_velocity_command_rad_s'] > 0 and event['magnet'] and event['thread'])


def recovered(events, attempt):
    reversing = [e for e in events if e['event'] == 'unwind_reverse_measured' and e['attempt'] == attempt]
    return (retraction_verified(events, attempt) and len(reversing) == 1
            and reversing[0]['spindle_velocity_rad_s'] < 0 and reversing[0]['spindle_velocity_command_rad_s'] < 0
            and reversing[0]['magnet'] and reversing[0]['thread'])


def verify(trial, mode, result, folder, events):
    model = result['thread_model']
    if (not model or not model['all_world_geoms'] or not model['unchanged'] or model['female_flanks'] <= 0
            or model['mocap_bodies'] or model['obstruction_geoms'] or not stopped(result)
            or any(e['event'].startswith('fault_fixture_') for e in result['events'])):
        return False
    jams = [e for e in events if e['event'] == 'jam_detected']
    entries = [e for e in events if e['event'] == 'thread_engaged']
    names = [e['event'] for e in events]
    positions = result['positions']
    installed = [site for site, p in positions.items() if p['status'] == 'installed']
    if not all(mechanical_evidence(site, result) for site in installed) or names.count('pickup_contact') != 1:
        return False
    if not trial.jam or mode == 'realign':
        if (result['status'] != 'pass' or installed != list(SITES) or not result['inspection_report']
                or result['inspection_verdict'] != 'pass' or not optical_evidence(Case(trial.label, trial.seed), result, folder)):
            return False
        last_retained = max(e['sim_time_s'] for e in result['events'] if e['event'] == 'retention_verified')
        if result['inspection_started_at_s'] < last_retained:
            return False
    else:
        code = 'thread_jam' if mode == 'stop' else 'jam_retry_limit'
        outcome = 'aborted' if mode == 'stop' else 'review'
        stop_event = 'spindle_stopped_after_abort' if mode == 'stop' else 'spindle_stopped_for_review'
        if (result['status'] != outcome or result['stop_code'] != code
                or installed != list(SITES[:SITES.index(trial.site)]) or result['inspection_started_at_s'] is not None
                or result['inspection_report'] or 'torque_cutoff' in names or stop_event not in names):
            return False
    if not trial.jam:
        return len(entries) == 1 and not jams and 'retry_retracted' not in names
    if not jams or not all(thread_jam_evidence(e) for e in jams):
        return False
    if mode == 'stop':
        return len(jams) == 1 and len(entries) == 1 and 'unwind_started' not in names
    if mode == 'retry':
        return (len(jams) == 2 and len(entries) == 2 and 'retry_axis_corrected' not in names
                and abs(entries[0]['tool_tilt_deg'] - entries[1]['tool_tilt_deg']) < 0.4
                and all(recovered(events, attempt) for attempt in (1, 2)))
    if len(jams) != 1 or len(entries) != 2 or not recovered(events, 1):
        return False
    corrected = next((e for e in events if e['event'] == 'retry_axis_corrected'), None)
    return (corrected is not None and corrected['magnet'] and not corrected['thread']
            and corrected['tool_tilt_deg'] <= REALIGN_TILT_DEG and entries[1]['tool_tilt_deg'] <= REALIGN_TILT_DEG
            and names.index('retry_retracted') < names.index('retry_axis_corrected') < names.index('torque_cutoff')
            and [p['status'] for p in positions[trial.site]['attempts']] == ['jammed', 'seated'])


def evaluate(trial, mode, profile, root):
    folder, result = run_assembly(Detector(profile), root / 'cases' / trial.label / mode, 'cross_thread',
                                  fault_site=trial.site, seed=trial.seed, insertion='force',
                                  recovery=mode, entry_tilt_deg=trial.tilt)
    events = [e for e in result['events'] if e['site'] == trial.site]
    jams = [e for e in events if e['event'] == 'jam_detected']
    before = next((e['tool_tilt_deg'] for e in events if e['event'] == 'thread_engaged'), None)
    after = next((e['tool_tilt_deg'] for e in events if e['event'] == 'retry_axis_corrected'), None)
    return dict(case=trial.label, site=trial.site, seed=trial.seed, recovery=mode, initial_tilt_deg=before,
                corrected_tilt_deg=after, jams=len(jams), first_jam_depth_mm=jams[0]['insertion_depth_mm'] if jams else None,
                first_flank_load_n=jams[0]['flank_force_n'] if jams else None,
                first_jam_torque_nm=jams[0]['spindle_torque_nm'] if jams else None,
                first_jam_trigger=jams[0]['jam_trigger'] if jams else None,
                reversed=sum(e['event'] == 'unwind_reverse_measured' for e in events),
                stop_code=result['stop_code'], observed=result['status'], installed=sum(
                    p['status'] == 'installed' for p in result['positions'].values()),
                correct=verify(trial, mode, result, folder, events),
                report=(folder / 'report.html').relative_to(root).as_posix())


def write_dashboard(folder, study):
    def describe(row):
        outcome = {'pass': '三颗合格', 'review': '待复检'}.get(row['observed'], '中止')
        details = f' · 退牙 {row["reversed"]} 次' if row['reversed'] else ''
        if row['corrected_tilt_deg'] is not None:
            details += f' · 扶正至 {row["corrected_tilt_deg"]:.2f}°'
        state = 'ok' if row['correct'] else 'bad'
        return f'<td class="{state}"><a href="{escape(row["report"])}">{outcome}{details}</a></td>'

    lines = []
    for index, trial in enumerate(TRIALS):
        group = study['cases'][3 * index:3 * index + 3]
        title = '垂直入牙' if not trial.jam else '斜咬牙'
        load = group[0]['first_flank_load_n']
        initial_load = f'{load:.1f} N' if load is not None else '—'
        lines.append(f'<tr><th>{trial.site} · {title}</th><td>({trial.tilt[0]:g}°, {trial.tilt[1]:g}°)</td>'
                     f'<td>{initial_load}</td>{"".join(map(describe, group))}</tr>')
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>UR5e 斜咬牙恢复对照</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{background:#f3f7f7;color:#17313a;font:16px/1.6 'Microsoft YaHei',sans-serif;margin:0}}
main{{max-width:1120px;margin:auto;padding:28px 18px}}h1{{margin:.2em 0}}a{{color:#155d70}}
table{{background:#fff;width:100%;border-collapse:collapse}}
td,th{{text-align:left;padding:8px;border-bottom:1px solid #d9e4e5}}thead{{background:#dbe9ed}}
.ok{{background:#f5faf7}}.bad{{background:#fce0d7}}</style>
<main><p>MuJoCo · UR5e · 三螺丝闭环</p><h1>斜咬牙恢复对照</h1>
<p>符合预期及证据 {study['correct']}/{study['rounds']} 轮。</p>
<p>每行三次试验使用同一螺丝、固定内牙、入孔姿态和随机种子。斜咬牙轮没有挡块和可移动的测试夹具。最后一列从退回后重新测量、扶正电批轴线开始，随后再次找孔和旋入。</p>
<p><a href="study.csv">CSV 明细</a> · <a href="study.json">JSON 与工况</a></p>
<table><thead><tr><th>位置</th><th>入牙倾角 X/Y</th><th>首次卡滞牙侧力</th><th>只停转</th>
<th>原姿态重试</th><th>扶正后重试</th></tr></thead><tbody>{''.join(lines)}</tbody></table></main></html>"""
    (folder / 'report.html').write_text(html, encoding='utf-8')


def run_thread_study(profile, output, jobs):
    if jobs < 1:
        raise ValueError('并发仿真进程数必须大于零')
    folder = new_run(output)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        cases = [pool.submit(evaluate, trial, mode, profile, folder) for trial in TRIALS for mode in RECOVERY_MODES]
        rows = [case.result() for case in cases]
    correct = sum(row['correct'] for row in rows)
    study = dict(source='simulation', groups=len(TRIALS), rounds=len(rows), correct=correct,
                 passed=correct == len(rows), matrix=[asdict(trial) for trial in TRIALS], cases=rows)
    write_csv(folder / 'study.csv', tuple(rows[0]), rows)
    write_json(folder / 'study.json', study)
    write_dashboard(folder, study)
    return folder, study
