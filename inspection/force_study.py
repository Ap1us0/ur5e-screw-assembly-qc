import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from html import escape

from .assembly_run import run_assembly
from .assembly_acceptance import Case, mechanical_evidence, optical_evidence
from .detector import Detector, SITES
from .force_insertion import FORCE_ENTRY_TOLERANCE_M, MAX_ATTEMPTS
from .samples import write_csv, write_json
from .workflow import new_run


@dataclass(frozen=True)
class Trial:
    label: str
    site: str = 'S2'
    seed: int = 106
    fault: str = 'none'
    offset: tuple[float, float] = (0.0, 0.0)


TRIALS = (
    Trial('nominal', seed=0),
    Trial('offset_y', 'S1', offset=(0.0, -1.1)),
    Trial('offset_x', 'S2', offset=(1.1, 0.0)),
    Trial('offset_diagonal', 'S3', offset=(-0.8, 0.75)),
    *(Trial(f'retryable_{site}', site, seed, 'retryable_jam')
      for site, seed in zip(SITES, (0, 106, 118))),
    *(Trial(f'blocked_{site}', site, seed, 'blocked')
      for site, seed in zip(SITES, (116, 117, 106))),
    Trial('outside_capture', 'S1', offset=(2.5, 0.0)),
)
EXPECTED = {
    'nominal': ('pass', 'pass'),
    'offset': ('aborted', 'pass'),
    'retryable': ('aborted', 'pass'),
    'blocked': ('aborted', 'review'),
    'outside': ('aborted', 'aborted'),
}


def retraction_verified(events, attempt):
    names = ('jam_detected', 'jam_spindle_stopped', 'unwind_started', 'unwind_complete', 'retry_retracted')
    recovery = [e for e in events if e['attempt'] == attempt and e['event'] in names]
    if tuple(e['event'] for e in recovery) != names:
        return False
    jam, stopped, reverse, unwound, retracted = recovery
    return (jam['sensed_torque_nm'] >= 0.155 and abs(jam['tool_fz_n']) >= 2.5
            and jam['sensed_tip_travel_mm'] < 5.25 and 0.45 < jam['insertion_depth_mm'] < 5.3
            and all(abs(e['spindle_velocity_rad_s']) <= 0.08 and e['spindle_velocity_command_rad_s'] == 0
                    for e in (stopped, reverse, unwound, retracted))
            and abs(unwound['insertion_depth_mm']) <= 0.45 and unwound['thread']
            and retracted['insertion_depth_mm'] <= -12 and retracted['magnet'] and not retracted['thread']
            and not any(e['event'] == 'torque_cutoff' and e['attempt'] == attempt for e in events))


def evidence(trial, mode, expected, result, folder):
    kind = trial.label.split('_')[0]
    events = [e for e in result['events'] if e['site'] == trial.site]
    names = [event['event'] for event in events]
    installed = [site for site, p in result['positions'].items() if p['status'] == 'installed']
    common = (result['status'] == expected and all(mechanical_evidence(site, result) for site in installed)
              and all(abs(row['spindle_velocity_rad_s']) <= 0.08 and row['spindle_velocity_command_rad_s'] == 0
                      for row in result['final'].values()))
    if not common:
        return False
    if expected == 'pass':
        if installed != list(SITES) or not result['inspection_report'] or result['inspection_verdict'] != 'pass':
            return False
        last_release = max(e['sim_time_s'] for e in result['events'] if e['event'] == 'retention_verified')
        if result['inspection_started_at_s'] < last_release or not optical_evidence(Case(trial.label, trial.seed), result, folder):
            return False
    elif (installed != list(SITES[:SITES.index(trial.site)]) or result['inspection_started_at_s'] is not None
          or result['inspection_report'] or 'torque_cutoff' in names
          or ('spindle_stopped_for_review' if expected == 'review' else 'spindle_stopped_after_abort') not in names):
        return False
    if mode == 'fixed':
        stop_code = {'offset': 'alignment', 'outside': 'alignment', 'retryable': 'obstruction', 'blocked': 'obstruction'}
        return result['stop_code'] == stop_code.get(kind) and 'retry_retracted' not in names
    if kind == 'offset':
        corrections = [e for e in events if e['event'] == 'force_correction']
        axis, probe = (next(e for e in events if e['event'] == marker) for marker in ('axis_alignment', 'probe_complete'))
        return (bool(corrections) and axis['radial_error_mm'] > 0.65
                and probe['radial_error_mm'] <= 1000 * FORCE_ENTRY_TOLERANCE_M
                and all(e['spindle_velocity_command_rad_s'] == 0 for e in corrections))
    if kind == 'retryable':
        return (retraction_verified(events, 1)
                and [p['status'] for p in result['positions'][trial.site]['attempts']] == ['jammed', 'seated']
                and names.count('pickup_contact') == 1 and names.count('fault_fixture_withdrawn') == 1
                and names.index('retry_retracted') < names.index('fault_fixture_withdrawn')
                < max(i for i, name in enumerate(names) if name == 'insertion_attempt') < names.index('torque_cutoff'))
    if kind == 'blocked':
        return (result['stop_code'] == 'jam_retry_limit' and names.count('insertion_attempt') == MAX_ATTEMPTS
                and names.count('pickup_contact') == 1 and 'fault_fixture_withdrawn' not in names
                and all(retraction_verified(events, attempt) for attempt in range(1, MAX_ATTEMPTS + 1)))
    if kind == 'outside':
        return (result['stop_code'] in ('thread_entry', 'probe_travel', 'probe_timeout', 'probe_load')
                and 'probe_retracted' in names and 'thread_engaged' not in names)
    return 'jam_detected' not in names


def evaluate(trial, mode, profile, root):
    folder, result = run_assembly(Detector(profile), root / 'cases' / trial.label / mode, trial.fault,
                                  fault_site=trial.site, seed=trial.seed, insertion=mode,
                                  entry_offset_mm=trial.offset, entry_offset_site=trial.site)
    expected = EXPECTED[trial.label.split('_')[0]][0 if mode == 'fixed' else 1]
    events = [event for event in result['events'] if event['site'] == trial.site]
    initial = next((e['radial_error_mm'] for e in events if e['event'] == 'axis_alignment'), None)
    entry = next((e['radial_error_mm'] for e in events if e['event'] in ('probe_complete', 'thread_engaged')), None)
    corrections = [e for e in events if e['event'] == 'force_correction']
    return dict(case=trial.label, site=trial.site, seed=trial.seed, fault=trial.fault, mode=mode,
                x_mm=trial.offset[0], y_mm=trial.offset[1], initial_mm=initial, entry_mm=entry,
                lateral_peak_n=max((e['averaged_lateral_n'] for e in corrections), default=None),
                correction_mm=math.hypot(*(sum(e['correction_mm'][axis] for e in corrections) for axis in range(2))),
                recovered=sum(e['event'] == 'retry_retracted' for e in events),
                stop_code=result['stop_code'], expected=expected, observed=result['status'],
                correct=evidence(trial, mode, expected, result, folder),
                false_pass=expected != 'pass' and result['status'] == 'pass', time_s=result['cycle_time_s'],
                report=(folder / 'report.html').relative_to(root).as_posix())


def write_dashboard(folder, study):
    def describe(row):
        terms = [{'pass': '完成三颗并通过质检', 'review': '待复检'}.get(row['observed'], '中止')]
        if row['recovered']:
            terms.append(f'{row["recovered"]} 次停转退牙')
        if row['correction_mm'] >= 0.05 and row['entry_mm'] is not None:
            terms.append(f'{row["initial_mm"]:.2f} → {row["entry_mm"]:.2f} mm')
        state = 'ok' if row['correct'] else 'bad'
        return f'<td class="{state}"><a href="{escape(row["report"])}">{" · ".join(terms)}</a></td>'

    group_labels = {'nominal': '正常', 'offset': '轻微偏孔', 'retryable': '夹具可撤障碍',
                    'blocked': '永久障碍', 'outside': '大幅偏孔'}
    rows = ''.join(
        f'<tr><th>{group_labels[trial.label.split("_")[0]]} · {trial.site}</th>'
        f'<td>({trial.offset[0]:g}, {trial.offset[1]:g})</td>{describe(fixed)}{describe(force)}</tr>'
        for trial, fixed, force in zip(TRIALS, study['cases'][::2], study['cases'][1::2])
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>UR5e 力反馈装配对照</title>
<style>body{{background:#f4f8f7;color:#172f39;margin:0;font:16px/1.6 'Microsoft YaHei',sans-serif}}
main{{max-width:1000px;margin:auto;padding:28px 18px}}h1{{margin:.15em 0}}a{{color:#135969}}
table{{width:100%;border-collapse:collapse;background:#fff}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #dbe6e6}}
thead{{background:#deecee}}.bad{{background:#f9deda}}.ok{{background:#f4faf6}}</style>
<main><p>MuJoCo · UR5e · 三螺丝闭环</p><h1>电批力反馈装配对照</h1>
<p>符合预期并有证据 {study['correct']}/{study['rounds']} 轮，误放行 {study['false_pass']} 轮。</p>
<p>每一行的两种模式使用相同场景、扰动和物理模型。括号为目标在 XY 方向注入的偏差，箭头为接触找正前后相对孔心的仿真误差。可撤障碍由仿真夹具在首轮退回后撤除。</p>
<p><a href="study.csv">CSV 明细</a> · <a href="study.json">JSON 与工况</a></p>
<table><thead><tr><th>工况</th><th>注入偏差（mm）</th><th>固定入孔</th><th>力反馈入孔</th></tr></thead>
<tbody>{rows}</tbody></table></main></html>"""
    (folder / 'report.html').write_text(page, encoding='utf-8')


def run_study(profile, output, jobs):
    if jobs < 1:
        raise ValueError('并发仿真进程数必须大于零')
    folder = new_run(output)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(evaluate, trial, mode, profile, folder)
                   for trial in TRIALS for mode in ('fixed', 'force')]
        rows = [future.result() for future in futures]
    correct = sum(row['correct'] for row in rows)
    false_pass = sum(row['false_pass'] for row in rows)
    study = dict(source='simulation', pairs=len(TRIALS), rounds=len(rows), correct=correct, false_pass=false_pass,
                 passed=correct == len(rows) and false_pass == 0, matrix=[asdict(trial) for trial in TRIALS], cases=rows)
    write_csv(folder / 'study.csv', tuple(rows[0]), rows)
    write_json(folder / 'study.json', study)
    write_dashboard(folder, study)
    return folder, study
