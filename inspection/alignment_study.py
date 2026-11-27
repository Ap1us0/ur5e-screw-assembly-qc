import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from html import escape

from .assembly_acceptance import Case, score
from .assembly_run import run_assembly
from .detector import Detector, SITES
from .samples import write_csv, write_json
from .vision_alignment import MAX_FIT_RESIDUAL_MM, FixtureShift
from .workflow import new_run


# The 0.20 mm scoring limit leaves margin below the 0.65 mm insertion gate.
MAX_CENTER_ERROR_MM = 0.20


@dataclass(frozen=True)
class ShiftCase:
    name: str
    pose: FixtureShift
    seed: int
    obstruction: str = 'none'
    site: str = 'S2'


CASES = (
    ShiftCase('small_diagonal', FixtureShift(2.5, -1.5, 1.5), 620),
    ShiftCase('positive_x', FixtureShift(10.5, 0, 0), 621),
    ShiftCase('positive_y', FixtureShift(0, 10.3, 0), 622),
    ShiftCase('yaw_only', FixtureShift(0, 0, 4.0), 623),
    ShiftCase('positive_yaw', FixtureShift(7, -4, 3), 624),
    ShiftCase('negative_yaw', FixtureShift(-8, 4, -4), 625),
    ShiftCase('glare_S1', FixtureShift(-9, -2, 4.5), 626, 'glare', 'S1'),
    ShiftCase('glare_S2', FixtureShift(-8, 4, -4), 627, 'glare', 'S2'),
    ShiftCase('glare_S3', FixtureShift(7.5, 6.0, 4.0), 628, 'glare', 'S3'),
    ShiftCase('large_negative_xy', FixtureShift(-10, -5.5, -5), 629),
    ShiftCase('large_positive_xy', FixtureShift(8.0, 7.7, 5.5), 630),
    ShiftCase('negative_y', FixtureShift(1.5, -10.8, -4), 631),
    ShiftCase('occlusion_S1', FixtureShift(5, 2, 2), 632, 'occluded', 'S1'),
    ShiftCase('occlusion_S2', FixtureShift(-5, -2, 3), 633, 'occluded', 'S2'),
    ShiftCase('occlusion_S3', FixtureShift(6, -4, -4), 634, 'occluded', 'S3'),
)


def calibration_evidence(case, result, folder):
    attempts = result['alignment_attempts']
    if any(not all(entry.get(key) and (folder / entry[key]).is_file() for key in ('photo', 'marked_photo'))
           for entry in attempts):
        return False
    actual_sequence = [(entry['site'], entry['angle'], entry['observation'] is not None) for entry in attempts]
    expected = []
    for site in SITES:
        if site != case.site or case.obstruction == 'none':
            expected.append((site, 'primary', True))
            continue
        expected.extend(((site, 'primary', False), (site, 'alternate', case.obstruction == 'glare')))
        if case.obstruction == 'occluded':
            break
    return actual_sequence == expected


def run_case(case, profile, batch):
    detector = Detector(profile)
    basedir = batch / 'cases' / case.name
    row = dict(case=case.name, seed=case.seed, **asdict(case.pose), obstruction=case.obstruction, site=case.site,
               fixed=None, fixed_report=None, vision=None, vision_report=None, max_center_error_mm=None,
               calibration_last_photo_s=None, vision_cycle_time_s=None, correct=False, error=None)
    try:
        controlled_stop = True
        if case.obstruction != 'occluded':
            folder, fixed = run_assembly(detector, basedir / 'fixed', 'none', seed=case.seed, fixture_shift=case.pose)
            row.update(fixed=fixed['status'], fixed_report=(folder / 'report.html').relative_to(batch).as_posix())
            controlled_stop = (fixed['status'] == 'aborted' and fixed['stop_code'] == 'alignment'
                               and fixed['inspection_report'] is None
                               and not any(event['event'] == 'thread_engaged' for event in fixed['events']))
        folder, vision = run_assembly(detector, basedir / 'vision', 'none', seed=case.seed, alignment='vision',
                                      fixture_shift=case.pose, alignment_fault=case.obstruction,
                                      alignment_fault_site=case.site)
        row.update(vision=vision['status'], vision_report=(folder / 'report.html').relative_to(batch).as_posix(),
                   vision_cycle_time_s=vision['cycle_time_s'])
        if vision['alignment_attempts']:
            row['calibration_last_photo_s'] = vision['alignment_attempts'][-1]['at_s']
        has_photos = calibration_evidence(case, vision, folder)
        if case.obstruction == 'occluded':
            row['correct'] = (has_photos and vision['status'] == 'aborted' and vision['stop_code'] == 'registration_image'
                              and vision['registration'] is None and vision['inspection_report'] is None
                              and all(part['status'] == 'pending' for part in vision['positions'].values())
                              and all(event['event'] != 'pickup_contact' for event in vision['events']))
        elif vision['registration'] is not None:
            fit = vision['registration']
            worst = max(fit['actual_center_errors_mm'].values())
            row['max_center_error_mm'] = worst
            lifecycle = score(Case(case.name, case.seed), vision, folder)['correct']
            row['correct'] = (controlled_stop and has_photos and lifecycle
                              and worst <= MAX_CENTER_ERROR_MM and max(fit['residuals_mm'].values()) <= MAX_FIT_RESIDUAL_MM
                              and vision['events'][0]['sim_time_s'] > vision['alignment_attempts'][-1]['at_s'])
        if not row['correct']:
            row['error'] = vision['error'] or '定位证据或闭环判定不符合预期'
    except (OSError, ValueError, RuntimeError) as exc:
        row['error'] = str(exc)
    write_json(basedir / 'case.json', row)
    return row


def write_report(folder, result):
    def link(case, mode):
        target = case[f'{mode}_report']
        return f'<a href="{escape(target)}">{escape(case[mode])}</a>' if target else '—'

    rows = []
    for c in result['cases']:
        error = f'{c["max_center_error_mm"]:.3f} mm' if c['max_center_error_mm'] is not None else '—'
        annotation = c['obstruction'] + ' ' + c['site'] if c['obstruction'] != 'none' else '—'
        rows.append(f'<tr><th>{c["case"]}</th><td>{c["x_mm"]:g}, {c["y_mm"]:g}, {c["yaw_deg"]:g}</td>'
                    f'<td>{annotation}</td><td>{link(c, "fixed")}</td><td>{link(c, "vision")}</td><td>{error}</td>'
                    f'<td>{"通过" if c["correct"] else "未通过"}</td></tr>')
    error_text = '—' if result['max_center_error_mm'] is None else f'{result["max_center_error_mm"]:.3f} mm'
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>视觉定位与固定示教对比</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{margin:0;background:#eef3f4;color:#193039;font:16px/1.6 "Microsoft YaHei",sans-serif}}
main{{max-width:1100px;margin:auto;padding:24px}}h1,p{{margin:.5em 0}}table{{width:100%;background:#fff;border-collapse:collapse}}
th,td{{padding:8px;border-bottom:1px solid #d8e3e5;text-align:left}}a{{color:#165669}}
</style><main><p>MuJoCo · UR5e · 同一夹具偏移成对运行</p><h1>腕部视觉定位验证</h1>
<p>同一组 {result['pairs']} 个偏移工况中，固定示教合格 {result['fixed_pass']} 轮，视觉定位合格
{result['vision_pass']} 轮。最大孔中心估计误差 {error_text}。</p>
<p>双视角遮挡 {result['occlusion_cases']} 轮，取件前中止 {result['occlusion_stopped']} 轮。
全部预期与证据核对 {result['correct']}/{result['rounds']}。<a href="study.csv">逐轮 CSV</a> ·
<a href="study.json">汇总 JSON</a></p>
<table><tr><th>工况</th><th>偏移 X / Y / 转角<br>mm / mm / °</th><th>成像扰动</th>
<th>固定示教</th><th>视觉定位</th><th>最大孔心误差</th><th>证据</th></tr>{''.join(rows)}</table></main></html>'''
    (folder / 'report.html').write_text(html, encoding='utf-8')


def run_study(profile, output, jobs):
    if not 1 <= jobs <= 8:
        raise ValueError('并发子进程数应在 1 到 8 之间')
    folder = new_run(output)
    print(f'夹具偏移对比批次 · {folder}', flush=True)
    for case in CASES:
        (folder / 'cases' / case.name).mkdir(parents=True)
    write_json(folder / 'matrix.json', [asdict(case) for case in CASES])
    if jobs == 1:
        cases = [run_case(case, profile, folder) for case in CASES]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_case, case, profile, folder) for case in CASES]
            cases = [future.result() for future in futures]
    paired = [row for row in cases if row['obstruction'] != 'occluded']
    occluded = [row for row in cases if row['obstruction'] == 'occluded']
    errors = [row['max_center_error_mm'] for row in paired if row['max_center_error_mm'] is not None]
    timings = [row['calibration_last_photo_s'] for row in paired if row['correct']]
    correct = sum(row['correct'] for row in cases)
    result = dict(source='simulation', rounds=len(cases), correct=correct, passed=correct == len(cases),
                  pairs=len(paired), fixed_pass=sum(c['fixed'] == 'pass' for c in paired),
                  vision_pass=sum(c['vision'] == 'pass' for c in paired), occlusion_cases=len(occluded),
                  occlusion_stopped=sum(c['correct'] for c in occluded), max_center_error_mm=max(errors, default=None),
                  median_calibration_last_photo_s=statistics.median(timings) if timings else None, cases=cases)
    write_json(folder / 'study.json', result)
    write_csv(folder / 'study.csv', tuple(cases[0]), cases)
    write_report(folder, result)
    return folder, result
