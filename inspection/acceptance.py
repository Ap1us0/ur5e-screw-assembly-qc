import csv
import json
from pathlib import Path

from .detector import MISSING, PRESENT, REVIEW, SITES
from .samples import write_csv, write_json
from .workflow import new_run, inspect_workpiece


def parse_missing(value):
    sites = [site.strip().upper() for site in value.replace(',', '|').split('|') if site.strip()]
    if len(set(sites)) != len(sites) or not set(sites) <= set(SITES):
        raise ValueError(f'缺件位置必须是互不重复的 S1、S2、S3，当前值 {value!r}')
    return tuple(site for site in SITES if site in sites)


def evaluate(manifest, output, source):
    with manifest.open(newline='', encoding='utf-8-sig') as stream:
        cases = list(csv.DictReader(stream))
    if len(cases) != 20:
        raise ValueError(f'验收清单必须恰好 20 轮，当前为 {len(cases)} 轮')
    rows = []
    seen_runs = set()
    for case in cases:
        report_path = (manifest.parent / case['report']).resolve()
        report = json.loads(report_path.read_text(encoding='utf-8-sig'))
        if report['version'] != 1 or report['source'] != source or report['run_id'] in seen_runs:
            raise ValueError(f'验收报告版本、来源错误或重复使用同一轮 {report_path}')
        seen_runs.add(report['run_id'])
        missing = parse_missing(case['missing'])
        expected = MISSING if missing else 'pass'
        observed = report['verdict']
        positions = report['positions']
        states = {entry['site']: entry['state'] for entry in positions}
        if (len(positions) != len(SITES) or set(states) != set(SITES)
                or not set(states.values()) <= {PRESENT, MISSING, REVIEW}):
            raise ValueError(f'验收报告位置异常 {report_path}')
        consistent = (MISSING if MISSING in states.values() else 'pass'
                      if all(value == PRESENT for value in states.values()) and not report['error'] else REVIEW)
        if observed != consistent:
            raise ValueError(f'整件与位置状态不一致 {report_path}')
        positions_correct = 0
        for entry in positions:
            photo = entry['photo']
            if entry['state'] != REVIEW and (not photo or not (report_path.parent / photo).is_file()):
                raise ValueError(f'已判定位置缺少照片 {report_path}/{entry["site"]}')
            truth = MISSING if entry['site'] in missing else PRESENT
            positions_correct += entry['state'] == truth
        rows.append(dict(run_id=report['run_id'], missing=list(missing), expected=expected, observed=observed,
                         correct=expected == observed, positions_correct=positions_correct, report=case['report'],
                         false_pass=bool(missing) and observed == 'pass'))
    missing_rounds = sum(bool(row['missing']) for row in rows)
    if not 0 < missing_rounds < 20:
        raise ValueError('验收清单须同时包含完好和缺件工件')
    correct = sum(row['correct'] for row in rows)
    false_pass = sum(row['false_pass'] for row in rows)
    result = dict(source=source, rounds=20, complete_rounds=20 - missing_rounds, missing_rounds=missing_rounds,
                  correct=correct, positions_correct=sum(row['positions_correct'] for row in rows),
                  false_pass=false_pass, passed=correct >= 18 and false_pass == 0, cases=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def simulate_acceptance(detector, output):
    from .simulation import Simulation

    folder = new_run(output)
    cases = []
    for index, missing in enumerate([()] * 10 + [('S1',), ('S2',), ('S3',), ('S1', 'S2'), ('S3',),
                                                 ('S2',), ('S2', 'S3'), ('S1',), ('S3',), SITES], start=1):
        fault = ('none', 'glare', 'blur', 'none')[index % 4]
        site = SITES[(index - 1) % len(SITES)]

        def open_rig():
            rig = Simulation()
            rig.reset(missing, fault, site, seed=index + 100)
            return rig

        run, report = inspect_workpiece(detector, 'simulation', open_rig, folder)
        print(f'{index:02}/20 {report["verdict"]} · {run.name}', flush=True)
        cases.append(dict(missing='|'.join(missing), report=f'{run.name}/report.json'))
    manifest = folder / 'trials.csv'
    write_csv(manifest, ('missing', 'report'), cases)
    return folder, evaluate(manifest, folder / 'acceptance.json', 'simulation')
