import json
import math
from html import escape
from pathlib import Path

import numpy as np

from .samples import write_csv, write_json
from .workflow import new_run


CASE_FIELDS = ('seed', 'outcome', 'fault', 'fault_site', 'image_fault', 'image_site')
METRICS = {'cycle_time_s': '整轮', 'assembly_time_s': '装配至启动质检',
           'inspection_time_s': '统一质检及回位', 'arm_settling_s': '各动作关节稳定等待合计'}


def read_batch(path, profile):
    batch = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    cases = batch['cases']
    labels = {case['label'] for case in cases}
    if batch['source'] != 'simulation' or batch['motion'] != profile or len(cases) != 20 or len(labels) != 20:
        raise ValueError(f'无效的 {profile} 验收批次 {path}')
    return batch


def durations(case):
    total, assembly, settle = (case.get(field) for field in ('cycle_time_s', 'assembly_time_s', 'arm_settling_s'))
    if (not all(isinstance(t, (float, int)) and math.isfinite(t) and t >= 0 for t in (total, assembly, settle))
            or not (0 < assembly < total and settle < total)):
        return dict.fromkeys(METRICS)
    return dict(cycle_time_s=total, assembly_time_s=assembly, inspection_time_s=total - assembly, arm_settling_s=settle)


def aggregate(rows):
    summaries = {}
    for metric in METRICS:
        before = np.array([row[f'baseline_{metric}'] for row in rows])
        after = np.array([row[f'optimized_{metric}'] for row in rows])
        old, new = float(np.median(before)), float(np.median(after))
        summaries[metric] = dict(baseline_median=old, optimized_median=new,
                                 baseline_p95=float(np.percentile(before, 95)),
                                 optimized_p95=float(np.percentile(after, 95)),
                                 median_change_percent=100 * (1 - new / old) if old else None)
    return summaries


def compare(baseline_file, optimized_file, output):
    before = read_batch(baseline_file, 'baseline')
    after = read_batch(optimized_file, 'optimized')
    right = {case['label']: case for case in after['cases']}
    rows = []
    for original in before['cases']:
        label = original['label']
        if label not in right or any(original[key] != right[label][key] for key in CASE_FIELDS):
            raise ValueError(f'无法进行成对比较，工况不同 {label}')
        optimized = right[label]
        row = {key: original[key] for key in ('label', *CASE_FIELDS)}
        row.update(baseline_result=original['observed'], optimized_result=optimized['observed'],
                   baseline_correct=original['correct'], optimized_correct=optimized['correct'])
        for prefix, case in (('baseline', original), ('optimized', optimized)):
            row.update({f'{prefix}_{metric}': value for metric, value in durations(case).items()})
        rows.append(row)

    production = [row for row in rows if row['outcome'] == 'pass' and row['baseline_correct'] and row['optimized_correct']
                  and row['baseline_cycle_time_s'] is not None and row['optimized_cycle_time_s'] is not None]
    metrics = aggregate(production) if production else {}
    verified = all(row['baseline_correct'] and row['optimized_correct'] for row in rows)
    result = dict(pairs=len(rows), production_pairs=len(production),
                  baseline_correct=before['correct'], optimized_correct=after['correct'],
                  baseline_false_pass=before['false_pass'], optimized_false_pass=after['false_pass'],
                  passed=verified and len(production) == sum(row['outcome'] == 'pass' for row in rows)
                  and not before['false_pass'] and not after['false_pass'] and bool(metrics)
                  and metrics['cycle_time_s']['median_change_percent'] > 0,
                  baseline_batch=Path(baseline_file).parent.name, optimized_batch=Path(optimized_file).parent.name,
                  metrics=metrics, cases=rows)
    folder = new_run(output)
    write_json(folder / 'comparison.json', result)
    write_csv(folder / 'paired-cycles.csv', tuple(rows[0]), rows)
    if production:
        plot_cycles(folder, production)
    write_report(folder, result)
    return folder, result


def plot_cycles(folder, rows):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8.9, 5.7), layout='constrained', facecolor='#f3f7f7')
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    indices = np.arange(len(rows))
    baseline = np.array([row['baseline_cycle_time_s'] for row in rows])
    optimized = np.array([row['optimized_cycle_time_s'] for row in rows])
    axes.hlines(indices, optimized, baseline, color='#b5c4c8', linewidth=1.6)
    axes.scatter(baseline, indices, color='#a97041', label='Baseline cubic', zorder=3, s=40)
    axes.scatter(optimized, indices, color='#16777a', label='Optimized quintic + feedforward', zorder=3, s=40)
    axes.set(yticks=indices, yticklabels=[row['label'] for row in rows], xlabel='Simulated cycle time (s)',
             title='Paired production cases | Three-screw assembly and final inspection')
    axes.invert_yaxis()
    axes.set_xlim(left=0)
    axes.grid(axis='x', alpha=0.21)
    axes.legend(loc='lower left')
    figure.savefig(folder / 'cycle-comparison.png', dpi=160)


def write_report(folder, result):
    records = []
    for metric, sample in result['metrics'].items():
        reduction = f'{sample["median_change_percent"]:.1f}%' if sample['median_change_percent'] is not None else '—'
        records.append(f'<tr><th>{METRICS[metric]}</th><td>{sample["baseline_median"]:.2f} s</td>'
                       f'<td>{sample["optimized_median"]:.2f} s</td><td>{reduction}</td>'
                       f'<td>{sample["optimized_p95"]:.2f} s</td></tr>')
    chart = ('<img src="cycle-comparison.png" alt="基准和优化版逐轮仿真节拍对比">' if records else '')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>UR5e 节拍优化对比</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{margin:0;background:#edf3f4;color:#193039;font:16px/1.6 "Microsoft YaHei",sans-serif}}
main{{max-width:1010px;margin:auto;padding:28px 20px}}table{{background:white;border-collapse:collapse;width:100%}}
th,td{{text-align:left;padding:8px;border-bottom:1px solid #d8e3e5}}img{{display:block;width:100%;margin-top:20px}}a{{color:#166670}}
</style><main><p>MuJoCo · UR5e · 相同工况成对测量</p><h1>三螺丝装配与质检节拍对比</h1>
<p>基准 {result['baseline_correct']}/{result['pairs']} 判对，误放行 {result['baseline_false_pass']}。优化版
{result['optimized_correct']}/{result['pairs']} 判对，误放行 {result['optimized_false_pass']}。</p>
<p>节拍统计基于 {result['production_pairs']} 组两种运动方式都通过的合格工况。整轮包含三颗旋紧、撤刀、
统一质检和回待机位。下表使用 MuJoCo 仿真时钟。</p>
<table><tr><th>指标</th><th>基准中位数</th><th>优化中位数</th><th>变化幅度</th><th>优化 P95</th></tr>
{''.join(records)}</table>{chart}<p><a href="paired-cycles.csv">逐轮配对数据</a> ·
<a href="comparison.json">比较结果 JSON</a></p><p>基准批次 {escape(result['baseline_batch'])}<br>
优化批次 {escape(result['optimized_batch'])}</p></main></html>'''
    (folder / 'report.html').write_text(page, encoding='utf-8')
