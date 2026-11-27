from datetime import datetime
from html import escape
from pathlib import Path
from uuid import uuid4

from .detector import ANGLES, LABELS, MISSING, PRESENT, REVIEW, SITES, Observation, save_image
from .samples import write_json


VERDICTS = {'pass': '合格', MISSING: '缺件', REVIEW: '待复检'}
ANGLE_LABELS = {'primary': '主视角', 'alternate': '补拍视角'}


class CameraError(RuntimeError):
    """A frame was unavailable after the station reached the pose."""


def timestamp():
    return datetime.now().astimezone().isoformat(timespec='milliseconds')


def new_run(root):
    folder = Path(root) / f'{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}'
    folder.mkdir(parents=True)
    return folder


def inspect_workpiece(detector, source, open_rig, output, progress=lambda _message: None, *, on_attempt=None,
                      owns_rig=True):
    if detector.source != source:
        raise ValueError(f'检测配置来源 {detector.source} 与设备模式 {source} 不符')
    folder = new_run(output)
    started = timestamp()
    records = [dict(site=site, state=REVIEW, reason='未拍摄', photo=None, attempts=[]) for site in SITES]
    rig = None
    error = None
    try:
        progress('正在连接设备' if source == 'hardware' else '正在启动 MuJoCo')
        rig = open_rig()
        if rig.source != source:
            raise ValueError('设备驱动与运行模式不符')
        for record in records:
            site = record['site']
            for angle in ANGLES:
                progress(f'{site} · {ANGLE_LABELS[angle]}')
                filename = None
                image = None
                try:
                    image = rig.capture(site, angle)
                except CameraError as exc:
                    observation = Observation(REVIEW, str(exc))
                except Exception as exc:
                    record['reason'] = f'设备流程中断：{exc}'
                    record['attempts'].append(dict(angle=angle, at=timestamp(), photo=None,
                                                   **Observation(REVIEW, record['reason']).record()))
                    raise
                else:
                    filename = f'{site}_{angle}.png'
                    save_image(folder / filename, image)
                    observation = detector.inspect(site, angle, image)
                record['attempts'].append(dict(angle=angle, at=timestamp(), photo=filename, **observation.record()))
                record.update(state=observation.state, reason=observation.reason)
                if filename:
                    record['photo'] = filename
                if on_attempt is not None:
                    on_attempt(site, angle, observation, image)
                if observation.state != REVIEW:
                    break
        progress('正在返回待机位')
        rig.finish()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if rig is not None and owns_rig:
            try:
                rig.close()
            except Exception as exc:
                close_error = f'{type(exc).__name__}: {exc}'
                error = f'{error} | {close_error}' if error else close_error

    states = [record['state'] for record in records]
    verdict = REVIEW
    if MISSING in states:
        verdict = MISSING
    elif all(state == PRESENT for state in states) and error is None:
        verdict = 'pass'
    report = dict(version=1, run_id=folder.name, source=source, started_at=started, completed_at=timestamp(),
                  verdict=verdict, error=error, positions=records)
    write_report(folder, report)
    return folder, report


def write_report(folder, report):
    write_json(folder / 'report.json', report)
    sections = []
    for record in report['positions']:
        attempts = []
        for attempt in record['attempts']:
            photo = attempt['photo']
            graphic = (f'<a href="{escape(photo)}"><img src="{escape(photo)}" alt="{record["site"]} 原图"></a>'
                       if photo else '<p>无图像</p>')
            selected = ' · 结果照片' if photo and photo == record['photo'] else ''
            quality = []
            if attempt['blur'] is not None:
                quality.append(f'清晰度 {attempt["blur"]:.1f} · 饱和 {attempt["glare"]:.1%}')
            if attempt['present_score'] is not None:
                quality.append(f'有件匹配 {attempt["present_score"]:.2f} · 缺件匹配 {attempt["missing_score"]:.2f}')
            attempts.append(f'<figure>{graphic}<figcaption>{ANGLE_LABELS[attempt["angle"]]}{selected}<br>'
                            f'{escape(attempt["reason"])}<br>{"<br>".join(quality)}</figcaption></figure>')
        sections.append(f'<section><h2>{record["site"]} · <span class="{record["state"]}">'
                        f'{LABELS[record["state"]]}</span></h2><p>{escape(record["reason"])}</p>'
                        f'<div class="photos">{"".join(attempts)}</div></section>')
    source_label = 'MuJoCo 仿真' if report['source'] == 'simulation' else 'UR5e 实机'
    error = f'<p class="review">{escape(report["error"])}</p>' if report['error'] else ''
    media = ''
    if 'trajectory' in report:
        media += '<p><a href="trajectory.csv">下载关节和相机轨迹</a></p>'
    if 'video' in report:
        media += '<video controls preload="metadata" src="inspection.mp4"></video>'
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>螺丝巡检结果</title>
<style>
*{{box-sizing:border-box}}body{{font:16px/1.55 "Microsoft YaHei",sans-serif;color:#1b292e;background:#f0f3f2;margin:0}}
main{{max-width:1040px;margin:auto;padding:30px 20px}}header,section{{background:white;border:1px solid #d8e1df;
padding:20px 24px;margin-bottom:14px;border-radius:12px}}h1,h2,p,figure{{margin:0}}header p,section p{{margin-top:5px}}
.pass,.present{{color:#18614c}}.review{{color:#855615}}.missing{{color:#9d3528}}.photos{{display:flex;flex-wrap:wrap;
gap:18px;margin-top:10px}}figure{{width:min(100%,360px)}}img{{display:block;width:100%;border-radius:6px}}
figcaption{{padding-top:5px;color:#455f65;font-size:14px}}a{{color:#1c5f70}}video{{display:block;width:100%;margin-top:12px}}
</style></head><body><main><header><p>{source_label} · {escape(report["completed_at"])}</p>
<h1>整件判定 <span class="{report["verdict"]}">{VERDICTS[report["verdict"]]}</span></h1>{error}
<p><a href="report.json">查看机器可读记录</a></p>{media}</header>{"".join(sections)}</main></body></html>'''
    (folder / 'report.html').write_text(html, encoding='utf-8')
