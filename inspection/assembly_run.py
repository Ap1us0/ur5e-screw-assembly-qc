import os
import threading
import time
from contextlib import ExitStack, closing
from dataclasses import asdict
from html import escape

import cv2
import mujoco
import numpy as np

from .assembly import Assembly, AssemblyStopped, ENGAGE_XY_M, PITCH_MM, TORQUE_TARGET_NM
from .demo import FPS, VIDEO_SIZE, VIEW_SIZE, overlay_text, overview_camera
from .detector import SITES, save_image
from .samples import write_csv, write_json
from .vision_alignment import FixtureShift, marked_photo
from .workflow import inspect_workpiece, new_run


STAGES = {
    'calibration': 'Locate empty hole',
    'ready': 'Ready', 'approach_feed': 'Approach feeder', 'index': 'Index bit', 'pick': 'Pick screw',
    'lift': 'Verify pickup', 'transfer': 'Transfer over fixture', 'align': 'Align to hole',
    'insert': 'Enter thread', 'tighten': 'Drive and torque-check',
    'release': 'Release bit', 'retract': 'Retract and verify', 'inspection': 'Inspect all positions',
}
RESULTS = {'pass': 'ASSEMBLY PASS', 'review': 'REVIEW REQUIRED', 'aborted': 'ASSEMBLY ABORTED', 'error': 'SIMULATION ERROR'}
RESULT_COLORS = {'pass': (99, 224, 165), 'review': (242, 198, 118), 'aborted': (247, 125, 113), 'error': (247, 125, 113)}
INSET_SIZE = (368, 276)


class AssemblyView:
    def __init__(self, rig, viewer, writer=None):
        self.rig = rig
        self.viewer = viewer
        self.writer = writer
        self.rows = []
        self.result = None
        self.overview = overview_camera(mujoco.MjvCamera())
        self.overview.distance = 1.58
        self.detail = mujoco.MjvCamera()
        self.detail.distance = 0.185
        self.detail.azimuth = 154
        self.detail.elevation = -17
        self.next_frame = 0.0
        self.started = time.perf_counter()
        self.overview_renderer = None
        self.detail_renderer = None
        self.photo = None
        self.photo_label = ''
        self.photos = {}
        self.registration_photo = None
        if viewer:
            with viewer.lock():
                overview_camera(viewer.cam)
                viewer.cam.distance = self.overview.distance
                viewer.opt.geomgroup[:] = rig.options.geomgroup
                viewer.opt.sitegroup[:] = rig.options.sitegroup

    def sample(self):
        now = self.rig.data.time
        if now < self.next_frame:
            return
        self.next_frame += 1 / FPS
        telemetry = self.rig.snapshot()
        self.rows.append(telemetry)
        if self.viewer and self.viewer.is_running():
            time.sleep(max(0.0, self.started + now - time.perf_counter()))
        self.draw(telemetry)

    def capture(self, site, angle, observation, image):
        if image is not None:
            self.photo = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.photo_label = f'{site} / {angle.title()} / {observation.state.title()}'
            self.photos[site] = self.photo, observation.state

    def preview_registration(self, attempt, marked):
        label = f'{attempt["site"]} / {attempt["angle"].title()} / '
        label += 'Hole located' if attempt['observation'] else 'No reliable hole'
        self.registration_photo = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB), label
        self.draw(hold_frames=FPS // 3)

    def closeup(self):
        if self.rig.stage == 'calibration' and self.registration_photo is not None:
            image, label = self.registration_photo
            return cv2.resize(image, INSET_SIZE, interpolation=cv2.INTER_AREA), label
        if self.result and len(self.photos) == 3:
            width, height = INSET_SIZE
            montage = np.full((height, width, 3), (29, 44, 56), dtype=np.uint8)
            for i, (site, (photo, state)) in enumerate(self.photos.items()):
                h, w = photo.shape[:2]
                area = photo[h * 3 // 8:h * 5 // 8, w * 3 // 8:w * 5 // 8]
                x = i * 120 + 12
                montage[76:184, x:x + 104] = cv2.resize(area, (104, 108), interpolation=cv2.INTER_AREA)
                overlay_text(montage, site, (x, 64), scale=0.65, weight=2)
                overlay_text(montage, state.title(), (x, 211), scale=0.55)
            return montage, 'Final wrist inspection'
        if self.photo is not None:
            return cv2.resize(self.photo, INSET_SIZE, interpolation=cv2.INTER_AREA), self.photo_label
        if self.detail_renderer is None:
            self.detail_renderer = mujoco.Renderer(self.rig.model, width=INSET_SIZE[0], height=INSET_SIZE[1])
        at_feeder = self.rig.stage in ('ready', 'approach_feed', 'index', 'pick', 'lift')
        station = self.rig.station
        center = station.screw_nominal if at_feeder else station.entry_actual
        self.detail.lookat[:] = center + (0, 0, 0.015)
        self.detail_renderer.update_scene(self.rig.data, camera=self.detail, scene_option=self.rig.options)
        label = f'Feeder {station.name}' if at_feeder else f'{station.name} close-up'
        return self.detail_renderer.render().copy(), label

    def draw(self, metrics=None, hold_frames=1):
        viewer = self.viewer
        visible = viewer is not None and viewer.is_running()
        if not visible and self.writer is None:
            return
        metrics = self.rig.snapshot() if metrics is None else metrics
        stage_title = STAGES[self.rig.stage]
        active = self.rig.stage not in ('ready', 'inspection')
        title = f'{metrics["site"]} / {stage_title}' if active else stage_title
        if self.result:
            title = RESULTS[self.result]
        closeup, image_title = self.closeup()
        driver_state = 'Running' if metrics['spindle_velocity_command_rad_s'] else 'Stopped'
        depth = max(0.0, metrics['insertion_depth_mm'])
        progress = '  /  '.join(f'{site} {entry["status"]}' for site, entry in self.rig.results.items())
        if self.rig.stage == 'calibration':
            located = sum(record['observation'] is not None for record in self.rig.alignment_attempts)
            progress = f'Wrist camera registration   {located} / 3 holes'
        cutoff = self.rig.results[metrics['site']]['torque_cutoff']
        display_torque = cutoff['spindle_torque_nm'] if self.result and cutoff else metrics['spindle_torque_nm']
        if visible:
            with viewer.lock():
                mujoco.mj_copyData(viewer.d, self.rig.model, self.rig.data)
            instructions = 'Enter: repeat  |  O: report' if self.result else 'Enter: start' if self.rig.stage == 'ready' else ''
            viewer.set_texts([
                (None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 'UR5e assembly\nStatus\nDepth\nSpindle torque\nTurns\nDriver',
                 f'{title}\n{progress}\n{depth:.2f} mm\n{display_torque:.3f} Nm\n'
                 f'{metrics["thread_turns"]:.2f}\n{driver_state}'),
                (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, instructions, ''),
            ])
            viewport = viewer.viewport
            if viewport and viewport.width >= 490 and viewport.height >= 320:
                width = min(viewport.width // 3, INSET_SIZE[0])
                height = width * 3 // 4
                rect = mujoco.MjrRect(viewport.width - width - 18, 24, width, height)
                inset = cv2.resize(closeup, (width, height), interpolation=cv2.INTER_AREA)
                cv2.rectangle(inset, (0, 0), (width, 22), (20, 31, 42), -1)
                overlay_text(inset, image_title, (7, 16), scale=0.46)
                viewer.set_images((rect, inset))
            else:
                viewer.clear_images()
            viewer.sync(state_only=True)
        if self.writer:
            frame = self.video_frame(metrics, title, progress, closeup, image_title, depth, driver_state, display_torque)
            for _ in range(hold_frames):
                self.writer.send(frame)

    def video_frame(self, metrics, title, progress, closeup, image_title, depth, driver_state, display_torque):
        width, height = VIEW_SIZE
        if self.overview_renderer is None:
            self.overview_renderer = mujoco.Renderer(self.rig.model, width=width, height=height)
        self.overview_renderer.update_scene(self.rig.data, camera=self.overview, scene_option=self.rig.options)
        frame = np.full((height, VIDEO_SIZE[0], 3), (18, 29, 40), dtype=np.uint8)
        frame[:, :width] = self.overview_renderer.render()
        x = width + 24
        overlay_text(frame, 'UR5e  /  SCREW ASSEMBLY', (x, 49), scale=0.78, weight=2)
        overlay_text(frame, title, (x, 94), scale=0.78, weight=2,
                     ink=RESULT_COLORS.get(self.result, (134, 210, 220)))
        overlay_text(frame, progress, (x, 124), scale=0.52)
        values = (
            ('Insertion depth', f'{depth:.2f} mm'),
            ('Thread turns', f'{metrics["thread_turns"]:.2f}'),
            ('Spindle torque', f'{display_torque:.3f} Nm'),
            ('Seating force', f'{metrics["seat_force_n"]:.1f} N'),
            ('Driver', driver_state),
        )
        for i, (label, value) in enumerate(values):
            y = 159 + i * 29
            overlay_text(frame, label, (x, y), scale=0.58)
            overlay_text(frame, value, (x + 205, y), scale=0.61, ink=(172, 221, 223))
        close_width, close_height = INSET_SIZE
        frame[320:320 + close_height, x:x + close_width] = closeup
        overlay_text(frame, image_title, (x, 621), scale=0.56)
        return frame

    def close(self):
        for renderer in (self.detail_renderer, self.overview_renderer):
            if renderer is not None:
                renderer.close()


def plot_trace(folder, rows):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    sites = {site: [row for row in rows if row['site'] == site and row['stage'] in ('align', 'insert', 'tighten')]
             for site in SITES}
    sites = {site: samples for site, samples in sites.items() if samples}
    if not sites:
        return False
    fig = Figure(figsize=(max(5.4, len(sites) * 4.2), 5.8), layout='constrained', facecolor='#f4f7f8')
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(3, len(sites))
    for col, (site, samples) in enumerate(sites.items()):
        tight = [row for row in samples if row['stage'] == 'tighten']
        active = tight or samples
        times = [row['sim_time_s'] - active[0]['sim_time_s'] for row in active]
        if not tight:
            radial = fig.add_subplot(grid[:, col])
            radial.plot(times, [row['radial_error_mm'] for row in active], color='#25666e')
            radial.axhline(1000 * ENGAGE_XY_M, linestyle='--', color='#b05442', label='Entry limit')
            radial.set(title=site, xlabel='Alignment time (s)', ylabel='Lateral offset (mm)')
            radial.legend(loc='upper right')
            continue
        depth = fig.add_subplot(grid[0, col])
        torque = fig.add_subplot(grid[1, col], sharex=depth)
        force = fig.add_subplot(grid[2, col], sharex=depth)
        depth.plot(times, [row['insertion_depth_mm'] for row in active], label='Measured', color='#25666e')
        depth.plot(times, [PITCH_MM * row['thread_turns'] for row in active], '--', label='Pitch × turns', color='#8295a7')
        depth.set(title=site, ylabel='Depth (mm)')
        depth.legend(loc='upper left', fontsize=8)
        torque.plot(times, [row['spindle_torque_nm'] for row in active], color='#b15e21')
        torque.axhline(TORQUE_TARGET_NM, linestyle='--', color='#6e8295', label='Torque threshold')
        torque.set_ylabel('Torque (N·m)')
        torque.legend(loc='upper left', fontsize=8)
        force.plot(times, [row['seat_force_n'] for row in active], label='Seat', color='#25666e')
        force.plot(times, [row['obstruction_force_n'] for row in active], label='Obstruction', color='#b05442')
        force.set(xlabel='Spindle time (s)', ylabel='Normal force (N)')
        force.legend(loc='upper left', fontsize=8)
        depth.tick_params(labelbottom=False)
        torque.tick_params(labelbottom=False)
    for axes in fig.axes:
        axes.grid(alpha=0.2)
    fig.savefig(folder / 'assembly_trace.png', dpi=130)
    return True


def write_assembly_report(folder, report):
    write_json(folder / 'assembly.json', report)
    title = {'pass': '装配合格', 'review': '待复检', 'aborted': '装配中止', 'error': '仿真异常'}[report['status']]
    links = ['<a href="assembly.json">装配记录</a>', '<a href="assembly_trace.csv">逐步数据</a>']
    visual = ''
    if report['inspection_report']:
        links.append(f'<a href="{escape(report["inspection_report"])}">三点照片与判定</a>')
    if report['video']:
        visual += '<video controls src="assembly.mp4"></video>'
    if report['plot']:
        visual += '<img src="assembly_trace.png" alt="深度、扭矩与接触力曲线">'
    labels = {'installed': '已装配', 'failed': '中止', 'working': '进行中', 'pending': '未装配'}
    rows = []
    for site, entry in report['positions'].items():
        retained, cutoff = entry['after_retract'], entry['torque_cutoff']
        depth = f'{retained["insertion_depth_mm"]:.2f} mm' if retained else '—'
        turns = f'{retained["thread_turns"]:.2f}' if retained else '—'
        torque = f'{cutoff["spindle_torque_nm"]:.3f} N·m' if cutoff else '—'
        rows.append(f'<tr><th>{site}</th><td>{labels[entry["status"]]}</td><td>{depth}</td>'
                    f'<td>{turns}</td><td>{torque}</td></tr>')
    error = f'<p>{escape(report["error"])}</p>' if report['error'] else ''
    motion_label = {'baseline': '基准运动', 'optimized': '优化运动'}[report['motion']]
    duration = f'{motion_label} · 整轮仿真 {report["cycle_time_s"]:.2f} s'
    alignment = ''
    if report['alignment'] == 'vision':
        write_registration_report(folder, report)
        links.append('<a href="registration.html">三孔定位原图与估计结果</a>')
        estimate = report['registration']
        if estimate:
            alignment = (f'<p>腕部相机估计夹具偏移 ({estimate["x_mm"]:.2f}, {estimate["y_mm"]:.2f}) mm，'
                         f'转角 {estimate["yaw_deg"]:.2f}°。</p>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>UR5e 装配结果</title>
<style>body{{background:#f3f6f6;color:#193039;font:16px/1.6 "Microsoft YaHei",sans-serif;margin:0}}
main{{max-width:1080px;margin:auto;padding:28px 18px}}h1{{margin:.15em 0}}p{{margin:10px 0}}
table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:7px;text-align:left;border-bottom:1px solid #d8e3e5}}
video,img{{display:block;width:100%;border-radius:8px;margin-top:18px}}a{{color:#165669}}</style>
<main><p>MuJoCo · UR5e</p><h1>{title}</h1><p>{duration}</p>{alignment}{error}<table><tr><th>位置</th><th>结果</th><th>撤刀后深度</th>
<th>圈数</th><th>停转扭矩</th></tr>{''.join(rows)}</table><p>{' · '.join(links)}</p>{visual}</main></html>'''
    (folder / 'report.html').write_text(page, encoding='utf-8')


def write_registration_report(folder, report):
    estimate = report['registration']
    info = ''
    if estimate:
        residual = max(estimate['residuals_mm'].values())
        error = max(estimate['actual_center_errors_mm'].values())
        info = (f'<p>估计偏移 ({estimate["x_mm"]:.3f}, {estimate["y_mm"]:.3f}) mm，转角 {estimate["yaw_deg"]:.3f}°。</p>'
                f'<p>最大拟合残差 {residual:.3f} mm。与仿真真值比较的最大孔中心误差 {error:.3f} mm。</p>')
    elif report['error']:
        info = f'<p>{escape(report["error"])}</p>'
    views = ''.join(f'<figure><img src="{escape(a["marked_photo"])}" alt="定位叠加图">'
                    f'<figcaption>{a["site"]} · {a["angle"]} · {escape(a["reason"])} · '
                    f'<a href="{escape(a["photo"])}">原图</a></figcaption></figure>'
                    for a in report['alignment_attempts'])
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>腕部相机三孔定位</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{background:#eef3f4;color:#193039;font:16px/1.6 "Microsoft YaHei",sans-serif;margin:0}}
main{{max-width:1000px;margin:auto;padding:24px}}h1,p{{margin:.4em 0}}img{{width:100%;border-radius:6px}}
.views{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:14px;margin-top:18px}}
figure{{margin:0;padding:10px;background:#fff;border-radius:8px}}a{{color:#165669}}
</style><main><p>MuJoCo · UR5e · 入牙前定位</p><h1>腕部相机三孔定位</h1>{info}
<p>黄色十字是示教位置，绿色圆与十字是画面中估计的孔中心。</p>
<p><a href="report.html">返回装配报告</a> · <a href="assembly.json">定位数据 JSON</a></p>
<div class="views">{views}</div></main></html>'''
    (folder / 'registration.html').write_text(html, encoding='utf-8')


def run_round(rig, detector, output, viewer, record):
    folder = new_run(output)
    rig.reset_cycle()
    result = dict(version=2, source='simulation', scenario=rig.scenario,
                  fault_site=rig.mechanical_fault_site if rig.scenario != 'none' else None,
                  seed=rig.seed, image_fault=rig.image_fault, image_site=rig.image_site, conditions=rig.conditions,
                  motion=rig.motion, alignment=rig.alignment, fixture_shift_truth=asdict(rig.fixture_shift),
                  alignment_fault=rig.alignment_fault, alignment_fault_site=rig.alignment_fault_site,
                  registration=None,
                  order=list(rig.sites), status='error', error=None, stop_code=None, inspection_started_at_s=None,
                  inspection_verdict=None, inspection_report=None)
    with ExitStack() as resources:
        # Keep the wrist context alive through the final composite with the other renderers.
        resources.callback(rig.close)
        writer = None
        if record:
            from imageio_ffmpeg import write_frames

            writer = resources.enter_context(closing(write_frames(
                str(folder / 'assembly.mp4'), VIDEO_SIZE, fps=FPS, codec='libx264',
                pix_fmt_in='rgb24', pix_fmt_out='yuv420p', ffmpeg_log_level='error',
                output_params=['-movflags', '+faststart'],
            )))
            writer.send(None)
        view = AssemblyView(rig, viewer, writer)
        resources.callback(view.close)
        rig.on_frame = view.sample
        try:
            view.draw(hold_frames=FPS // 2)
            if rig.alignment == 'vision':
                def store_attempt(attempt, image, observation):
                    stem = f'calibration/{attempt["site"]}_{attempt["angle"]}'
                    (folder / 'calibration').mkdir(exist_ok=True)
                    marked = marked_photo(image, observation, attempt['nominal_uv'])
                    attempt.update(photo=f'{stem}.png', marked_photo=f'{stem}_marked.png')
                    save_image(folder / attempt['photo'], image)
                    save_image(folder / attempt['marked_photo'], marked)
                    view.preview_registration(attempt, marked)

                fit = rig.register_fixture(store_attempt)
                fit['actual_center_errors_mm'] = {
                    site: float(1000 * np.linalg.norm(center - rig.stations[site].entry_actual[:2]))
                    for site, center in fit['centers_m'].items()
                }
                result['registration'] = fit
            rig.assemble()
            rig.stage = 'inspection'
            result['inspection_started_at_s'] = float(rig.data.time)
            inspection_folder, report = inspect_workpiece(detector, rig.source, lambda: rig, folder / 'inspection',
                                                           on_attempt=view.capture, owns_rig=False)
            result['inspection_report'] = (inspection_folder / 'report.html').relative_to(folder).as_posix()
            result['inspection_verdict'] = report['verdict']
            result['error'] = report['error']
            result['status'] = 'pass' if report['verdict'] == 'pass' else 'review'
            rig.verify_all_retained()
        except AssemblyStopped as exc:
            message = str(exc) if exc.code == 'registration_fit' else f'{rig.station.name} · {exc}'
            result.update(status='aborted', stop_code=exc.code, error=message)
            rig.event('aborted')
            try:
                rig.brake_spindle()
                rig.event('spindle_stopped_after_abort')
            except RuntimeError as brake_error:
                result.update(status='error', error=f'{exc} / {brake_error}')
        except RuntimeError as exc:
            result.update(status='error', error=f'{rig.station.name} · {exc}')
        finally:
            rig.on_frame = None
            rig.data.ctrl[rig.motor] = 0.0
        view.result = result['status']
        view.draw(hold_frames=2 * FPS)
    result.update(positions=rig.results, events=rig.events, moves=rig.moves, cycle_time_s=float(rig.data.time),
                  alignment_attempts=rig.alignment_attempts,
                  final={site: rig.snapshot(station) for site, station in rig.stations.items()})
    if not view.rows:
        view.rows.append(rig.snapshot())
    write_csv(folder / 'assembly_trace.csv', tuple(view.rows[0]), view.rows)
    result['plot'] = plot_trace(folder, view.rows)
    result['video'] = record and (folder / 'assembly.mp4').stat().st_size > 0
    write_assembly_report(folder, result)
    print(f'{RESULTS[result["status"]]} · {folder / "report.html"}', flush=True)
    if result['error']:
        print(result['error'], flush=True)
    return folder, result


def run_assembly(detector, output, fault, *, sites=SITES, fault_site='S2', seed=0, image_fault='none',
                 image_site='S2', motion='optimized', alignment='fixed', fixture_shift=FixtureShift(),
                 alignment_fault='none', alignment_fault_site='S2', viewer=False, video=False):
    if detector.source != 'simulation':
        raise ValueError('装配演示需要仿真检测配置')
    start, open_report = threading.Event(), threading.Event()

    def key_pressed(key):
        if key == 257:  # GLFW_KEY_ENTER
            start.set()
        elif key == ord('O'):
            open_report.set()

    with ExitStack() as resources:
        rig = Assembly(fault, sites, fault_site, seed=seed, image_fault=image_fault, image_site=image_site,
                       motion=motion, alignment=alignment, fixture_shift=fixture_shift,
                       alignment_fault=alignment_fault, alignment_fault_site=alignment_fault_site)
        resources.callback(rig.close)
        if not viewer:
            return run_round(rig, detector, output, None, video)

        from mujoco.viewer import launch_passive

        data = mujoco.MjData(rig.model)
        result = None
        window = resources.enter_context(launch_passive(rig.model, data, key_callback=key_pressed,
                                                        show_left_ui=False, show_right_ui=False))
        with closing(AssemblyView(rig, window)) as initial:
            initial.draw()
        while window.is_running():
            if open_report.is_set():
                open_report.clear()
                if result:
                    os.startfile(result[0] / 'report.html')
            if start.wait(0.05) and window.is_running():
                start.clear()
                result = run_round(rig, detector, output, window, video)
                start.clear()
        return result
