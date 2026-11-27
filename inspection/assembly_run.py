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
from .force_insertion import FORCE_ENTRY_TOLERANCE_M
from .samples import write_csv, write_json
from .vision_alignment import FixtureShift, marked_photo
from .workflow import inspect_workpiece, new_run


STAGES = {
    'calibration': 'Locate empty hole',
    'ready': 'Ready', 'approach_feed': 'Approach feeder', 'index': 'Index bit', 'pick': 'Pick screw',
    'lift': 'Verify pickup', 'transfer': 'Transfer over fixture', 'align': 'Align to hole',
    'insert': 'Enter thread', 'tighten': 'Drive and torque-check',
    'probe': 'Force-guided alignment', 'jam_brake': 'Jam: brake spindle',
    'unwind': 'Reverse out of thread', 'retry_retract': 'Retract with screw',
    'angled_entry': 'Angled thread entry', 'retry_realign': 'Correct driver tilt',
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
        speed = metrics['spindle_velocity_command_rad_s']
        driver_state = ('Reverse' if speed < 0 else 'Forward' if speed > 0 else
                        'Braking' if abs(metrics['spindle_velocity_rad_s']) > 0.08 else 'Stopped')
        depth = max(0.0, metrics['insertion_depth_mm'])
        progress = '  /  '.join(f'{site} {entry["status"]}' for site, entry in self.rig.results.items())
        if self.rig.stage == 'calibration':
            located = sum(record['observation'] is not None for record in self.rig.alignment_attempts)
            progress = f'Wrist camera registration   {located} / 3 holes'
        elif active and metrics['attempt'] and self.rig.stage in (
                'probe', 'angled_entry', 'tighten', 'jam_brake', 'unwind', 'retry_retract', 'retry_realign'):
            progress = f'Attempt {metrics["attempt"]}   |   {progress}'
        cutoff = self.rig.results[metrics['site']]['torque_cutoff']
        display_torque = cutoff['spindle_torque_nm'] if self.result and cutoff else metrics['spindle_torque_nm']
        if visible:
            with viewer.lock():
                mujoco.mj_copyData(viewer.d, self.rig.model, self.rig.data)
            instructions = 'Enter: repeat  |  O: report' if self.result else 'Enter: start' if self.rig.stage == 'ready' else ''
            viewer.set_texts([
                (None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 'UR5e assembly\nStatus\nDepth\nSpindle torque\nTurns\nDriver\nWrist force XY / Z\nScrew tilt',
                 f'{title}\n{progress}\n{depth:.2f} mm\n{display_torque:.3f} Nm\n'
                 f'{metrics["thread_turns"]:.2f}\n{driver_state}\n'
                 f'{metrics["tool_lateral_n"]:.2f} / {metrics["tool_fz_n"]:+.2f} N\n{metrics["tilt_deg"]:.2f} deg'),
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
            ('Wrist force XY / Z', f'{metrics["tool_lateral_n"]:.2f} / {metrics["tool_fz_n"]:+.2f} N'),
            ('Driver', driver_state),
            ('Screw tilt', f'{metrics["tilt_deg"]:.2f} deg'),
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


def plot_force_trace(folder, rows, events):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    stages = ('probe', 'tighten', 'jam_brake', 'unwind', 'retry_retract')
    sites = {site: [row for row in rows if row['site'] == site and row['stage'] in stages]
             for site in SITES}
    sites = {site: samples for site, samples in sites.items() if samples}
    if not sites:
        return False
    colors = dict(probe='#e1eef4', tighten='#e5f2eb', jam_brake='#f8ddd6', unwind='#f9ead5', retry_retract='#e9e5f2')
    labels = dict(probe='Probe', tighten='Drive', jam_brake='Brake', unwind='Reverse', retry_retract='Retract')
    series = (
        ('XY error (mm)', (('radial_error_mm', 'Screw axis', '#1b6d74'),)),
        ('Wrist load (N)', (('tool_lateral_n', 'XY magnitude', '#1b6d74'), ('tool_fz_n', 'Signed Z', '#b05328'))),
        ('Spindle (rad/s)', (('spindle_velocity_rad_s', 'Measured', '#1b6d74'),
                             ('spindle_velocity_command_rad_s', 'Command', '#828aad'))),
        ('Depth (mm)', (('insertion_depth_mm', 'Thread travel', '#1b6d74'),)),
    )
    fig = Figure(figsize=(max(5.8, len(sites) * 4.6), 7), layout='constrained', facecolor='#f5f8f8')
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(len(series), len(sites))
    for column, (site, samples) in enumerate(sites.items()):
        start = samples[0]['sim_time_s']
        times = np.array([sample['sim_time_s'] - start for sample in samples])
        cuts = [0] + [index for index in range(1, len(samples))
                      if samples[index]['stage'] != samples[index - 1]['stage']] + [len(samples)]
        jam_times = [e['sim_time_s'] - start for e in events if e['site'] == site and e['event'] == 'jam_detected']
        for row, (ylabel, fields) in enumerate(series):
            axes = fig.add_subplot(grid[row, column])
            for begin, end in zip(cuts, cuts[1:]):
                stage = samples[begin]['stage']
                left, right = times[begin], times[end - 1]
                axes.axvspan(left, right, color=colors[stage], zorder=0)
                if row == 0 and right - left >= .25:
                    label = labels[stage]
                    if stage == 'probe':
                        label += f' {samples[begin]["attempt"]}'
                    axes.text((left + right) / 2, .94, label, fontsize=8, ha='center', va='top',
                              transform=axes.get_xaxis_transform())
            for key, label, color in fields:
                axes.plot(times, [sample[key] for sample in samples], color=color, label=label, linewidth=1.1)
            for time_at_jam in jam_times:
                axes.axvline(time_at_jam, color='#a83026', linestyle='--', linewidth=1)
            if row == 0:
                axes.set_title(site, loc='left')
                axes.axhline(1000 * FORCE_ENTRY_TOLERANCE_M, color='#a83026', linestyle=':', linewidth=1)
            if column == 0:
                axes.set_ylabel(ylabel)
            if len(fields) > 1:
                axes.legend(loc='upper right', fontsize=8)
            if row == len(series) - 1:
                axes.set_xlabel('Time since probe start (s)')
            else:
                axes.tick_params(labelbottom=False)
            axes.set_xlim(times[0], max(times[-1], times[0] + 0.04))
            axes.grid(alpha=.19)
    fig.savefig(folder / 'force_trace.png', dpi=135)
    return True


def plot_thread_contact(folder, rows, events, site):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    stages = ('probe', 'angled_entry', 'tighten', 'jam_brake', 'unwind', 'retry_retract', 'retry_realign')
    samples = [row for row in rows if row['site'] == site and row['stage'] in stages]
    if not samples:
        return False
    start = samples[0]['sim_time_s']
    times = [row['sim_time_s'] - start for row in samples]
    fig = Figure(figsize=(10.5, 8.2), layout='constrained', facecolor='#f5f8f8')
    FigureCanvasAgg(fig)
    series = (
        ('Tilt (deg)', ('tool_tilt_deg', 'Driver'), ('tilt_deg', 'Screw')),
        ('Load (N)', ('flank_force_n', 'Thread flank contact'), ('tool_lateral_n', 'Wrist XY')),
        ('Torque (N·m)', ('spindle_torque_nm', 'Spindle')),
        ('Speed (rad/s)', ('spindle_velocity_rad_s', 'Measured'), ('spindle_velocity_command_rad_s', 'Command')),
        ('Depth (mm)', ('insertion_depth_mm', 'Screw')),
    )
    markers = {'jam_detected': ('Jam', '#a93424'), 'unwind_reverse_measured': ('Reverse', '#ad721c'),
               'retry_axis_corrected': ('Axis corrected', '#67499d'), 'torque_cutoff': ('Seated', '#206350')}
    axes = fig.subplots(len(series), 1, sharex=True)
    for i, (ylabel, *fields) in enumerate(series):
        axis = axes[i]
        for (key, label), color in zip(fields, ('#186774', '#8960a2')):
            axis.plot(times, [row[key] for row in samples], color=color, linewidth=1.2, label=label)
        for event in events:
            if event['site'] == site and event['event'] in markers:
                label, color = markers[event['event']]
                at = event['sim_time_s'] - start
                axis.axvline(at, color=color, linewidth=.95, linestyle='--')
                if i == 0:
                    # Brake and reverse can be less than a tenth of a second apart.
                    level = .69 if event['event'] == 'unwind_reverse_measured' else .94
                    axis.text(at, level, f' {label}', color=color, fontsize=9,
                              ha='right' if event['event'] == 'torque_cutoff' else 'left',
                              va='top', transform=axis.get_xaxis_transform())
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.22)
        if len(fields) > 1:
            axis.legend(loc='lower center' if i == 0 else 'upper right', fontsize=8, ncol=2 if i == 0 else 1)
    axes[0].set_title(f'{site}  |  Thread contact and spindle recovery', loc='left')
    axes[-1].set_xlabel('Time since probe start (s)')
    axes[-1].set_xlim(0, max(.05, times[-1] + .1))
    fig.savefig(folder / 'thread_contact.png', dpi=135)
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
    if report['force_plot']:
        links.append('<a href="force_trace.png">找正、停转和反转曲线</a>')
        visual = '<img src="force_trace.png" alt="横向偏差、腕部力、批头正反转与入孔深度">' + visual
    if report['thread_plot']:
        links.append('<a href="thread_contact.png">斜咬牙和扶正曲线</a>')
        visual = '<img src="thread_contact.png" alt="电批倾斜、牙侧接触力、扭矩、反转速度和旋入深度">' + visual
    labels = {'installed': '已装配', 'failed': '中止', 'working': '进行中', 'pending': '未装配'}
    rows = []
    attempt_labels = {'probing': '找正', 'driving': '旋入', 'rejected': '入牙前停机', 'jammed': '卡滞', 'seated': '入座'}
    for site, entry in report['positions'].items():
        retained, cutoff = entry['after_retract'], entry['torque_cutoff']
        depth = f'{retained["insertion_depth_mm"]:.2f} mm' if retained else '—'
        turns = f'{retained["thread_turns"]:.2f}' if retained else '—'
        torque = f'{cutoff["spindle_torque_nm"]:.3f} N·m' if cutoff else '—'
        attempts = ' → '.join(attempt_labels[a['status']] for a in entry['attempts']) or '—'
        rows.append(f'<tr><th>{site}</th><td>{labels[entry["status"]]}</td><td>{attempts}</td><td>{depth}</td>'
                    f'<td>{turns}</td><td>{torque}</td></tr>')
    error = f'<p>{escape(report["error"])}</p>' if report['error'] else ''
    motion_label = {'baseline': '基准运动', 'optimized': '优化运动'}[report['motion']]
    insertion_label = '力反馈找正' if report['insertion'] == 'force' else '固定入孔'
    duration = f'{motion_label} · {insertion_label} · 整轮仿真 {report["cycle_time_s"]:.2f} s'
    alignment = ''
    thread_detail = ''
    if report['thread_model']:
        info = report['thread_model']
        tilt = report['entry_tilt_deg']
        recovery = {'stop': '只停转', 'retry': '原姿态重试', 'realign': '扶正后重试'}[report['recovery']]
        thread_detail = (f'<p>斜咬牙试验 · 入牙倾角 ({tilt[0]:g}°, {tilt[1]:g}°) · {recovery} · '
                         f'固定内牙 {info["female_flanks"]} 段 · 挡块 {info["obstruction_geoms"]} 个。</p>')
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
<main><p>MuJoCo · UR5e</p><h1>{title}</h1><p>{duration}</p>{thread_detail}{alignment}{error}<table><tr><th>位置</th><th>结果</th><th>装配尝试</th><th>撤刀后深度</th>
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
    female = rig.female_flanks
    fixed_xpos = rig.data.geom_xpos[female].copy()
    fixed_xmat = rig.data.geom_xmat[female].copy()
    result = dict(version=4, source='simulation', scenario=rig.scenario,
                  fault_site=rig.mechanical_fault_site if rig.scenario != 'none' else None,
                  seed=rig.seed, image_fault=rig.image_fault, image_site=rig.image_site, conditions=rig.conditions,
                  motion=rig.motion, alignment=rig.alignment, fixture_shift_truth=asdict(rig.fixture_shift),
                  insertion=rig.insertion, entry_offset_mm=rig.entry_offset_mm, entry_offset_site=rig.entry_offset_site,
                  entry_tilt_deg=rig.entry_tilt_deg, recovery=rig.recovery, external_jam_fixture=rig.jam_mocap is not None,
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
    result['thread_model'] = dict(
        female_flanks=len(female), all_world_geoms=bool(np.all(rig.model.geom_bodyid[female] == 0)),
        unchanged=bool(np.array_equal(fixed_xpos, rig.data.geom_xpos[female])
                       and np.array_equal(fixed_xmat, rig.data.geom_xmat[female])),
        mocap_bodies=int(rig.model.nmocap), obstruction_geoms=sum(s.block >= 0 for s in rig.stations.values()),
    ) if female else None
    if not view.rows:
        view.rows.append(rig.snapshot())
    write_csv(folder / 'assembly_trace.csv', tuple(view.rows[0]), view.rows)
    result['plot'] = plot_trace(folder, view.rows)
    result['force_plot'] = rig.insertion == 'force' and plot_force_trace(folder, view.rows, rig.events)
    result['thread_plot'] = bool(female) and plot_thread_contact(folder, view.rows, rig.events, rig.mechanical_fault_site)
    result['video'] = record and (folder / 'assembly.mp4').stat().st_size > 0
    write_assembly_report(folder, result)
    print(f'{RESULTS[result["status"]]} · {folder / "report.html"}', flush=True)
    if result['error']:
        print(result['error'], flush=True)
    return folder, result


def run_assembly(detector, output, fault, *, sites=SITES, fault_site='S2', seed=0, image_fault='none',
                 image_site='S2', motion='optimized', alignment='fixed', fixture_shift=FixtureShift(),
                 alignment_fault='none', alignment_fault_site='S2', viewer=False, video=False,
                 insertion='fixed', entry_offset_mm=(0.0, 0.0), entry_offset_site='S2',
                 recovery='realign', entry_tilt_deg=None):
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
                       alignment_fault=alignment_fault, alignment_fault_site=alignment_fault_site,
                       insertion=insertion, entry_offset_mm=entry_offset_mm, entry_offset_site=entry_offset_site,
                       recovery=recovery, entry_tilt_deg=entry_tilt_deg)
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
