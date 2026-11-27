import os
import tempfile
import threading
import time
from collections import deque
from contextlib import ExitStack, closing
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .detector import MISSING, PRESENT, REVIEW, SITES
from .samples import write_csv
from .simulation import Simulation
from .workflow import inspect_workpiece, write_report


FPS = 30
VIEW_SIZE = (864, 640)
VIDEO_SIZE = (1280, 640)
STATUS = {None: 'Waiting', PRESENT: 'Present', MISSING: 'Missing', REVIEW: 'Review'}
VERDICT = {'pass': 'PASS', MISSING: 'MISSING SCREW', REVIEW: 'REVIEW REQUIRED'}
COLORS = {None: (0.47, 0.61, 0.72), PRESENT: (0.35, 0.91, 0.68),
          MISSING: (1.0, 0.43, 0.36), REVIEW: (0.98, 0.75, 0.34)}


def overview_camera(camera):
    camera.lookat[:] = (-0.30, 0.0, 0.28)
    camera.distance = 1.43
    camera.azimuth = 138
    camera.elevation = -24
    return camera


def color(state):
    return tuple(round(channel * 255) for channel in COLORS[state])


def overlay_text(frame, message, xy, *, scale=0.64, ink=(220, 231, 239), weight=1):
    cv2.putText(frame, message, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, ink, weight, cv2.LINE_AA)


class Demo:
    def __init__(self, rig, viewer=None, video=None):
        self.rig = rig
        self.viewer = viewer
        self.view_data = viewer.d if viewer else None
        self.video = video
        self.renderer = None
        self.camera = overview_camera(mujoco.MjvCamera())
        self.states = dict.fromkeys(SITES)
        self.attempts = dict.fromkeys(SITES, 0)
        self.snapshots = {}
        self.photo = None
        self.photo_label = ''
        self.phase = 'Ready'
        self.verdict = None
        self.samples = []
        self.trail = deque(maxlen=120)
        self.next_frame = 0.0
        self.started = time.perf_counter()
        self.hold_time = 0.0
        if viewer:
            with viewer.lock():
                overview_camera(viewer.cam)
                viewer.opt.geomgroup[:] = rig.options.geomgroup
                viewer.opt.sitegroup[:] = rig.options.sitegroup

    def sample(self, event='motion'):
        data = self.rig.data
        if event == 'motion' and data.time < self.next_frame:
            return
        # mj_step integrates qpos after computing kinematics. Refresh site positions for those new angles.
        mujoco.mj_kinematics(self.rig.model, data)
        if event == 'motion':
            self.next_frame += 1 / FPS
        position = data.site_xpos[self.rig.frame]
        self.trail.append(position.copy())
        active = self.rig.active_pose
        self.phase = 'Returning to home' if active is None else f'{active[0]} / {active[1].title()}'
        if active is None:
            target = [None] * 3
            pos_err, rot_err = None, None
        else:
            _, target, desired = self.rig.plans[active]
            actual = data.site_xmat[self.rig.frame].reshape(3, 3)
            pos_err = 1000 * np.linalg.norm(target - position)
            rot_err = np.degrees(np.arccos(np.clip((np.trace(desired @ actual.T) - 1) / 2, -1, 1)))
        self.samples.append(dict(
            sim_time_s=round(float(data.time), 6), event=event, site=active[0] if active else '',
            angle=active[1] if active else '', camera_x_m=position[0], camera_y_m=position[1], camera_z_m=position[2],
            target_x_m=target[0], target_y_m=target[1], target_z_m=target[2],
            position_error_mm=pos_err, orientation_error_deg=rot_err,
            **{f'q{index}_rad': angle for index, angle in enumerate(data.qpos, start=1)},
        ))
        if self.viewer and self.viewer.is_running():
            time.sleep(max(0.0, self.started + data.time + self.hold_time - time.perf_counter()))
        self.draw()

    def after_capture(self, site, angle, observation, image):
        self.states[site] = observation.state
        self.attempts[site] += 1
        self.photo = None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detail = STATUS[observation.state]
        if observation.state == REVIEW:
            if '反光' in observation.reason:
                detail = 'Glare'
            elif '模糊' in observation.reason:
                detail = 'Blur'
            detail += ' - retry' if angle == 'primary' else ' - review'
        self.photo_label = f'{site} / {angle.title()} / {detail}'
        if self.photo is not None:
            self.snapshots[site] = (self.photo, self.photo_label)
        self.sample('capture')
        self.hold(0.85)

    def finish(self, verdict):
        self.verdict = verdict
        for site in SITES:
            if self.states[site] == verdict and site in self.snapshots:
                self.photo, self.photo_label = self.snapshots[site]
                break
        self.draw_viewer()
        self.hold(1.8)

    def hold(self, seconds):
        if self.video:
            frame = self.video_frame()
            for _ in range(round(FPS * seconds)):
                self.video.send(frame)
        if self.viewer and self.viewer.is_running():
            time.sleep(max(0.0, self.started + self.rig.data.time + self.hold_time + seconds - time.perf_counter()))
        self.hold_time += seconds

    def geometry(self, scene):
        identity = np.eye(3).ravel()
        for site in SITES:
            target = self.rig.model.site(site).pos + [0, 0, 0.015]
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                (0.007, 0, 0), target, identity, (*COLORS[self.states[site]], 1))
            scene.ngeom += 1

        trail = list(self.trail)[::3]
        segments = [(start, end, (0.27, 0.78, 0.96, 0.73), 1.7)
                    for start, end in zip(trail, trail[1:]) if np.linalg.norm(end - start) > 0.0004]
        if self.rig.active_pose and not self.verdict:
            site, _ = self.rig.active_pose
            segments.append((self.rig.data.site_xpos[self.rig.frame], self.rig.model.site(site).pos,
                             (0.88, 0.79, 0.43, 0.72), 2.2))
        for start, end, rgba, width in segments:
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, (0, 0, 0), (0, 0, 0), identity, rgba)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)
            scene.ngeom += 1

    def draw_viewer(self):
        if self.viewer is None or not self.viewer.is_running():
            return
        with self.viewer.lock():
            mujoco.mj_copyData(self.view_data, self.rig.model, self.rig.data)
            scene = self.viewer.user_scn
            scene.ngeom = 0
            self.geometry(scene)
        top_left = mujoco.mjtGridPos.mjGRID_TOPLEFT
        state_text = '\n'.join(STATUS[self.states[site]] + (' (retaken)' if self.attempts[site] == 2 else '') for site in SITES)
        title = VERDICT[self.verdict] if self.verdict else self.phase
        help_text = ('Enter: repeat  |  O: report' if self.verdict else
                     'Press Enter to start' if self.phase == 'Ready' else '')
        self.viewer.set_texts([
            (None, top_left, 'UR5e  |  Screw inspection\n' + '\n'.join(SITES), title + '\n' + state_text),
            (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, help_text, ''),
        ])
        viewport = self.viewer.viewport
        if self.photo is not None and viewport and viewport.width >= 480 and viewport.height >= 280:
            width = min(viewport.width // 3, 352)
            height = width * 3 // 4
            rect = mujoco.MjrRect(viewport.width - width - 18, 24, width, height)
            inset = cv2.resize(self.photo, (width, height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(inset, (0, 0), (width, 25), (20, 31, 42), -1)
            overlay_text(inset, self.photo_label, (8, 17), scale=0.45)
            self.viewer.set_images((rect, inset))
        else:
            self.viewer.clear_images()
        self.viewer.sync(state_only=True)

    def video_frame(self):
        width, height = VIEW_SIZE
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.rig.model, width=width, height=height)
        self.renderer.update_scene(self.rig.data, camera=self.camera, scene_option=self.rig.options)
        self.geometry(self.renderer.scene)
        frame = np.empty((height, VIDEO_SIZE[0], 3), dtype=np.uint8)
        frame[:, :width] = self.renderer.render()
        frame[:, width:] = (18, 29, 40)
        x = width + 24
        overlay_text(frame, 'UR5e SCREW INSPECTION', (x, 45), scale=0.81, weight=2)
        title = VERDICT[self.verdict] if self.verdict else self.phase
        ink = color(PRESENT if self.verdict == 'pass' else self.verdict)
        overlay_text(frame, title, (x, 82), scale=0.83, weight=2, ink=ink)
        for index, site in enumerate(SITES):
            y = 109 + index * 43
            cv2.rectangle(frame, (x, y), (VIDEO_SIZE[0] - 24, y + 34), (29, 44, 56), -1)
            overlay_text(frame, site, (x + 12, y + 23))
            state = self.states[site]
            retry = ' / retaken' if self.attempts[site] == 2 else ''
            overlay_text(frame, STATUS[state] + retry, (x + 98, y + 23), ink=color(state))
        overlay_text(frame, 'WRIST CAMERA', (x, 272), scale=0.57, weight=2)
        if self.photo is not None:
            frame[290:566, x:x + 368] = cv2.resize(self.photo, (368, 276), interpolation=cv2.INTER_AREA)
            overlay_text(frame, self.photo_label, (x, 590), scale=0.56)
        else:
            cv2.rectangle(frame, (x, 290), (x + 368, 566), (29, 44, 56), -1)
            overlay_text(frame, 'Waiting for first capture', (x + 23, 432), ink=color(None))
        return frame

    def draw(self):
        self.draw_viewer()
        if self.video:
            self.video.send(self.video_frame())

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def record_round(rig, window, movie, detector, output):
    with ExitStack() as recording:
        recording.callback(rig.close)
        writer = None
        if movie is not None:
            from imageio_ffmpeg import write_frames

            writer = recording.enter_context(closing(write_frames(
                str(movie), VIDEO_SIZE, fps=FPS, codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
                ffmpeg_log_level='error', output_params=['-movflags', '+faststart'],
            )))
            writer.send(None)
        demo = Demo(rig, window, writer)
        recording.callback(demo.close)
        demo.draw_viewer()
        demo.hold(0.50)
        rig.on_frame = demo.sample
        try:
            folder, report = inspect_workpiece(detector, 'simulation', lambda: rig, output, print,
                                              on_attempt=demo.after_capture, owns_rig=False)
        finally:
            rig.on_frame = None
        demo.finish(report['verdict'])
    if demo.samples:
        write_csv(folder / 'trajectory.csv', tuple(demo.samples[0]), demo.samples)
        report['trajectory'] = 'trajectory.csv'
    if movie is not None:
        if not movie.is_file() or not movie.stat().st_size:
            raise RuntimeError('MP4 编码未生成有效文件')
        movie.replace(folder / movie.name)
        report['video'] = movie.name
    write_report(folder, report)
    print(f'{VERDICT[report["verdict"]]} · {folder / "report.html"}', flush=True)
    return folder, report


def run_demo(detector, output, missing, fault, fault_site, seed, *, viewer=False, video=False):
    rig = Simulation()
    rig.reset(missing, fault, fault_site, seed)
    start_requested, report_requested = threading.Event(), threading.Event()

    def key_pressed(key):
        if key == 257:  # GLFW_KEY_ENTER
            start_requested.set()
        elif key == ord('O'):
            report_requested.set()

    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as resources:
        resources.callback(rig.close)
        movie = None
        if video:
            temp = resources.enter_context(tempfile.TemporaryDirectory(dir=output))
            movie = Path(temp) / 'inspection.mp4'
        window = None
        if viewer:
            from mujoco.viewer import launch_passive

            view_data = mujoco.MjData(rig.model)
            mujoco.mj_copyData(view_data, rig.model, rig.data)
            window = resources.enter_context(launch_passive(rig.model, view_data, key_callback=key_pressed,
                                                            show_left_ui=False, show_right_ui=False))
        if window is None:
            return record_round(rig, None, movie, detector, output)
        Demo(rig, window).draw_viewer()
        result = None
        while window.is_running():
            if report_requested.is_set():
                report_requested.clear()
                if result:
                    os.startfile(result[0] / 'report.html')
            if start_requested.wait(0.05) and window.is_running():
                start_requested.clear()
                rig.reset(missing, fault, fault_site, seed)
                result = record_round(rig, window, movie, detector, output)
                start_requested.clear()
        return result
