from pathlib import Path
from zipfile import ZipFile

import cv2
import mujoco
import numpy as np

from .detector import ANGLES, SITES


MODELS = Path(__file__).resolve().parents[1] / 'models'
FAULTS = ('none', 'glare', 'blur', 'both_bad')
HOME = np.array([0.0, -1.35, 1.85, -2.06, -1.57, 1.35])


def compile_model(xml):
    with ZipFile(MODELS / 'ur5e-meshes.zip') as archive:
        assets = {name: archive.read(name) for name in archive.namelist() if name.endswith('.obj')}
    return mujoco.MjModel.from_xml_string(xml, assets)


def make_model():
    return compile_model((MODELS / 'inspection.xml').read_text(encoding='utf-8'))


class Simulation:
    source = 'simulation'

    def __init__(self, model=None):
        self.model = make_model() if model is None else model
        self.data = mujoco.MjData(self.model)
        self.frame = self.model.site('camera_frame').id
        self.renderer = None
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0
        self.options.geomgroup[5] = 0
        self.options.sitegroup[:] = 0
        self.parking = HOME.copy()
        self.active_pose = None
        self.on_frame = None
        self.plans = {(site, angle): self.plan_pose(site, angle) for site in SITES for angle in ANGLES}
        self.reset()

    def plan_pose(self, site, angle):
        target = self.model.site(site).pos
        offset = np.array([0.0, 0.0, 0.17] if angle == 'primary' else [0.056, -0.021, 0.169])
        point = target + offset
        z_axis = offset / np.linalg.norm(offset)
        x_axis = np.array([1.0, 0.0, 0.0]) - z_axis[0] * z_axis
        x_axis /= np.linalg.norm(x_axis)
        rotation = np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis))
        return self.solve_pose(self.frame, point, rotation), point, rotation

    def solve_pose(self, frame, point, rotation, seed=HOME):
        data = mujoco.MjData(self.model)
        data.qpos[:6] = seed
        pos_jac = np.zeros((3, self.model.nv))
        rot_jac = np.zeros_like(pos_jac)
        quaternion = np.zeros(4)
        rot_error = np.zeros(3)
        for _ in range(200):
            mujoco.mj_forward(self.model, data)
            pos_error = point - data.site_xpos[frame]
            current = data.site_xmat[frame].reshape(3, 3)
            mujoco.mju_mat2Quat(quaternion, (rotation @ current.T).ravel())
            if quaternion[0] < 0:
                quaternion *= -1
            mujoco.mju_quat2Vel(rot_error, quaternion, 1.0)
            if np.linalg.norm(pos_error) < 1e-5 and np.linalg.norm(rot_error) < 2e-4:
                return data.qpos[:6].copy()
            mujoco.mj_jacSite(self.model, data, pos_jac, rot_jac, frame)
            jacobian = np.vstack((pos_jac[:, :6], 0.18 * rot_jac[:, :6]))
            error = np.r_[pos_error, 0.18 * rot_error]
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.001**2 * np.eye(6), error)
            scale = max(1.0, np.max(np.abs(step)) / 0.12)
            bounds = self.model.jnt_range[:6]
            data.qpos[:6] = np.clip(data.qpos[:6] + step / scale, bounds[:, 0], bounds[:, 1])
        raise RuntimeError(f'MuJoCo 逆运动学未收敛，目标 {point.round(4)}')

    def reset(self, missing=(), fault='none', fault_site='S2', seed=0):
        if not set(missing) <= set(SITES) or fault not in FAULTS or fault_site not in SITES:
            raise ValueError('仿真工况无效')
        self.fault, self.fault_site = fault, fault_site
        self.rng = np.random.default_rng(seed)
        self.variation = seed != 0
        light = self.model.light('task_light').id
        self.model.light_diffuse[light] = 0.75 * (self.rng.uniform(0.80, 1.07) if self.variation else 1)
        for site in SITES:
            for part in ('head', 'slot1', 'slot2'):
                self.model.geom(f'{part}_{site}').group[0] = 5 if site in missing else 0
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = self.parking
        self.data.ctrl[:6] = self.parking
        mujoco.mj_forward(self.model, self.data)
        self.active_pose = None

    def motion_commands(self, start, goal, speed):
        delta = goal - start
        count = max(1, int(np.ceil(np.max(np.abs(delta)) / (speed * self.model.opt.timestep))))
        for step in range(1, count + 1):
            ratio = step / count
            yield start + (ratio**2 * (3 - 2 * ratio)) * delta

    def drive(self, goal, *, speed=0.8, tolerance=3e-4):
        began = self.data.time
        start = self.data.ctrl[:6].copy()
        for command in self.motion_commands(start, goal, speed):
            self.data.ctrl[:6] = command
            self.step()
        trajectory_end = self.data.time
        for _ in range(round(4 / self.model.opt.timestep)):
            if (np.max(np.abs(self.data.qpos[:6] - goal)) < tolerance
                    and np.max(np.abs(self.data.qvel[:6])) < 0.002):
                return float(trajectory_end - began), float(self.data.time - trajectory_end)
            self.step()
        raise RuntimeError(f'MuJoCo 关节未在时限内稳定，偏差 {np.max(np.abs(self.data.qpos[:6] - goal)):.4f} rad')

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if any(warning.number for warning in self.data.warning):
            raise RuntimeError('MuJoCo 动力学推进异常')
        self.check_contacts()
        if self.on_frame is not None:
            self.on_frame()

    def check_contacts(self):
        if self.data.ncon:
            self.reject_contact(self.data.contact[0])

    def reject_contact(self, contact):
        raise RuntimeError(f'MuJoCo 检测到碰撞 {self.model.geom(contact.geom1).name}/'
                           f'{self.model.geom(contact.geom2).name}')

    def capture(self, site, angle):
        self.active_pose = (site, angle)
        goal, _, _ = self.plans[self.active_pose]
        self.drive(goal)
        mujoco.mj_forward(self.model, self.data)
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, width=640, height=480)
        self.renderer.update_scene(self.data, camera='inspection_camera', scene_option=self.options)
        image = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        if self.variation:
            shift = self.rng.integers(-2, 3, size=2).astype(np.float32)
            transform = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            image = cv2.warpAffine(image, transform, (640, 480), borderMode=cv2.BORDER_REFLECT)
            noise = self.rng.normal(0, 0.7, image.shape[:2]).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
        if site == self.fault_site:
            if angle == 'primary' and self.fault in ('glare', 'both_bad'):
                cv2.ellipse(image, (319, 242), (38, 32), -20, 0, 360, (255, 255, 255), -1)
            if (self.fault, angle) in (('blur', 'primary'), ('both_bad', 'alternate')):
                image = cv2.GaussianBlur(image, (31, 31), 8)
        return image

    def finish(self):
        self.active_pose = None
        self.drive(self.parking)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
