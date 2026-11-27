import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .detector import ANGLES, SITES
from .simulation import FAULTS, MODELS, Simulation, compile_model
from .vision_alignment import ALIGNMENT_FAULTS, HOLE_PLANE_Z_M, FixtureShift, Pinhole, find_hole, fit_fixture, mask_observation


ASSEMBLY_FAULTS = ('none', 'misaligned', 'blocked', 'empty_feeder')
MOTION_PROFILES = ('optimized', 'baseline')
ALIGNMENT_MODES = ('fixed', 'vision')
DEPTH_M = 0.006
PITCH_MM = 2.0
ENGAGE_XY_M = 0.00065
REFINE_XY_M = 0.00010
ENGAGE_TILT_DEG = 1.5
TORQUE_TARGET_NM = 0.14
TORQUE_MAX_NM = 0.20
BEARING_FRICTION_ARM_M = 0.0055
MAX_REFERENCE_ACCELERATION_RAD_S2 = 5.5


def instance(template, station):
    element = deepcopy(template)
    shared = {'spindle_angle', 'driver_tip'}
    for child in element.iter():
        for attr in ('name', 'joint1', 'joint2', 'site1', 'site2'):
            value = child.get(attr)
            if value is not None and value not in shared:
                child.set(attr, f'{value}_{station}')
    return element


def set_position(element, x=None, y=None):
    old_x, old_y, z = map(float, element.get('pos').split())
    element.set('pos', f'{old_x if x is None else x:.6f} {old_y if y is None else y:.6f} {z:.6f}')


def assembly_model(fault, stations, fault_site, conditions, fixture_shift, alignment):
    scene = ET.parse(MODELS / 'inspection.xml').getroot()
    extra = ET.parse(MODELS / 'assembly.xml').getroot()
    scene.set('model', 'UR5e vision-guided assembly' if alignment == 'vision' else 'UR5e screw assembly')
    scene.find('option').set('timestep', '0.0005')
    world = scene.find('worldbody')
    if fixture_shift != FixtureShift():
        moved = ('fixture', 'workpiece', 'left_clamp', 'right_clamp')
        prefixes = ('hole_', 'head_', 'slot1_', 'slot2_')
        for geom in world.findall('geom'):
            name = geom.get('name', '')
            if name in moved or name.startswith(prefixes):
                xy = fixture_shift.apply([float(v) for v in geom.get('pos').split()[:2]])
                set_position(geom, *xy)
                if name in moved or name.startswith(('slot1_', 'slot2_')):
                    euler = [float(v) for v in geom.get('euler', '0 0 0').split()]
                    euler[2] += math.radians(fixture_shift.yaw_deg)
                    geom.set('euler', ' '.join(f'{v:.9f}' for v in euler))
    equality = ET.SubElement(scene, 'equality')
    for station in stations:
        index = SITES.index(station)
        x, y, _ = map(float, world.find(f"site[@name='{station}']").get('pos').split())
        feeder_x = -0.430 if len(stations) == 1 else -0.430 + (index - 1) * 0.068
        empty = fault == 'empty_feeder' and station == fault_site
        for template in extra.find('worldbody'):
            name = template.get('name')
            if name == 'blocked_seat' and (fault != 'blocked' or station != fault_site):
                continue
            element = instance(template, station)
            if name in ('thread_guide', 'blocked_seat'):
                set_position(element, *fixture_shift.apply([x, y]))
            else:
                set_position(element, feeder_x)
            if name == 'loose_screw':
                dx, dy = conditions[station]['feeder_offset_mm']
                set_position(element, feeder_x + dx / 1000, (-0.339 if empty else -0.236) + dy / 1000)
                for i, slot in enumerate(element.findall("geom[@class='slot']")):
                    slot.set('euler', f'0 0 {0.2 + 0.3 * index + i * math.pi / 2:.6f}')
            world.append(element)
        if empty:
            pad = deepcopy(extra.find("worldbody/geom[@name='feeder_base']"))
            pad.set('name', f'reject_pad_{station}')
            set_position(pad, feeder_x, -0.339)
            world.append(pad)
        equality.extend(instance(template, station) for template in extra.find('equality'))
    scene.find(".//body[@name='wrist_3_link']").extend(extra.find('wrist'))
    scene.find('actuator').extend(extra.find('actuator'))
    home = scene.find(".//key[@name='home']")
    scene.find('keyframe').remove(home)
    return compile_model(ET.tostring(scene, encoding='unicode'))


class AssemblyStopped(RuntimeError):
    """An assembly gate stopped motion before completion."""

    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Station:
    name: str
    socket: int
    screw: int
    head: int
    block: int
    magnet: int
    guide: int
    key: int
    turn_joint: int
    turn_dof: int
    free_damping: float
    screw_nominal: np.ndarray
    entry_nominal: np.ndarray
    entry_actual: np.ndarray


class Assembly(Simulation):
    def __init__(self, fault='none', sites=SITES, fault_site='S2', *, seed=0, image_fault='none', image_site='S2',
                 motion='optimized', alignment='fixed', fixture_shift=FixtureShift(), alignment_fault='none',
                 alignment_fault_site='S2'):
        if fault not in ASSEMBLY_FAULTS:
            raise ValueError(f'未知装配工况 {fault}')
        self.sites = tuple(sites)
        if not self.sites or len(set(self.sites)) != len(self.sites) or not set(self.sites) <= set(SITES):
            raise ValueError('装配位置必须是不重复的 S1、S2、S3')
        if fault != 'none' and fault_site not in self.sites:
            raise ValueError(f'故障注入位置 {fault_site} 不在装配顺序内')
        if seed < 0 or image_fault not in FAULTS or image_site not in SITES:
            raise ValueError('仿真种子或图像故障设置无效')
        if motion not in MOTION_PROFILES:
            raise ValueError(f'未知运动曲线 {motion}')
        if alignment not in ALIGNMENT_MODES or alignment_fault not in ALIGNMENT_FAULTS or alignment_fault_site not in SITES:
            raise ValueError('视觉定位方式或故障设置无效')
        if alignment == 'vision' and self.sites != SITES:
            raise ValueError('视觉定位需要 S1、S2、S3 三个空孔')
        if alignment != 'vision' and alignment_fault != 'none':
            raise ValueError('定位图像故障需要视觉定位模式')
        self.motion = motion
        self.alignment, self.fixture_shift = alignment, fixture_shift
        self.alignment_fault, self.alignment_fault_site = alignment_fault, alignment_fault_site
        self.seed, self.image_fault, self.image_site = seed, image_fault, image_site
        rng = np.random.default_rng(seed)
        self.conditions = {site: dict(feeder_offset_mm=rng.uniform(-0.23, 0.23, 2).tolist() if seed else [0., 0.],
                                     position_bias_mm=rng.uniform(-0.28, 0.28, 2).tolist() if seed else [0., 0.],
                                     bearing_scale=float(rng.uniform(0.87, 1.13)) if seed else 1.) for site in self.sites}
        super().__init__(assembly_model(fault, self.sites, fault_site, self.conditions, fixture_shift, alignment))
        self.taught_camera_plans = self.plans.copy()
        # A MuJoCo general position servo produces Kp*(u-q)-Kd*qdot with the model's 1:1 gear.
        self.velocity_feedforward = -self.model.actuator_biasprm[:6, 2] / self.model.actuator_gainprm[:6, 0]
        self.scenario = fault
        self.mechanical_fault_site = fault_site
        self.motor = self.model.actuator('spindle').id
        self.spindle_joint = self.model.joint('spindle_angle').id
        self.reference = self.model.site('driver_reference').id
        self.tip = self.model.site('driver_tip').id
        self.plate = self.model.geom('workpiece').id
        self.bit = self.model.geom('driver_bit').id
        self.stations = {}
        self.allowed_pairs = set()
        for site in self.sites:
            model = self.model
            turn = model.joint(f'thread_angle_{site}').id
            dof = int(model.jnt_dofadr[turn])
            block = model.geom(f'blocked_seat_{site}').id if fault == 'blocked' and site == fault_site else -1
            feeder = model.geom(f'feeder_seat_{site}')
            nominal = feeder.pos.copy()
            nominal[2] += feeder.size[1]
            actual = model.body(f'thread_guide_{site}').pos.copy()
            taught = np.r_[model.site(site).pos[:2], actual[2]]
            station = Station(site, model.site(f'screw_socket_{site}').id, model.site(f'screw_axis_{site}').id,
                              model.geom(f'assembled_head_{site}').id, block,
                              model.equality(f'magnetic_bit_{site}').id, model.equality(f'thread_capture_{site}').id,
                              model.equality(f'keyed_drive_{site}').id, turn, dof, float(model.dof_damping[dof]),
                              nominal, taught, actual)
            self.stations[site] = station
            self.allowed_pairs.update(frozenset((station.head, geom)) for geom in (
                self.bit, self.plate, block, model.geom(f'feeder_seat_{site}').id, model.geom(f'feeder_base_{site}').id,
            ))
            if fault == 'empty_feeder' and site == fault_site:
                self.allowed_pairs.add(frozenset((station.head, model.geom(f'reject_pad_{site}').id)))
        self.reset_cycle()

    def reset_cycle(self):
        self.stage = 'ready'
        self.events = []
        self.moves = []
        self.registration = None
        self.alignment_attempts = []
        self.plans = self.taught_camera_plans.copy()
        self.station = self.stations[self.sites[0]]
        self.results = {site: dict(status='pending', after_retract=None, torque_cutoff=None, peak_torque_nm=0.0,
                                   peak_seat_force_n=0.0) for site in self.sites}
        for station in self.stations.values():
            self.model.dof_damping[station.turn_dof] = station.free_damping
            self.model.eq_data[station.key, 0] = 0
        self.reset(self.sites, self.image_fault, self.image_site, self.seed)

    def check_contacts(self):
        for contact in self.data.contact:
            if frozenset((contact.geom1, contact.geom2)) not in self.allowed_pairs:
                self.reject_contact(contact)

    def normal_force(self, station, *surfaces):
        pairs = {frozenset((station.head, geom)) for geom in surfaces}
        total = 0.0
        force = np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            if frozenset((contact.geom1, contact.geom2)) in pairs:
                mujoco.mj_contactForce(self.model, self.data, i, force)
                total += float(force[0])
        return total

    def snapshot(self, station=None):
        station = self.station if station is None else station
        data = self.data
        position = data.site_xpos[station.screw]
        tip = data.site_xpos[self.tip]
        normal = data.site_xmat[station.screw].reshape(3, 3)[:, 2]
        depth = station.entry_actual[2] - position[2]
        return dict(sim_time_s=float(data.time), stage=self.stage, site=station.name,
                    screw_x_m=float(position[0]), screw_y_m=float(position[1]), screw_z_m=float(position[2]),
                    tip_x_m=float(tip[0]), tip_y_m=float(tip[1]), tip_z_m=float(tip[2]),
                    radial_error_mm=float(1000 * np.linalg.norm(position[:2] - station.entry_actual[:2])),
                    tilt_deg=math.degrees(math.acos(float(np.clip(normal[2], -1, 1)))),
                    insertion_depth_mm=float(1000 * depth),
                    thread_turns=float(data.joint(station.turn_joint).qpos[0] / (2 * math.pi)),
                    spindle_velocity_command_rad_s=float(data.ctrl[self.motor]),
                    spindle_angle_rad=float(data.joint(self.spindle_joint).qpos[0]),
                    spindle_velocity_rad_s=float(data.joint(self.spindle_joint).qvel[0]),
                    spindle_torque_nm=float(abs(data.actuator_force[self.motor])),
                    seat_force_n=self.normal_force(station, self.plate),
                    obstruction_force_n=self.normal_force(station, station.block),
                    bit_contact_n=self.normal_force(station, self.bit),
                    magnet=bool(data.eq_active[station.magnet]), thread=bool(data.eq_active[station.guide]),
                    max_arm_velocity_rad_s=float(np.max(np.abs(data.qvel[:6]))),
                    **{f'q{index}_rad': float(angle) for index, angle in enumerate(data.qpos[:6], start=1)})

    def event(self, name):
        mujoco.mj_forward(self.model, self.data)
        self.events.append(dict(event=name, **self.snapshot()))

    def motion_commands(self, start, goal, speed):
        if self.motion == 'baseline':
            return super().motion_commands(start, goal, speed)
        delta = goal - start
        distance = float(np.max(np.abs(delta)))
        # The quintic's peak velocity is 1.875*distance/T. Keep the cubic's 1.5*speed bound.
        target_duration = max(0.10, 1.25 * distance / speed,
                              math.sqrt((10 / math.sqrt(3)) * distance / MAX_REFERENCE_ACCELERATION_RAD_S2))
        dt = self.model.opt.timestep
        steps = math.ceil(target_duration / dt)
        t = (np.arange(1, steps + 1, dtype=float) / steps)[:, None]
        position = t**3 * (10 + t * (6 * t - 15))
        velocity = 30 * t**2 * (1 - t)**2 / (steps * dt)
        bounds = self.model.actuator_ctrlrange[:6]
        commands = start + (position + velocity * self.velocity_feedforward) * delta
        return np.clip(commands, bounds[:, 0], bounds[:, 1])

    def drive(self, goal, *, speed=0.8, tolerance=3e-4):
        began = float(self.data.time)
        travel, settling = super().drive(goal, speed=speed, tolerance=tolerance)
        site, angle = self.active_pose or (self.station.name, None)
        self.moves.append(dict(stage=self.stage, site=site, angle=angle, began_at_s=began,
                               travel_s=travel, settling_s=settling,
                               settled_error_rad=float(np.max(np.abs(self.data.qpos[:6] - goal))),
                               settled_velocity_rad_s=float(np.max(np.abs(self.data.qvel[:6])))))
        return travel, settling

    def move_xyz(self, stage, xyz, speed=0.55):
        self.stage = stage
        goal = self.solve_pose(self.reference, xyz, np.eye(3), self.data.qpos[:6])
        self.drive(goal, speed=speed, tolerance=0.0018)
        mujoco.mj_forward(self.model, self.data)

    def step_for(self, duration):
        for _ in range(round(duration / self.model.opt.timestep)):
            self.step()
        mujoco.mj_forward(self.model, self.data)

    def register_fixture(self, on_attempt):
        self.stage = 'calibration'
        measurements = []
        camera_id = self.model.camera('inspection_camera').id
        taught = np.array([self.stations[site].entry_nominal[:2] for site in SITES])
        for site, xy in zip(SITES, taught):
            self.station = self.stations[site]
            observation = None
            for angle in ANGLES:
                image = self.capture_raw(site, angle)
                camera = Pinhole(self.data.cam_xpos[camera_id].copy(), self.data.cam_xmat[camera_id].reshape(3, 3).copy(),
                                 image.shape[1], image.shape[0], self.model.cam_fovy[camera_id])
                fault = self.alignment_fault if site == self.alignment_fault_site else 'none'
                if fault == 'glare' and angle == 'alternate':
                    fault = 'none'
                image = mask_observation(image, camera.project([*xy, HOLE_PLANE_Z_M]), fault)
                observation, reason, expected = find_hole(image, camera, xy)
                attempt = dict(site=site, angle=angle, at_s=float(self.data.time), reason=reason, nominal_uv=expected,
                               observation=asdict(observation) if observation else None)
                self.alignment_attempts.append(attempt)
                on_attempt(attempt, image, observation)
                if observation is not None:
                    break
            if observation is None:
                raise AssemblyStopped('两个定位视角均无法确认孔位', 'registration_image')
            measurements.append(observation.xy_m)
        try:
            self.registration, residuals = fit_fixture(taught, measurements)
        except ValueError as exc:
            raise AssemblyStopped(str(exc), 'registration_fit') from exc
        # Registered centers come from image geometry. The simulator's true sites are reserved for scoring.
        centers = self.registration.apply(taught)
        self.plans = {(site, angle): self.plan_pose(site, angle, np.r_[xy, self.model.site(site).pos[2]])
                      for site, xy in zip(SITES, centers) for angle in ANGLES}
        self.active_pose = None
        return dict(**asdict(self.registration), residuals_mm=dict(zip(SITES, residuals)),
                    centers_m={site: xy.tolist() for site, xy in zip(SITES, centers)})

    def assemble(self):
        if self.alignment == 'vision' and self.registration is None:
            raise AssemblyStopped('未完成视觉定位，装配尚未启动', 'registration_required')
        for site, station in self.stations.items():
            self.station = station
            result = self.results[site]
            result['status'] = 'working'
            try:
                result.update(after_retract=self.install_one(station), status='installed')
            except RuntimeError:
                result['status'] = 'failed'
                raise
        self.verify_all_retained()

    def index_bit(self):
        self.stage = 'index'
        joint = self.data.joint(self.spindle_joint)
        angle = float(joint.qpos[0])
        target = round(angle / math.tau) * math.tau
        try:
            for _ in range(round(2 / self.model.opt.timestep)):
                error = target - float(joint.qpos[0])
                if abs(error) < 0.006 and abs(joint.qvel[0]) < 0.04:
                    return
                self.data.ctrl[self.motor] = np.clip(10 * error, -6, 6)
                self.step()
            raise AssemblyStopped('批头未回到可取件的角度', 'bit_index')
        finally:
            self.data.ctrl[self.motor] = 0.0

    def install_one(self, station):
        feed = station.screw_nominal.copy()
        feed[2] += self.model.site(station.socket).pos[2]
        home_above_feed = feed + (0, 0, 0.073)
        self.move_xyz('approach_feed', home_above_feed)
        self.index_bit()
        self.move_xyz('pick', feed - (0, 0, 0.0015), speed=0.34)
        tip_gap = np.linalg.norm(self.data.site_xpos[self.tip] - self.data.site_xpos[station.socket])
        if tip_gap > 0.0014 or self.normal_force(station, self.bit) < 0.15:
            raise AssemblyStopped(f'批头未可靠接触供料位螺丝，间距 {1000 * tip_gap:.2f} mm', 'pickup_contact')
        self.event('pickup_contact')
        self.data.eq_active[station.magnet] = True
        self.move_xyz('lift', home_above_feed)
        if self.data.site_xpos[station.screw, 2] - station.screw_nominal[2] < 0.050:
            raise AssemblyStopped('提起后未检测到螺丝离开供料座', 'pickup_retention')
        self.event('screw_lifted')

        approach = station.entry_nominal.copy()
        if self.registration is not None:
            approach[:2] = self.registration.apply(approach[:2])
        approach[2] += self.model.site(station.socket).pos[2]
        approach[:2] += np.asarray(self.conditions[station.name]['position_bias_mm']) / 1000
        if self.scenario == 'misaligned' and station.name == self.mechanical_fault_site:
            approach[0] += 0.003
        self.move_xyz('transfer', approach + (0, 0, 0.090))
        self.move_xyz('align', approach + (0, 0, 0.016), speed=0.34)
        start_pose = self.snapshot()
        self.event('axis_alignment')
        if start_pose['radial_error_mm'] > 1000 * ENGAGE_XY_M or start_pose['tilt_deg'] > ENGAGE_TILT_DEG:
            raise AssemblyStopped(f"对孔超差，横向 {start_pose['radial_error_mm']:.2f} mm，未驱动批头", 'alignment')
        if self.registration is None:
            correction = station.entry_nominal[:2] - self.data.site_xpos[station.screw, :2]
        else:
            correction = self.registration.apply(station.entry_nominal[:2]) - self.data.site_xpos[self.tip, :2]
        if np.linalg.norm(correction) > REFINE_XY_M:
            approach[:2] += correction
            self.move_xyz('align', approach + (0, 0, 0.016), speed=0.20)
            self.event('alignment_refined')
        # The unloaded driver reference is calibrated with the bit at spring equilibrium.
        self.move_xyz('insert', approach, speed=0.20)
        at_entry = self.snapshot()
        if (abs(at_entry['insertion_depth_mm']) > 0.85 or at_entry['radial_error_mm'] > 1000 * ENGAGE_XY_M
                or at_entry['tilt_deg'] > ENGAGE_TILT_DEG):
            raise AssemblyStopped('螺丝没有对准螺纹入口', 'thread_entry')
        self.model.eq_data[station.key, 0] = (self.data.joint(self.spindle_joint).qpos[0]
                                              - self.data.joint(station.turn_joint).qpos[0])
        self.data.eq_active[station.guide] = True
        self.data.eq_active[station.key] = True
        self.event('thread_engaged')
        self.tighten(station)
        self.brake_spindle()
        self.event('spindle_stopped')

        self.stage = 'release'
        self.data.eq_active[station.magnet] = False
        self.data.eq_active[station.key] = False
        self.event('driver_released')
        self.move_xyz('retract', approach + (0, 0, 0.090))
        self.step_for(0.20)
        retained = self.verify_retained(station)
        self.event('retention_verified')
        return retained

    def verify_all_retained(self):
        for station in self.stations.values():
            try:
                self.verify_retained(station)
            except AssemblyStopped:
                self.results[station.name]['status'] = 'failed'
                self.station = station
                raise

    def verify_retained(self, station):
        retained = self.snapshot(station)
        if (abs(retained['insertion_depth_mm'] - DEPTH_M * 1000) > 0.55
                or retained['radial_error_mm'] > 0.30 or retained['magnet'] or not retained['thread']):
            raise AssemblyStopped(f'{station.name} 撤刀后螺丝未保持在装配位', 'retention')
        return retained

    def brake_spindle(self):
        self.data.ctrl[self.motor] = 0.0
        speed = self.data.joint(self.spindle_joint).qvel
        stable = 0
        dt = self.model.opt.timestep
        for _ in range(round(0.75 / dt)):
            self.step()
            stable = stable + 1 if abs(speed[0]) <= 0.08 else 0
            if stable >= round(0.045 / dt):
                return
        raise AssemblyStopped('批头在时限内未稳定停转', 'spindle_brake')

    def tighten(self, station):
        self.stage = 'tighten'
        result = self.results[station.name]
        steady = 0
        dt = self.model.opt.timestep
        bearing_arm = BEARING_FRICTION_ARM_M * self.conditions[station.name]['bearing_scale']
        try:
            for step in range(round(8 / dt)):
                depth = station.entry_actual[2] - self.data.site_xpos[station.screw, 2]
                speed = 5.6 if depth >= 0.0054 else 13.0
                self.data.ctrl[self.motor] = min(speed, (step + 1) * dt * 50)
                rate = abs(self.data.qvel[station.turn_dof])
                # Regularized under-head friction from the measured normal load. Implicit damping
                # dissipates power even when the contact begins within a single integration step.
                bearing_force = self.normal_force(station, self.plate, station.block)
                self.model.dof_damping[station.turn_dof] = (station.free_damping
                    + bearing_arm * bearing_force / max(rate, 0.30))
                self.step()
                sample = self.snapshot()
                torque = sample['spindle_torque_nm']
                force = sample['seat_force_n']
                result['peak_torque_nm'] = max(result['peak_torque_nm'], torque)
                result['peak_seat_force_n'] = max(result['peak_seat_force_n'], force)
                depth = sample['insertion_depth_mm']
                rate = abs(float(self.data.qvel[station.turn_dof]))
                if sample['obstruction_force_n'] > 3.0 and depth < DEPTH_M * 1000 - 0.7:
                    raise AssemblyStopped(f'未达到入座深度即触碰障碍，深度 {depth:.2f} mm，已停转', 'obstruction')
                if depth > 6.65 or sample['radial_error_mm'] > 0.8 or sample['tilt_deg'] > 3.5:
                    raise AssemblyStopped('旋入超出深度或轴线容差，已停转', 'thread_limits')
                pitch_error = abs(depth - PITCH_MM * sample['thread_turns'])
                seated = (abs(depth - DEPTH_M * 1000) <= 0.6 and pitch_error <= 0.25 and force >= 4.0
                          and rate <= 0.7 and TORQUE_TARGET_NM <= torque <= TORQUE_MAX_NM + 1e-6)
                steady = steady + 1 if seated else 0
                if steady >= round(0.10 / dt):
                    result['torque_cutoff'] = {**sample, 'thread_rate_rad_s': rate}
                    self.event('torque_cutoff')
                    return
            final = self.snapshot()
            raise AssemblyStopped(f"旋紧未达到联合门限，深度 {final['insertion_depth_mm']:.2f} mm，"
                                  f"扭矩 {final['spindle_torque_nm']:.3f} N·m，入座力 {final['seat_force_n']:.1f} N",
                                  'torque_gate')
        finally:
            self.data.ctrl[self.motor] = 0.0
