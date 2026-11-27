import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .detector import ANGLES, SITES
from .force_insertion import (CONTACT_DT_S, FORCE_ENTRY_TOLERANCE_M, INSERTION_MODES, LATERAL_DEADBAND_N,
                              MAX_ATTEMPTS, MAX_PROBE_FORCE_N, MAX_SEARCH_M, OBSTRUCTION_FAULTS,
                              add_chamfer, add_chamfer_meshes, correction_step)
from .simulation import FAULTS, MODELS, Simulation, compile_model
from .thread_contact import (FLANK_DRAG_ARM_M, MAX_ANGLED_ENTRY_DEG, REALIGN_TILT_DEG, RECOVERY_MODES, THREAD_PITCH_MM,
                             add_thread_flanks, initial_axis, rotation_without_yaw)
from .thread_health import DAMAGE_FAULTS, SlipMonitor, stripped_thread_forces
from .vision_alignment import ALIGNMENT_FAULTS, HOLE_PLANE_Z_M, FixtureShift, Pinhole, find_hole, fit_fixture, mask_observation


ASSEMBLY_FAULTS = ('none', 'misaligned', *OBSTRUCTION_FAULTS, 'cross_thread', *DAMAGE_FAULTS, 'empty_feeder')
MOTION_PROFILES = ('optimized', 'baseline')
ALIGNMENT_MODES = ('fixed', 'vision')
DEPTH_M = 0.006
PITCH_MM = THREAD_PITCH_MM
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


def assembly_model(fault, stations, fault_site, conditions, fixture_shift, alignment, insertion):
    scene = ET.parse(MODELS / 'inspection.xml').getroot()
    extra = ET.parse(MODELS / 'assembly.xml').getroot()
    normal_title = ('UR5e force-guided assembly' if insertion == 'force' else
                    'UR5e vision-guided assembly' if alignment == 'vision' else 'UR5e screw assembly')
    fault_titles = dict.fromkeys(DAMAGE_FAULTS, 'UR5e thread damage inspection')
    fault_titles['cross_thread'] = 'UR5e cross-thread recovery'
    title = fault_titles.get(fault, normal_title)
    scene.set('model', title)
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
    chamfer_count = add_chamfer_meshes(scene.find('asset'))
    equality = ET.SubElement(scene, 'equality')
    for station in stations:
        index = SITES.index(station)
        x, y, _ = map(float, world.find(f"site[@name='{station}']").get('pos').split())
        feeder_x = -0.430 if len(stations) == 1 else -0.430 + (index - 1) * 0.068
        empty = fault == 'empty_feeder' and station == fault_site
        xy = fixture_shift.apply([x, y])
        add_chamfer(world, station, xy, chamfer_count)
        for template in extra.find('worldbody'):
            name = template.get('name')
            if name == 'blocked_seat' and (fault not in OBSTRUCTION_FAULTS or station != fault_site):
                continue
            element = instance(template, station)
            if name in ('thread_guide', 'blocked_seat'):
                set_position(element, *xy)
            else:
                set_position(element, feeder_x)
            if name == 'loose_screw':
                dx, dy = conditions[station]['feeder_offset_mm']
                set_position(element, feeder_x + dx / 1000, (-0.339 if empty else -0.236) + dy / 1000)
                for i, slot in enumerate(element.findall("geom[@class='slot']")):
                    slot.set('euler', f'0 0 {0.2 + 0.3 * index + i * math.pi / 2:.6f}')
                if fault in ('cross_thread', 'damaged_thread') and station == fault_site:
                    guide = extra.find("worldbody/body[@name='thread_guide']")
                    add_thread_flanks(scene.find('asset'), world, element, xy, float(guide.get('pos').split()[2]),
                                      crushed=fault == 'damaged_thread')
            if name == 'blocked_seat' and fault == 'retryable_jam':
                holder = ET.SubElement(world, 'body', name=f'jam_fixture_{station}', mocap='true')
                holder.append(element)
            else:
                world.append(element)
        if empty:
            pad = deepcopy(extra.find("worldbody/geom[@name='feeder_base']"))
            pad.set('name', f'reject_pad_{station}')
            set_position(pad, feeder_x, -0.339)
            world.append(pad)
        equality.extend(instance(template, station) for template in extra.find('equality'))
    scene.find(".//body[@name='wrist_3_link']").extend(extra.find('wrist'))
    scene.find('actuator').extend(extra.find('actuator'))
    scene.append(extra.find('sensor'))
    home = scene.find(".//key[@name='home']")
    scene.find('keyframe').remove(home)
    return compile_model(ET.tostring(scene, encoding='unicode'))


class AssemblyStopped(RuntimeError):
    """An assembly gate stopped motion before completion."""

    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


class AssemblyReview(AssemblyStopped):
    """Fastening failed a quality gate and the assembly requires inspection."""


class InsertionAnomaly(AssemblyStopped):
    attempt_status = 'unverified'
    review_code = 'thread_signature_unverified'


class InsertionJammed(InsertionAnomaly):
    attempt_status = 'jammed'
    review_code = 'jam_retry_limit'


class InsertionSlipped(InsertionAnomaly):
    attempt_status = 'slipped'
    review_code = 'thread_slip_retry_limit'


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
    flank_pairs: frozenset
    thread_entry: int
    depth_dof: int
    pitch: int


class Assembly(Simulation):
    def __init__(self, fault='none', sites=SITES, fault_site='S2', *, seed=0, image_fault='none', image_site='S2',
                 motion='optimized', alignment='fixed', fixture_shift=FixtureShift(), alignment_fault='none',
                 alignment_fault_site='S2', insertion='fixed', entry_offset_mm=(0.0, 0.0), entry_offset_site='S2',
                 recovery='realign', entry_tilt_deg=None, strip_depth_mm=None):
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
        if insertion not in INSERTION_MODES:
            raise ValueError(f'未知对孔方式 {insertion}')
        if recovery not in RECOVERY_MODES:
            raise ValueError(f'未知重试方式 {recovery}')
        if fault in ('cross_thread', *DAMAGE_FAULTS) and insertion != 'force':
            raise ValueError('斜咬牙和牙损试验需要力反馈入孔')
        if strip_depth_mm is not None and fault != 'stripped_thread':
            raise ValueError('滑牙起始深度只用于滑牙工况')
        if fault == 'stripped_thread':
            strip_depth_mm = 2.4 if strip_depth_mm is None else strip_depth_mm
            if not 1.3 <= strip_depth_mm <= 4.2:
                raise ValueError('滑牙起始深度须在 1.3 至 4.2 mm')
        self.strip_depth_mm = strip_depth_mm
        if entry_tilt_deg is not None and fault not in ('cross_thread', 'damaged_thread'):
            raise ValueError('入口倾角可用于斜咬牙和牙侧损伤试验')
        tilt = np.asarray((5.25, 0) if entry_tilt_deg is None else entry_tilt_deg, dtype=float)
        if (tilt.shape != (2,) or not np.isfinite(tilt).all() or np.linalg.norm(tilt) > MAX_ANGLED_ENTRY_DEG):
            raise ValueError('入口倾角的 XY 合计幅值不得超过 6°')
        self.recovery = recovery
        self.entry_tilt_deg = tilt.tolist() if fault == 'cross_thread' or entry_tilt_deg is not None else None
        offset = np.asarray(entry_offset_mm, dtype=float)
        if (offset.shape != (2,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > 5
                or (np.any(offset) and entry_offset_site not in self.sites)):
            raise ValueError('对孔扰动需要指定已选螺丝位，XY 合计幅值不得超过 5 mm')
        self.insertion = insertion
        self.entry_offset_mm = offset.tolist()
        self.entry_offset_site = entry_offset_site
        self.motion = motion
        self.alignment, self.fixture_shift = alignment, fixture_shift
        self.alignment_fault, self.alignment_fault_site = alignment_fault, alignment_fault_site
        self.seed, self.image_fault, self.image_site = seed, image_fault, image_site
        rng = np.random.default_rng(seed)
        self.conditions = {site: dict(feeder_offset_mm=rng.uniform(-0.23, 0.23, 2).tolist() if seed else [0., 0.],
                                     position_bias_mm=rng.uniform(-0.28, 0.28, 2).tolist() if seed else [0., 0.],
                                     bearing_scale=float(rng.uniform(0.87, 1.13)) if seed else 1.) for site in self.sites}
        super().__init__(assembly_model(fault, self.sites, fault_site, self.conditions, fixture_shift, alignment, insertion))
        self.taught_camera_plans = self.plans.copy()
        # A MuJoCo general position servo produces Kp*(u-q)-Kd*qdot with the model's 1:1 gear.
        self.velocity_feedforward = -self.model.actuator_biasprm[:6, 2] / self.model.actuator_gainprm[:6, 0]
        self.scenario = fault
        self.mechanical_fault_site = fault_site
        self.motor = self.model.actuator('spindle').id
        self.spindle_joint = self.model.joint('spindle_angle').id
        self.reference = self.model.site('driver_reference').id
        self.tip = self.model.site('driver_tip').id
        self.load_cell = self.model.site('driver_load_cell').id
        self.force_sensor = self.model.sensor('driver_force').id
        self.torque_sensor = self.model.sensor('driver_torque').id
        self.jam_mocap = (int(self.model.body(f'jam_fixture_{fault_site}').mocapid[0])
                          if fault == 'retryable_jam' else None)
        self.plate = self.model.geom('workpiece').id
        self.bit = self.model.geom('driver_bit').id
        self.stations = {}
        self.allowed_pairs = set()
        male = [g for g in range(self.model.ngeom) if self.model.geom(g).name.startswith('male_flank_')]
        self.female_flanks = [g for g in range(self.model.ngeom) if self.model.geom(g).name.startswith('female_flank_')]
        flank_pairs = frozenset(frozenset((a, b)) for a in male for b in self.female_flanks)
        self.allowed_pairs.update(flank_pairs)
        for site in self.sites:
            model = self.model
            turn = model.joint(f'thread_angle_{site}').id
            dof = int(model.jnt_dofadr[turn])
            block = model.geom(f'blocked_seat_{site}').id if fault in OBSTRUCTION_FAULTS and site == fault_site else -1
            feeder = model.geom(f'feeder_seat_{site}')
            nominal = feeder.pos.copy()
            nominal[2] += feeder.size[1]
            actual = model.body(f'thread_guide_{site}').pos.copy()
            taught = np.r_[model.site(site).pos[:2], actual[2]]
            station = Station(site, model.site(f'screw_socket_{site}').id, model.site(f'screw_axis_{site}').id,
                              model.geom(f'assembled_head_{site}').id, block,
                              model.equality(f'magnetic_bit_{site}').id, model.equality(f'thread_capture_{site}').id,
                              model.equality(f'keyed_drive_{site}').id, turn, dof, float(model.dof_damping[dof]),
                              nominal, taught, actual, flank_pairs if site == fault_site else frozenset(),
                              model.site(f'thread_entry_{site}').id,
                              int(model.joint(f'thread_depth_{site}').dofadr[0]),
                              model.equality(f'thread_pitch_{site}').id)
            self.stations[site] = station
            self.allowed_pairs.update(frozenset((station.head, geom)) for geom in (
                self.bit, self.plate, block, model.geom(f'feeder_seat_{site}').id, model.geom(f'feeder_base_{site}').id,
            ))
            pilot = model.geom(f'screw_pilot_{site}').id
            self.allowed_pairs.update(frozenset((pilot, geom)) for geom in range(model.ngeom)
                                      if model.geom(geom).name.startswith('lead_in_'))
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
        self.force_zero = np.zeros(3)
        self.attempt = 0
        self.angle_origin = None
        self.results = {site: dict(status='pending', after_retract=None, torque_cutoff=None, peak_torque_nm=0.0,
                                   peak_seat_force_n=0.0, thread_diagnosis=None, attempts=[]) for site in self.sites}
        for station in self.stations.values():
            self.model.dof_damping[station.turn_dof] = station.free_damping
            self.model.eq_data[station.key, 0] = 0
            self.model.site_quat[station.thread_entry] = (1, 0, 0, 0)
        self.reset(self.sites, self.image_fault, self.image_site, self.seed)
        if self.strip_depth_mm is not None:
            damaged = self.stations[self.mechanical_fault_site]
            self.data.eq_active[damaged.pitch] = False
        if self.jam_mocap is not None:
            self.data.mocap_pos[self.jam_mocap, 2] = -0.040
            mujoco.mj_forward(self.model, self.data)

    def step(self):
        if self.strip_depth_mm is not None:
            station = self.stations[self.mechanical_fault_site]
            axial, spin = 0.0, 0.0
            if self.data.eq_active[station.guide]:
                depth_joint = self.model.dof_jntid[station.depth_dof]
                axial, spin = stripped_thread_forces(
                    float(self.data.joint(station.turn_joint).qpos[0]),
                    float(self.data.joint(depth_joint).qpos[0]),
                    float(self.data.qvel[station.turn_dof]), float(self.data.qvel[station.depth_dof]),
                    self.strip_depth_mm / 1000,
                )
            self.data.qfrc_applied[station.depth_dof] = axial
            self.data.qfrc_applied[station.turn_dof] = spin
        super().step()

    def check_contacts(self):
        for contact in self.data.contact:
            pair = frozenset((contact.geom1, contact.geom2))
            if pair not in self.allowed_pairs:
                self.reject_contact(contact)

    def normal_force(self, station, *surfaces):
        pairs = {frozenset((station.head, geom)) for geom in surfaces}
        return self.contact_load(pairs)

    def contact_load(self, pairs):
        if not pairs:
            return 0.0
        total = 0.0
        force = np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            if frozenset((contact.geom1, contact.geom2)) in pairs:
                mujoco.mj_contactForce(self.model, self.data, i, force)
                total += float(force[0])
        return total

    def load_vector(self, sensor):
        axes = self.data.site_xmat[self.load_cell].reshape(3, 3)
        return axes @ self.data.sensor(sensor).data

    def tool_force(self):
        return self.load_vector(self.force_sensor) - self.force_zero

    def snapshot(self, station=None):
        station = self.station if station is None else station
        data = self.data
        position = data.site_xpos[station.screw]
        tip = data.site_xpos[self.tip]
        normal = data.site_xmat[station.screw].reshape(3, 3)[:, 2]
        depth = station.entry_actual[2] - position[2]
        force = self.tool_force()
        moment = self.load_vector(self.torque_sensor)
        tool_axis = data.site_xmat[self.reference].reshape(3, 3)[:, 2]
        encoder_depth = 1000 * (station.entry_nominal[2] + self.model.site(station.socket).pos[2] - tip[2])
        spindle_angle = float(data.joint(self.spindle_joint).qpos[0])
        drive_turns = ((spindle_angle - self.angle_origin) / math.tau
                       if self.angle_origin is not None and station is self.station else None)
        return dict(sim_time_s=float(data.time), stage=self.stage, site=station.name,
                    attempt=self.attempt, tool_fx_n=float(force[0]), tool_fy_n=float(force[1]),
                    tool_fz_n=float(force[2]), tool_lateral_n=float(np.linalg.norm(force[:2])),
                    tool_mx_nm=float(moment[0]), tool_my_nm=float(moment[1]), tool_mz_nm=float(moment[2]),
                    tool_tilt_deg=math.degrees(math.atan2(np.linalg.norm(tool_axis[:2]), tool_axis[2])),
                    flank_force_n=self.contact_load(station.flank_pairs),
                    screw_x_m=float(position[0]), screw_y_m=float(position[1]), screw_z_m=float(position[2]),
                    tip_x_m=float(tip[0]), tip_y_m=float(tip[1]), tip_z_m=float(tip[2]),
                    radial_error_mm=float(1000 * np.linalg.norm(position[:2] - station.entry_actual[:2])),
                    tilt_deg=math.degrees(math.acos(float(np.clip(normal[2], -1, 1)))),
                    insertion_depth_mm=float(1000 * depth),
                    encoder_depth_mm=float(encoder_depth), drive_turns=drive_turns,
                    thread_turns=float(data.joint(station.turn_joint).qpos[0] / (2 * math.pi)),
                    spindle_velocity_command_rad_s=float(data.ctrl[self.motor]),
                    spindle_angle_rad=spindle_angle,
                    spindle_velocity_rad_s=float(data.joint(self.spindle_joint).qvel[0]),
                    spindle_torque_nm=float(abs(data.actuator_force[self.motor])),
                    seat_force_n=self.normal_force(station, self.plate),
                    obstruction_force_n=self.normal_force(station, station.block),
                    bit_contact_n=self.normal_force(station, self.bit),
                    magnet=bool(data.eq_active[station.magnet]), thread=bool(data.eq_active[station.guide]),
                    max_arm_velocity_rad_s=float(np.max(np.abs(data.qvel[:6]))),
                    **{f'q{index}_rad': float(angle) for index, angle in enumerate(data.qpos[:6], start=1)})

    def event(self, name, **fields):
        mujoco.mj_forward(self.model, self.data)
        self.events.append(dict(event=name, **self.snapshot(), **fields))

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

    def move_xyz(self, stage, xyz, speed=0.55, rotation=None):
        self.stage = stage
        goal = self.solve_pose(self.reference, xyz, np.eye(3) if rotation is None else rotation, self.data.qpos[:6])
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
            except AssemblyReview:
                result['status'] = 'review'
                raise
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
        self.attempt = 0
        self.angle_origin = None
        self.force_zero[:] = 0
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
        if station.name == self.entry_offset_site:
            approach[:2] += np.asarray(self.entry_offset_mm) / 1000
        if self.scenario == 'misaligned' and station.name == self.mechanical_fault_site:
            approach[0] += 0.003
        self.move_xyz('transfer', approach + (0, 0, 0.090))
        self.move_xyz('align', approach + (0, 0, 0.016), speed=0.34)
        start_pose = self.snapshot()
        self.event('axis_alignment')
        if self.insertion == 'force':
            approach = self.insert_and_tighten(station, approach)
        else:
            approach = self.fixed_entry(station, approach, start_pose)
            self.engage_thread(station)
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

    def fixed_entry(self, station, approach, start_pose):
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
        # The arm can settle before the held screw finishes bouncing off the chamfer.
        self.step_for(0.12)
        return approach

    def force_entry(self, approach, origin):
        point = approach + (0, 0, 0.0026)
        # Allow for the loaded position servo's axial sag relative to the reference.
        probe_floor = approach[2] - 0.00065
        self.move_xyz('probe', point, speed=0.20)
        self.step_for(0.10)
        self.force_zero = self.load_vector(self.force_sensor)
        self.event('load_tared')
        quiet = 0
        for _ in range(85):
            samples = []
            for _ in range(round(CONTACT_DT_S / self.model.opt.timestep)):
                self.step()
                samples.append(self.tool_force())
            force = np.mean(samples, axis=0)
            if np.linalg.norm(force) > MAX_PROBE_FORCE_N:
                raise AssemblyStopped('寻孔触力超过上限，未启动电批', 'probe_load')
            lateral = np.linalg.norm(force[:2])
            if lateral <= LATERAL_DEADBAND_N:
                if point[2] <= probe_floor + 1e-7:
                    quiet += 1
                    if quiet >= 4:
                        self.event('probe_complete')
                        return np.r_[point[:2], approach[2]]
                    continue
            else:
                quiet = 0
                correction = correction_step(force[:2])
                point[:2] += correction
                if np.linalg.norm(point[:2] - origin) > MAX_SEARCH_M:
                    raise AssemblyStopped('寻孔微调超过行程，未启动电批', 'probe_travel')
                self.event('force_correction', correction_mm=(1000 * correction).tolist(),
                           averaged_lateral_n=float(lateral))
            if lateral < 0.45:
                descent = 0.00022 if lateral <= LATERAL_DEADBAND_N else 0.00009
                point[2] = max(probe_floor, point[2] - descent)
            self.data.ctrl[:6] = self.solve_pose(self.reference, point, np.eye(3), self.data.ctrl[:6])
        raise AssemblyStopped('寻孔超过时限，未启动电批', 'probe_timeout')

    def insert_and_tighten(self, station, goal):
        approach = goal
        angled_injection = self.entry_tilt_deg is not None and station.name == self.mechanical_fault_site
        rotation = initial_axis(self.entry_tilt_deg) if angled_injection else np.eye(3)
        result = self.results[station.name]
        attempts = result['attempts']
        for self.attempt in range(1, MAX_ATTEMPTS + 1):
            self.angle_origin = None
            attempt = dict(number=self.attempt, status='probing')
            attempts.append(attempt)
            self.event('insertion_attempt')
            try:
                approach = self.force_entry(approach, goal[:2])
                if not np.allclose(rotation, np.eye(3)):
                    self.move_xyz('angled_entry', approach + (0, 0, 0.0001), speed=0.16, rotation=rotation)
                    self.step_for(0.12)
                    self.event('angled_entry')
                self.engage_thread(station)
            except AssemblyStopped as exc:
                attempt.update(status='rejected', reason=str(exc))
                # Lift vertically from the current XY point without dragging the screw.
                retreat = self.data.site_xpos[self.reference].copy()
                retreat[2] = goal[2] + 0.016
                self.move_xyz('retry_retract', retreat, speed=0.24)
                self.event('probe_retracted')
                if self.attempt > 1:
                    result['thread_diagnosis'] = 'retry_entry_unverified'
                    raise AssemblyReview('退牙后重新找孔失败，等待复检', 'thread_retry_entry') from exc
                raise
            attempt['status'] = 'driving'
            try:
                self.tighten(station)
            except InsertionAnomaly as exc:
                attempt.update(status=exc.attempt_status, reason=str(exc))
                result['thread_diagnosis'] = exc.review_code
                if self.recovery == 'stop':
                    raise
                try:
                    self.recover_thread(station, approach, rotation)
                except AssemblyStopped as unwind:
                    raise AssemblyReview(f'旋入异常，退牙未完成：{unwind}', unwind.code) from unwind
                if self.attempt == MAX_ATTEMPTS:
                    raise AssemblyReview(f'复试仍异常，已停转退回，等待复检。{exc}', exc.review_code) from exc
                if self.recovery == 'realign':
                    rotation = self.realign_after_jam(approach, rotation)
                if self.jam_mocap is not None:
                    self.clear_simulated_chip(station)
            except AssemblyStopped as exc:
                if self.attempt > 1:
                    attempt.update(status='unverified', reason=str(exc))
                    result['thread_diagnosis'] = 'thread_signature_unverified'
                    raise AssemblyReview(f'复试触发装配保护，等待复检。{exc}', exc.code) from exc
                raise
            else:
                attempt['status'] = 'seated'
                result['thread_diagnosis'] = 'recovered_after_retry' if self.attempt > 1 else 'normal'
                return approach

    def engage_thread(self, station):
        at_entry = self.snapshot()
        tilted_bite = bool(station.flank_pairs) and at_entry['tilt_deg'] > ENGAGE_TILT_DEG
        tolerance = ENGAGE_XY_M if tilted_bite or self.insertion == 'fixed' else FORCE_ENTRY_TOLERANCE_M
        tilt_limit = MAX_ANGLED_ENTRY_DEG if tilted_bite else ENGAGE_TILT_DEG
        if (abs(at_entry['insertion_depth_mm']) > 0.85 or at_entry['radial_error_mm'] > 1000 * tolerance
                or at_entry['tilt_deg'] > tilt_limit):
            raise AssemblyStopped('螺丝没有对准螺纹入口', 'thread_entry')
        if tilted_bite:
            axis = self.data.site_xmat[station.screw].reshape(3, 3)[:, 2]
            wrist_tilt = rotation_without_yaw(axis)
            # The reduced thread joint retains the incoming bite angle until reverse clears it.
            mujoco.mju_mat2Quat(self.model.site_quat[station.thread_entry], wrist_tilt.ravel())
        else:
            self.model.site_quat[station.thread_entry] = (1, 0, 0, 0)
        self.model.eq_data[station.key, 0] = (self.data.joint(self.spindle_joint).qpos[0]
                                              - self.data.joint(station.turn_joint).qpos[0])
        self.data.eq_active[station.guide] = True
        self.data.eq_active[station.key] = True
        self.angle_origin = float(self.data.joint(self.spindle_joint).qpos[0])
        self.event('thread_engaged')
        if self.jam_mocap is not None and station.name == self.mechanical_fault_site and self.attempt <= 1:
            self.data.mocap_pos[self.jam_mocap, 2] = 0.0
            self.event('fault_fixture_extended')

    def recover_thread(self, station, approach, rotation):
        self.stage = 'jam_brake'
        self.brake_spindle()
        self.event('jam_spindle_stopped')
        self.stage = 'unwind'
        self.model.dof_damping[station.turn_dof] = station.free_damping
        # Spindle and screw are keyed while reversing. Remove the thread constraint only
        # after the encoder returns to its entry phase and motion has stopped.
        entry_angle = float(self.model.eq_data[station.key, 0])
        encoder = self.data.joint(self.spindle_joint).qpos
        self.event('unwind_started')
        reverse_measured = False
        try:
            for _ in range(round(6 / self.model.opt.timestep)):
                angle = float(encoder[0]) - entry_angle
                if abs(angle) <= 0.07:
                    break
                self.data.ctrl[self.motor] = -np.sign(angle) * min(5.5, 1.2 + 1.6 * abs(angle))
                self.step()
                if not reverse_measured and self.data.joint(self.spindle_joint).qvel[0] < -0.12:
                    self.event('unwind_reverse_measured')
                    reverse_measured = True
            else:
                raise AssemblyStopped('退牙编码器未在时限内回到入口', 'unwind_timeout')
        finally:
            self.data.ctrl[self.motor] = 0.0
        self.brake_spindle()
        if abs(station.entry_actual[2] - self.data.site_xpos[station.screw, 2]) > 0.00045:
            raise AssemblyStopped('退牙后的入口深度超限', 'unwind_depth')
        self.event('unwind_complete')
        self.data.eq_active[station.key] = False
        self.data.eq_active[station.guide] = False
        self.move_xyz('retry_retract', approach + (0, 0, 0.016), speed=0.24, rotation=rotation)
        tip_gap = np.linalg.norm(self.data.site_xpos[self.tip] - self.data.site_xpos[station.socket])
        lift = self.data.site_xpos[station.screw, 2] - station.entry_actual[2]
        if not self.data.eq_active[station.magnet] or tip_gap > 0.0014 or lift < 0.012:
            raise AssemblyStopped('退回时未保持螺丝', 'retry_retention')
        self.event('retry_retracted')

    def realign_after_jam(self, approach, rotation):
        axis = self.data.site_xmat[self.reference].reshape(3, 3)[:, 2]
        tilt = math.degrees(math.atan2(np.linalg.norm(axis[:2]), axis[2]))
        if tilt <= REALIGN_TILT_DEG:
            return rotation
        self.event('retry_axis_measured', measured_axis=axis.tolist())
        # The fixed work plane provides the target normal. Remove the measured pitch/roll
        # in free space, then let the existing force probe reacquire the hole.
        correction = rotation_without_yaw(axis).T
        corrected = correction @ rotation
        self.move_xyz('retry_realign', approach + (0, 0, 0.016), speed=0.18, rotation=corrected)
        self.step_for(0.12)
        self.event('retry_axis_corrected')
        return corrected if np.linalg.norm(corrected[:2, 2]) > math.sin(math.radians(REALIGN_TILT_DEG)) else np.eye(3)

    def clear_simulated_chip(self, station):
        # This is the removable obstruction fixture, independent of the stall detector.
        if self.jam_mocap is not None and station.name == self.mechanical_fault_site:
            self.data.mocap_pos[self.jam_mocap, 2] = -0.040
            self.event('fault_fixture_withdrawn')

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
        stalled = 0
        stall_begin_travel = 0.0
        stall_begin_angle = 0.0
        tip_at_entry = float(self.data.site_xpos[self.tip, 2])
        dt = self.model.opt.timestep
        monitor = SlipMonitor()
        monitor_stride = max(1, round(0.010 / dt))
        bearing_arm = BEARING_FRICTION_ARM_M * self.conditions[station.name]['bearing_scale']
        try:
            for step in range(round(8 / dt)):
                depth = station.entry_actual[2] - self.data.site_xpos[station.screw, 2]
                speed = 5.6 if depth >= 0.0054 else 13.0
                if station.flank_pairs:
                    speed = min(speed, 7.5)
                self.data.ctrl[self.motor] = min(speed, (step + 1) * dt * 50)
                rate = abs(self.data.qvel[station.turn_dof])
                # Regularized under-head friction from the measured normal load. Implicit damping
                # dissipates power even when the contact begins within a single integration step.
                bearing_force = self.normal_force(station, self.plate, station.block)
                drag = (bearing_arm * bearing_force
                        + FLANK_DRAG_ARM_M * self.contact_load(station.flank_pairs))
                self.model.dof_damping[station.turn_dof] = (station.free_damping
                    + drag / max(rate, 0.30))
                self.step()
                sample = self.snapshot()
                torque = sample['spindle_torque_nm']
                force = sample['seat_force_n']
                result['peak_torque_nm'] = max(result['peak_torque_nm'], torque)
                result['peak_seat_force_n'] = max(result['peak_seat_force_n'], force)
                depth = sample['insertion_depth_mm']
                rate = abs(float(self.data.qvel[station.turn_dof]))
                if self.insertion == 'force':
                    tip_travel = tip_at_entry - sample['tip_z_m']
                    angle = sample['spindle_angle_rad']
                    if step % monitor_stride == 0:
                        slip = monitor.observe(angle, sample['encoder_depth_mm'], torque)
                        if slip:
                            self.event('thread_slip_detected', **slip)
                            raise InsertionSlipped(
                                f"电批转过 {slip['window_turns']:.2f} 圈，深度仅增加 {slip['window_advance_mm']:.2f} mm",
                                'thread_slip')
                    loaded = (abs(sample['tool_fz_n']) >= 2.5 or sample['tool_lateral_n'] >= 1.7
                              or math.hypot(sample['tool_mx_nm'], sample['tool_my_nm']) >= 0.012)
                    early_stall = (tip_travel < DEPTH_M - 0.00075 and loaded
                                   and torque >= 0.155 and abs(sample['spindle_velocity_rad_s']) <= 0.8)
                    if (not early_stall or abs(tip_travel - stall_begin_travel) > 0.00012
                            or abs(angle - stall_begin_angle) > 0.12):
                        stalled = 0
                        stall_begin_travel = tip_travel
                        stall_begin_angle = angle
                    if early_stall:
                        stalled += 1
                    if stalled >= round(0.085 / dt):
                        self.event('jam_detected', sensed_torque_nm=torque, sensed_tip_travel_mm=1000 * tip_travel,
                                   window_turns=(angle - stall_begin_angle) / math.tau,
                                   window_advance_mm=1000 * (tip_travel - stall_begin_travel),
                                   jam_trigger='stall')
                        raise InsertionJammed(f'入座前扭矩升高且进给停滞，批头进给 {1000 * tip_travel:.2f} mm',
                                             'thread_jam')
                    if (station.flank_pairs and loaded and torque >= 0.155
                            and sample['tilt_deg'] > MAX_ANGLED_ENTRY_DEG + 0.75 and 0 < depth < 5.3):
                        self.event('jam_detected', sensed_torque_nm=torque, sensed_tip_travel_mm=1000 * tip_travel,
                                   jam_trigger='tilt_load')
                        raise InsertionJammed('旋入初段扭矩升高且轴线偏差超限', 'thread_jam')
                elif sample['obstruction_force_n'] > 3.0 and depth < DEPTH_M * 1000 - 0.7:
                    raise AssemblyStopped(f'未达到入座深度即触碰障碍，深度 {depth:.2f} mm，已停转', 'obstruction')
                tilt_limit = MAX_ANGLED_ENTRY_DEG + 0.75 if station.flank_pairs else 3.5
                if depth > 6.65 or sample['radial_error_mm'] > 0.8 or sample['tilt_deg'] > tilt_limit:
                    raise AssemblyStopped('旋入超出深度或轴线容差，已停转', 'thread_limits')
                pitch_error = abs(depth - PITCH_MM * sample['thread_turns'])
                seated = (abs(depth - DEPTH_M * 1000) <= 0.6 and pitch_error <= 0.25 and force >= 4.0
                          and sample['tilt_deg'] <= ENGAGE_TILT_DEG and rate <= 0.7
                          and TORQUE_TARGET_NM <= torque <= TORQUE_MAX_NM + 1e-6)
                if self.insertion == 'force':
                    encoder_depth = sample['encoder_depth_mm']
                    seated = (seated and abs(encoder_depth - DEPTH_M * 1000) <= 0.6
                              and abs(encoder_depth - PITCH_MM * sample['drive_turns']) <= 0.45)
                steady = steady + 1 if seated else 0
                if steady >= round(0.10 / dt):
                    result['torque_cutoff'] = {**sample, 'thread_rate_rad_s': rate}
                    self.event('torque_cutoff')
                    return
            final = self.snapshot()
            message = (f"旋紧未达到联合门限，深度 {final['encoder_depth_mm']:.2f} mm，"
                       f"扭矩 {final['spindle_torque_nm']:.3f} N·m，入座力 {final['seat_force_n']:.1f} N")
            if self.insertion == 'force':
                self.event('thread_signature_unverified')
                raise InsertionAnomaly(message, 'torque_gate')
            raise AssemblyStopped(message, 'torque_gate')
        finally:
            self.data.ctrl[self.motor] = 0.0
