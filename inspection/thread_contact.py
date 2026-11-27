import math
import xml.etree.ElementTree as ET

import numpy as np


THREAD_PITCH_MM = 2.0
MAX_ANGLED_ENTRY_DEG = 6.0
REALIGN_TILT_DEG = 0.5
RECOVERY_MODES = ('stop', 'retry', 'realign')
FLANK_DRAG_ARM_M = 0.005


def rotation_without_yaw(normal):
    """Shortest rotation from world Z to the measured axis."""
    x, y, z = normal
    skew = np.array([[0, 0, x], [0, 0, y], [-x, -y, 0]])
    return np.eye(3) + skew + (skew @ skew) / (1 + z)


def initial_axis(tilt_deg):
    x, y = np.radians(tilt_deg)
    return rotation_without_yaw(np.array([math.cos(x) * math.sin(y), -math.sin(x), math.cos(x) * math.cos(y)]))


def tooth_sectors(assets, parent, prefix, profile, first_z, turns, xy, *, male):
    """Convex flank segments follow the same helix as the pitch constraint."""
    pitch = THREAD_PITCH_MM / 1000
    sectors = 12
    for turn in range(turns):
        for sector in range(sectors):
            start = math.tau * sector / sectors
            end = math.tau * (sector + 1) / sectors
            vertices = [
                (radius * math.cos(angle), radius * math.sin(angle), offset + pitch * angle / math.tau)
                for angle in (start, end) for radius, offset in profile
            ]
            name = f'{prefix}_{sector}'
            if turn == 0:
                ET.SubElement(assets, 'mesh', name=name, vertex=' '.join(
                    f'{value:.9f}' for vertex in vertices for value in vertex))
            ET.SubElement(parent, 'geom', name=f'{name}_{turn}', type='mesh', mesh=name,
                          pos=f'{xy[0]:.9f} {xy[1]:.9f} {first_z + turn * pitch:.9f}',
                          contype='16' if male else '0', conaffinity='0' if male else '16', condim='4',
                          friction='0.55 0.002 0.0001', solref='0.004 1', solimp='0.98 0.999 0.00009',
                          rgba='0 0 0 0', group='3')


def add_thread_flanks(assets, world, screw, xy, entry_z):
    # The roots are wide enough to preserve flank contact under a tilted start.
    male = ((.00372, -.00078), (.00461, -.00026), (.00461, .00026), (.00372, .00078))
    female = ((.00548, -.00078), (.00420, -.00028), (.00420, .00028), (.00548, .00078))
    male_start = -.01265
    tooth_sectors(assets, screw, 'male_flank', male, male_start, 2, (0, 0), male=True)
    tooth_sectors(assets, world, 'female_flank', female, entry_z + male_start - .011, 4, xy, male=False)
