import math
import xml.etree.ElementTree as ET

import numpy as np


INSERTION_MODES = ('fixed', 'force')
OBSTRUCTION_FAULTS = ('blocked', 'retryable_jam')
MAX_SEARCH_M = 0.0018
FORCE_ENTRY_TOLERANCE_M = 0.00035
MAX_ATTEMPTS = 2
CONTACT_DT_S = 0.04
LATERAL_DEADBAND_N = 0.10
MAX_PROBE_FORCE_N = 16.0


def add_chamfer_meshes(assets):
    """Convex sectors form a circular lead-in; the center is open."""
    count = 12
    for sector in range(count):
        a, b = (index * math.tau / count for index in (sector, sector + 1))
        vertices = [
            (radius * math.cos(angle), radius * math.sin(angle), height)
            for height, inner in ((0.073, 0.00402), (0.0768, 0.00560))
            for radius in (inner, 0.0075) for angle in (a, b)
        ]
        ET.SubElement(assets, 'mesh', name=f'lead_in_{sector}', vertex=' '.join(
            f'{value:.9f}' for vertex in vertices for value in vertex))
    return count


def add_chamfer(world, site, xy, count):
    for sector in range(count):
        ET.SubElement(world, 'geom', name=f'lead_in_{site}_{sector}', type='mesh', mesh=f'lead_in_{sector}',
                      pos=f'{xy[0]:.9f} {xy[1]:.9f} 0', contype='0', conaffinity='8',
                      condim='1', solref='0.005 1', rgba='0 0 0 0', group='3')


def correction_step(lateral_force):
    load = np.linalg.norm(lateral_force)
    if load <= LATERAL_DEADBAND_N:
        return np.zeros(2)
    # Minimum useful displacement clears the IK solver's 10 µm convergence tolerance.
    distance = min(0.00018, 0.000015 + 0.00040 * (load - LATERAL_DEADBAND_N))
    return -lateral_force * (distance / load)
