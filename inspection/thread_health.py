import math
from collections import deque

from .thread_contact import THREAD_PITCH_MM


DAMAGE_FAULTS = ('damaged_thread', 'stripped_thread')
PITCH_M_PER_RAD = THREAD_PITCH_MM / (1000 * math.tau)
SLIP_WINDOW_TURNS = 0.36
SLIP_MAX_TORQUE_NM = 0.10
SLIP_MIN_DEFICIT_MM = 0.48


def stripped_thread_forces(angle, depth, angle_rate, depth_rate, strip_depth_m):
    """Passive axial transmission with a permanently stripped interval.

    Beyond the last load-bearing turns, surviving flanks provide only a small
    fraction of the nominal lead. The opposing joint forces share this ratio.
    """
    nominal = angle * PITCH_M_PER_RAD
    slope = PITCH_M_PER_RAD
    if nominal > strip_depth_m:
        slope *= 0.045
        nominal = strip_depth_m + (nominal - strip_depth_m) * 0.045
    axial = 48000 * (nominal - depth) + 490 * (slope * angle_rate - depth_rate)
    return axial, -slope * axial


class SlipMonitor:
    """Detect unproductive angular travel from a spindle and a depth encoder."""

    def __init__(self):
        self.window = deque()

    def observe(self, angle_rad, depth_mm, torque_nm):
        self.window.append((angle_rad, depth_mm, torque_nm))
        span = math.tau * SLIP_WINDOW_TURNS
        while len(self.window) > 2 and angle_rad - self.window[1][0] >= span:
            self.window.popleft()
        angle_start, depth_start, _ = self.window[0]
        turns = (angle_rad - angle_start) / math.tau
        if turns < SLIP_WINDOW_TURNS:
            return None
        expected = turns * THREAD_PITCH_MM
        advance = depth_mm - depth_start
        torque = sum(point[2] for point in self.window) / len(self.window)
        if torque > SLIP_MAX_TORQUE_NM or expected - advance < SLIP_MIN_DEFICIT_MM or advance > 0.32 * expected:
            return None
        return dict(window_turns=turns, window_advance_mm=advance, window_expected_mm=expected,
                    window_mean_torque_nm=torque, window_pitch_deficit_mm=expected - advance)
