import math
from dataclasses import dataclass

import cv2
import numpy as np


FIXTURE_ORIGIN = np.array([-0.49, 0.0])
HOLE_PLANE_Z_M = 0.07845
MAX_SHIFT_MM = 12.0
MAX_YAW_DEG = 6.0
MAX_FIT_RESIDUAL_MM = 0.32
ALIGNMENT_FAULTS = ('none', 'glare', 'occluded')


@dataclass(frozen=True)
class FixtureShift:
    x_mm: float = 0.0
    y_mm: float = 0.0
    yaw_deg: float = 0.0

    def __post_init__(self):
        if not (math.hypot(self.x_mm, self.y_mm) <= MAX_SHIFT_MM + 1e-9 and abs(self.yaw_deg) <= MAX_YAW_DEG):
            raise ValueError(f'夹具仿真偏移上限 {MAX_SHIFT_MM:g} mm、转角上限 {MAX_YAW_DEG:g}°')

    def apply(self, xy):
        theta = math.radians(self.yaw_deg)
        c, s = math.cos(theta), math.sin(theta)
        rotation = np.array([[c, -s], [s, c]])
        return (np.asarray(xy) - FIXTURE_ORIGIN) @ rotation.T + FIXTURE_ORIGIN + np.array([self.x_mm, self.y_mm]) / 1000


@dataclass(frozen=True)
class Pinhole:
    position: np.ndarray
    rotation: np.ndarray
    width: int
    height: int
    fovy_deg: float

    @property
    def focal(self):
        return self.height / (2 * math.tan(math.radians(self.fovy_deg) / 2))

    @property
    def center(self):
        return np.array([(self.width - 1) / 2, (self.height - 1) / 2])

    def project(self, points):
        camera_points = (np.asarray(points) - self.position) @ self.rotation
        if np.any(camera_points[..., 2] >= 0):
            raise ValueError('工件在相机视锥后方')
        return (self.focal * camera_points[..., :2] * (1, -1)
                / (-camera_points[..., 2:3])) + self.center

    def intersect_plane(self, pixels, z=HOLE_PLANE_Z_M):
        rays = np.atleast_2d((np.asarray(pixels) - self.center) * (1, -1) / self.focal)
        directions = np.column_stack((rays, -np.ones(len(rays)))) @ self.rotation.T
        if np.any(directions[:, 2] >= -0.1) or self.position[2] <= z:
            raise ValueError('相机视线与工件平面的投影不可靠')
        return self.position + ((z - self.position[2]) / directions[:, 2, None]) * directions


@dataclass(frozen=True)
class HoleObservation:
    uv: tuple[float, float]
    xy_m: tuple[float, float]
    radius_mm: float
    roundness: float
    pixel_area: float
    contrast: float


def find_hole(image, camera, taught_xy):
    expected = camera.project([*taught_xy, HOLE_PLANE_Z_M])
    u, v = np.round(expected).astype(int)
    margin = 112
    x0, x1 = max(u - margin, 0), min(u + margin + 1, camera.width)
    y0, y1 = max(v - margin, 0), min(v + margin + 1, camera.height)
    if min(x1 - x0, y1 - y0) < 44:
        return None, '示教孔位在有效成像范围外', tuple(map(float, expected))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop = cv2.GaussianBlur(gray[y0:y1, x0:x1], (3, 3), 0)
    background = float(np.median(crop))
    mask = (crop < background - 32).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        pixels = contour[:, 0, :]
        pixel_area = cv2.contourArea(contour)
        if (len(pixels) < 20 or pixel_area < 550 or np.any(pixels.min(axis=0) <= 1)
                or np.any(pixels.max(axis=0) >= np.array(crop.shape[::-1]) - 2)):
            continue
        plane = camera.intersect_plane(pixels + (x0, y0))[:, :2]
        local_mm = ((plane - taught_xy) * 1000).astype(np.float32)
        (cx, cy), (a, b), _ = cv2.fitEllipse(local_mm)
        radius = (a + b) / 4
        roundness = min(a, b) / max(a, b) if max(a, b) else 0
        integral = cv2.contourArea(local_mm) / (math.pi * radius**2) if radius else 0
        if not (6.4 <= radius <= 8.2 and roundness >= 0.88 and 0.86 <= integral <= 1.12):
            continue
        candidate_xy = np.asarray(taught_xy) + np.array([cx, cy]) / 1000
        candidate_uv = camera.project([*candidate_xy, HOLE_PLANE_Z_M])
        center_u, center_v = np.round(candidate_uv).astype(int)
        if not (x0 <= center_u < x1 and y0 <= center_v < y1):
            continue
        contrast = background - float(gray[center_v, center_u])
        if contrast >= 45:
            candidates.append(HoleObservation(tuple(map(float, candidate_uv)), tuple(map(float, candidate_xy)),
                                              float(radius), float(roundness), float(pixel_area), contrast))
    if len(candidates) != 1:
        reason = '未找到可信的孔缘' if not candidates else '同一搜索区域内存在多个可信孔缘'
        return None, reason, tuple(map(float, expected))
    return candidates[0], '孔径、圆度和对比度校验通过', tuple(map(float, expected))


def fit_fixture(taught, measured):
    taught, measured = np.asarray(taught), np.asarray(measured)
    if taught.shape != (3, 2) or measured.shape != taught.shape or not np.isfinite([taught, measured]).all():
        raise ValueError('夹具定位需要三个有效孔中心')
    centered = taught - taught.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError('标定孔位几何退化')
    u, _, vt = np.linalg.svd(centered.T @ (measured - measured.mean(axis=0)))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        raise ValueError('孔位几何与示教布局不一致')
    offset = measured.mean(axis=0) - (rotation @ (taught.mean(axis=0) - FIXTURE_ORIGIN) + FIXTURE_ORIGIN)
    shift = FixtureShift(float(1000 * offset[0]), float(1000 * offset[1]),
                         math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])))
    residual = np.linalg.norm(shift.apply(taught) - measured, axis=1) * 1000
    if residual.max() > MAX_FIT_RESIDUAL_MM:
        raise ValueError(f'三孔定位拟合残差 {residual.max():.2f} mm 超限')
    return shift, residual.tolist()


def mask_observation(image, expected, fault):
    if fault == 'none':
        return image
    x, y = (int(round(value)) for value in expected)
    masked = image.copy()
    fill = (255, 255, 255) if fault == 'glare' else (149, 149, 149)
    cv2.rectangle(masked, (x - 105, y - 105), (x + 105, y + 105), fill, thickness=cv2.FILLED)
    return masked


def marked_photo(image, observation, expected):
    marked = image.copy()
    taught_uv = tuple(round(point) for point in expected)
    cv2.drawMarker(marked, taught_uv, (30, 199, 252), cv2.MARKER_CROSS, 24, 2)
    if observation is not None:
        found = tuple(round(point) for point in observation.uv)
        cv2.line(marked, taught_uv, found, (252, 229, 108), 2, cv2.LINE_AA)
        radius = round(math.sqrt(observation.pixel_area / math.pi)) + 8
        cv2.circle(marked, found, radius, (125, 238, 54), 2, cv2.LINE_AA)
        cv2.drawMarker(marked, found, (125, 238, 54), cv2.MARKER_CROSS, 20, 2)
    return marked
