import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


SITES = ('S1', 'S2', 'S3')
ANGLES = ('primary', 'alternate')
PRESENT, MISSING, REVIEW = 'present', 'missing', 'review'
LABELS = {PRESENT: '有螺丝', MISSING: '缺螺丝', REVIEW: '待复检'}
cv2.ocl.setUseOpenCL(False)


def load_image(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法读取图像 {path}')
    return image


def save_image(path, image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise ValueError(f'无法编码图像 {path}')
    path.write_bytes(data.tobytes())


@dataclass
class Observation:
    state: str
    reason: str
    blur: float | None = None
    glare: float | None = None
    dark: float | None = None
    present_score: float | None = None
    missing_score: float | None = None

    def record(self):
        return asdict(self)


class View:
    def __init__(self, spec, base, frame_size):
        self.frame_size = frame_size
        self.x, self.y, self.width, self.height = spec['roi']
        self.margin = spec['shift_px']
        frame_width, frame_height = frame_size
        if not all(type(value) is int for value in (*spec['roi'], self.margin)):
            raise ValueError('ROI 和偏移容差必须为整数像素')
        if not (min(self.width, self.height) >= 24 and 0 <= self.margin <= min(self.width, self.height) // 4
                and self.x >= self.margin and self.y >= self.margin
                and self.x + self.width + self.margin <= frame_width
                and self.y + self.height + self.margin <= frame_height):
            raise ValueError(f'ROI 超出图像范围或过小 {spec["roi"]}')
        self.min_sharpness = spec['min_sharpness']
        self.max_glare = spec['max_glare']
        self.max_dark = spec['max_dark']
        self.min_score = spec['min_score']
        self.min_margin = spec['min_margin']
        thresholds = (self.max_glare, self.max_dark, self.min_score, self.min_margin)
        if not (all(0 < value < 1 for value in thresholds)
                and math.isfinite(self.min_sharpness) and self.min_sharpness > 0):
            raise ValueError('检测阈值无效')
        self.references = {}
        for state in (PRESENT, MISSING):
            if not isinstance(spec[state], list) or not spec[state]:
                raise ValueError(f'缺少参考图 {state}')
            self.references[state] = []
            for name in spec[state]:
                ref = load_image(base / name)
                if ref.shape[:2][::-1] != frame_size:
                    raise ValueError(f'参考图分辨率错误 {name}')
                crop = self.crop(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
                _, _, _, reasons = self.quality(crop)
                if reasons or crop.std() < 2:
                    raise ValueError(f'参考 ROI 质量不足 {name} {reasons}')
                self.references[state].append(crop.copy())

    def crop(self, gray, margin=0):
        return gray[self.y - margin:self.y + self.height + margin,
                    self.x - margin:self.x + self.width + margin]

    def quality(self, crop):
        glare = float(np.mean(crop >= 248))
        dark = float(np.mean(crop <= 8))
        blur = float(cv2.Laplacian(crop, cv2.CV_32F).var())
        reasons = []
        if glare >= self.max_glare:
            reasons.append('反光')
        if blur < self.min_sharpness:
            reasons.append('模糊')
        if dark >= self.max_dark:
            reasons.append('欠曝')
        return blur, glare, dark, reasons

    def inspect(self, image):
        if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2][::-1] != self.frame_size:
            return Observation(REVIEW, '图像分辨率或格式不符')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur, glare, dark, reasons = self.quality(self.crop(gray))
        if reasons:
            return Observation(REVIEW, '、'.join(reasons), blur, glare, dark)
        search = self.crop(gray, self.margin)
        scores = {
            state: max(float(cv2.matchTemplate(search, ref, cv2.TM_CCOEFF_NORMED).max())
                       for ref in refs)
            for state, refs in self.references.items()
        }
        state = max(scores, key=scores.get)
        winner, runner_up = max(scores.values()), min(scores.values())
        accepted = winner >= self.min_score and winner - runner_up >= self.min_margin
        return Observation(state if accepted else REVIEW, '模板匹配' if accepted else '模板匹配不确定',
                           blur, glare, dark, scores[PRESENT], scores[MISSING])


class Detector:
    def __init__(self, profile):
        profile = Path(profile)
        config = json.loads(profile.read_text(encoding='utf-8-sig'))
        self.frame_size = tuple(config['frame_size'])
        if (config['version'] != 1 or config['source'] not in ('simulation', 'hardware')
                or len(self.frame_size) != 2 or any(type(n) is not int or n <= 0 for n in self.frame_size)):
            raise ValueError('检测配置版本、来源或图像分辨率无效')
        self.source = config['source']
        self.views = {}
        if set(config['views']) != set(SITES):
            raise ValueError('检测配置必须恰好包含 S1、S2、S3')
        for site in SITES:
            entries = config['views'][site]
            if set(entries) != set(ANGLES):
                raise ValueError(f'{site} 必须包含主视角与补拍视角')
            for angle in ANGLES:
                self.views[site, angle] = View(entries[angle], profile.parent, self.frame_size)

    def inspect(self, site, angle, image):
        return self.views[site, angle].inspect(image)
