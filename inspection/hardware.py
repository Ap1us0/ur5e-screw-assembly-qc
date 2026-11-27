import hashlib
import ipaddress
import json
import math
import re
import socket
import threading
import time
from pathlib import Path

import cv2

from .detector import ANGLES, SITES
from .workflow import CameraError


POSE_CODES = {(site, angle): number for number, (site, angle) in enumerate(
    ((site, angle) for site in SITES for angle in ANGLES), start=1)}


class StationConfig:
    def __init__(self, path):
        self.config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        self.pc_ip = str(ipaddress.IPv4Address(self.config['pc_ip']))
        self.robot_ip = str(ipaddress.IPv4Address(self.config['robot_ip']))
        self.route_id = self.config['route_id']
        self.port = self.config['port']
        camera = self.config['camera']
        self.camera_index = camera['index']
        self.frame_size = tuple(camera['frame_size'])
        if (self.config['version'] != 1 or not re.fullmatch(r'[A-Za-z0-9_-]{4,32}', self.route_id)
                or type(self.port) is not int or not 1024 <= self.port <= 65535
                or type(self.camera_index) is not int or self.camera_index < 0
                or len(self.frame_size) != 2 or any(type(n) is not int or n <= 0 for n in self.frame_size)):
            raise ValueError('工站配置中的版本、编号、端口或相机参数无效')
        self.connect_timeout = self.config['connect_timeout_s']
        self.move_timeout = self.config['move_timeout_s']
        if not (1 <= self.connect_timeout <= 120 and 2 <= self.move_timeout <= 180):
            raise ValueError('工站通信超时参数无效')
        self.home = self.joints(self.config['home'])
        if set(self.config['routes']) != set(SITES):
            raise ValueError('工站须配置三个螺丝位')
        self.routes = {}
        for (site, angle), code in POSE_CODES.items():
            route = self.config['routes'][site][angle]
            if not isinstance(route, list) or not route:
                raise ValueError(f'{site}/{angle} 需要至少一个实机示教拍摄位')
            self.routes[code] = [self.joints(point) for point in route]
        goals = {self.home, *(route[-1] for route in self.routes.values())}
        if len(goals) != len(POSE_CODES) + 1:
            raise ValueError('待机位与六个拍摄位必须各不相同')
        self.speed = self.config['speed_rad_s']
        self.acceleration = self.config['acceleration_rad_s2']
        if not all(math.isfinite(value) and value > 0 for value in (self.speed, self.acceleration)):
            raise ValueError('工站速度和加速度必须为正数')
        route_bytes = json.dumps([self.home, self.routes, self.speed, self.acceleration],
                                 separators=(',', ':')).encode('ascii')
        self.route_token = f'{self.route_id}_{hashlib.sha256(route_bytes).hexdigest()[:12]}'

    @staticmethod
    def joints(values):
        if len(values) != 6 or any(type(n) not in (float, int) or not math.isfinite(n)
                                   or abs(n) > 2 * math.pi for n in values):
            raise ValueError('实机示教关节位必须含六个弧度值')
        return tuple(values)


class StationLink:
    def __init__(self, config):
        self.connection = None
        self.buffer = bytearray()
        self.config = config

    def connect(self):
        deadline = time.monotonic() + self.config.connect_timeout
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            # Windows defaults to an exclusive listener unless SO_REUSEADDR is enabled.
            listener.bind((self.config.pc_ip, self.config.port))
            listener.listen(1)
            while self.connection is None:
                listener.settimeout(max(0.001, deadline - time.monotonic()))
                connection, address = listener.accept()
                if address[0] == self.config.robot_ip:
                    self.connection = connection
                else:
                    connection.close()
                if time.monotonic() >= deadline:
                    raise TimeoutError('未连接到指定 UR5e')
            self.expect(f'READY {self.config.route_token}', deadline)

    def expect(self, expected, deadline):
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'UR5e 应答超时，等待 {expected}')
            self.connection.settimeout(remaining)
            try:
                part = self.connection.recv(64)
            except TimeoutError as exc:
                raise TimeoutError(f'UR5e 应答超时，等待 {expected}') from exc
            if not part:
                raise ConnectionError(f'UR5e 已断开，等待 {expected}')
            self.buffer.extend(part)
            if len(self.buffer) > 128:
                raise ConnectionError('UR5e 应答过长')
        line, _, self.buffer = self.buffer.partition(b'\n')
        if line.rstrip(b'\r') != expected.encode('ascii') or self.buffer:
            raise ConnectionError(f'工站应答错误，期望 {expected}，收到 {bytes(line)!r}')

    def move(self, code):
        if code not in (0, *POSE_CODES.values()):
            raise ValueError(f'未知工站指令 {code}')
        deadline = time.monotonic() + self.config.move_timeout
        self.connection.settimeout(self.config.move_timeout)
        self.connection.sendall(f'({code})\n'.encode('ascii'))
        self.expect(f'AT {code}', deadline)

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


class FreshCamera:
    def __init__(self, index, frame_size, timeout=5.0):
        self.index = index
        self.frame_size = frame_size
        self.timeout = timeout
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.image = None
        self.sequence = 0
        self.failure = None

    def read(self):
        capture = None
        try:
            capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                raise CameraError(f'无法打开 UVC 相机 {self.index}')
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
            while not self.stop.is_set():
                ok, image = capture.read()
                if not ok or image is None:
                    self.stop.wait(0.05)
                    continue
                if image.shape[:2][::-1] != self.frame_size:
                    raise CameraError(f'相机返回 {image.shape[:2][::-1]}，配置为 {self.frame_size}')
                with self.condition:
                    self.image = image
                    self.sequence += 1
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.failure = str(exc)
                self.condition.notify_all()
        finally:
            if capture is not None:
                capture.release()

    def wait_frame(self, count):
        with self.condition:
            target = self.sequence + count
            ready = self.condition.wait_for(lambda: self.sequence >= target or self.failure is not None,
                                            timeout=self.timeout)
            if self.failure:
                raise CameraError(self.failure)
            if not ready:
                raise CameraError('等待相机新帧超时')
            return self.image.copy()

    def start(self):
        self.thread.start()
        self.wait_frame(1)

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                raise CameraError('相机驱动关闭超时')


class Hardware:
    source = 'hardware'

    def __init__(self, config):
        self.link = StationLink(config)
        self.camera = FreshCamera(config.camera_index, config.frame_size)
        try:
            self.camera.start()
            self.link.connect()
        except BaseException:
            self.close()
            raise

    def capture(self, site, angle):
        self.link.move(POSE_CODES[site, angle])
        return self.camera.wait_frame(3)

    def finish(self):
        self.link.move(0)

    def close(self):
        self.link.close()
        self.camera.close()
