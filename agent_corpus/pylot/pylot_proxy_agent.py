"""
PylotProxyAgent — Host-side proxy that bridges Drivora's AutonomousAgent
interface with a remote Pylot container running PylotServer.

Supports multi-ego scenarios: each ego vehicle gets its own Pylot container
with a unique name and port, derived automatically from the ego ID.

Config file (JSON) passed via ego_config in setup():
{
    "container_host": "localhost",
    "base_port": 12667,
    "jpeg_quality": 90,
    "docker_image": "drivora/pylot:latest",
    "container_name_prefix": "drivora-pylot",
    "gpu": 0,
    "pylot_config": "configs/drivora.conf",
    "auto_stop_container": false
}

For multi-ego, each instance auto-derives:
  - container_name = "{prefix}-{ego_id}"   e.g. "drivora-pylot-ego-0"
  - port = base_port + ego_index           e.g. 12667, 12668, ...
"""

import json
import re
import time
import subprocess
import traceback

import cv2
import zmq
import carla
import msgpack
import msgpack_numpy as m
m.patch()

from loguru import logger

from agent_corpus.atomic.base_agent import AutonomousAgent


def get_entry_point():
    return 'PylotProxyAgent'


def _extract_ego_index(ego_id: str) -> int:
    """Extract numeric index from ego ID like 'ego_0', 'ego_1', etc."""
    match = re.search(r'(\d+)', ego_id or '')
    return int(match.group(1)) if match else 0


class PylotProxyAgent(AutonomousAgent):

    # ── Container lifecycle ───────────────────────────────────────────────

    def _ensure_container_running(self):
        """Start the Pylot container if not running, or reuse existing one."""
        name = self._container_name

        # Check if already running
        result = subprocess.run(
            f"docker ps -q -f name=^{name}$",
            shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            logger.info(f"Pylot container '{name}' already running — reusing.")
            return

        # Remove stopped container if exists
        result = subprocess.run(
            f"docker ps -aq -f name=^{name}$",
            shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            logger.info(f"Removing stopped container '{name}'...")
            subprocess.run(f"docker rm -f {name}", shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.0)

        # Start new container
        cmd = (
            f"docker run --name {name} -d --rm "
            f"--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES={self._gpu} "
            f"--gpus 'device={self._gpu}' "
            f"-p {self._port}:{self._port} "
            f"{self._docker_image} "
            f"--config {self._pylot_config} --port {self._port}"
        )
        logger.info(f"Starting Pylot container: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

        # Wait until container is running
        start_time = time.time()
        max_wait = 60.0
        while time.time() - start_time < max_wait:
            result = subprocess.run(
                f"docker ps -q -f name=^{name}$",
                shell=True, capture_output=True, text=True)
            if result.stdout.strip():
                logger.info(f"Container '{name}' is running.")
                break
            time.sleep(1.0)
        else:
            raise TimeoutError(f"Pylot container '{name}' did not start within {max_wait}s")

        # Wait for server to be ready by polling with ping
        logger.info("Waiting for PylotServer to be ready (TF model loading may take 30-60s)...")
        self._wait_for_server_ready()

    def _stop_container(self):
        """Stop the Pylot container."""
        name = self._container_name
        logger.info(f"Stopping Pylot container '{name}'...")
        subprocess.run(f"docker stop {name}", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _wait_for_server_ready(self, max_wait=120):
        """Poll the server with ping until it responds, up to max_wait seconds."""
        ctx = zmq.Context()
        start = time.time()
        attempt = 0
        while time.time() - start < max_wait:
            attempt += 1
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 3000)
            sock.setsockopt(zmq.SNDTIMEO, 3000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://{self._host}:{self._port}")
            try:
                sock.send(msgpack.packb({'cmd': 'ping'}, default=m.encode))
                resp = msgpack.unpackb(sock.recv(), raw=False, object_hook=m.decode)
                if resp.get('status') == 'ok':
                    elapsed = time.time() - start
                    logger.info(f"PylotServer ready after {elapsed:.1f}s (attempt {attempt})")
                    sock.close()
                    ctx.term()
                    return
            except Exception:
                if attempt % 5 == 1:
                    logger.info(f"Still waiting for PylotServer... ({time.time()-start:.0f}s elapsed)")
            finally:
                sock.close()
            time.sleep(3.0)

        ctx.term()
        raise RuntimeError(f"PylotServer not ready after {max_wait}s")

    # ── Agent interface ───────────────────────────────────────────────────

    def setup(self, path_to_conf_file):
        """Load config, auto-start container, connect ZMQ, fetch sensor specs.

        For multi-ego support, the container name and port are derived from
        self.id (set by setup_env before this method is called):
          - container_name = "{prefix}-{ego_id}"
          - port = base_port + ego_index
        """
        # ── Load proxy config ──
        with open(path_to_conf_file, 'r') as f:
            cfg = json.load(f)

        self._host = cfg.get('container_host', 'localhost')
        self._jpeg_quality = cfg.get('jpeg_quality', 90)
        self._docker_image = cfg.get('docker_image', 'drivora/pylot:latest')
        self._gpu = cfg.get('gpu', 0)
        self._pylot_config = cfg.get('pylot_config', 'configs/drivora.conf')
        self._auto_stop = cfg.get('auto_stop_container', False)

        # ── Derive per-ego container name and port ──
        ego_index = _extract_ego_index(self.id)
        base_port = cfg.get('base_port', cfg.get('container_port', 12667))
        prefix = cfg.get('container_name_prefix', cfg.get('container_name', 'drivora-pylot'))

        self._port = base_port + ego_index
        self._container_name = f"{prefix}-{self.id}" if self.id else prefix

        logger.info(f"PylotProxy [{self.id}] → container='{self._container_name}' port={self._port}")

        # ── Auto-start container (skip if prefix is empty) ──
        if self._container_name:
            self._ensure_container_running()

        # ── ZMQ REQ socket (main socket for all operations) ──
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, 60000)
        self._socket.setsockopt(zmq.SNDTIMEO, 10000)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self._host}:{self._port}")
        logger.info(f"PylotProxy [{self.id}] connected to tcp://{self._host}:{self._port}")

        # ── Get sensor config from container ──
        resp = self._send_cmd({'cmd': 'get_sensors'})
        if resp.get('status') != 'ok':
            raise RuntimeError(f"get_sensors failed: {resp}")
        self._sensor_specs = resp['sensors']

        # ── Identify camera sensor IDs (for JPEG encoding) ──
        self._camera_ids = set()
        for spec in self._sensor_specs:
            if spec['type'].startswith('sensor.camera'):
                self._camera_ids.add(spec['id'])

        # ── State ──
        self._route_sent = False
        self._opendrive_sent = False
        self._last_control = carla.VehicleControl()

        logger.info(f"PylotProxy [{self.id}] setup complete — {len(self._sensor_specs)} sensors "
                     f"({len(self._camera_ids)} cameras)")

    def sensors(self):
        """Return sensor specs obtained from the container."""
        return self._sensor_specs

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """Override to also buffer the route for sending to container."""
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._buffered_route = []
        for transform, road_option in self._global_plan_world_coord:
            self._buffered_route.append({
                'x': transform.location.x,
                'y': transform.location.y,
                'z': transform.location.z,
                'pitch': transform.rotation.pitch,
                'yaw': transform.rotation.yaw,
                'roll': transform.rotation.roll,
                'road_option': road_option.value,
            })

    def run_step(self, input_data, timestamp):
        """Pack sensor data -> send to container -> receive control."""
        try:
            control, log_data = self._run_step_inner(input_data, timestamp)
        except Exception as e:
            logger.error(f"PylotProxy [{self.id}] run_step error: {e}")
            traceback.print_exc()
            control = carla.VehicleControl(throttle=0.0, brake=1.0)
            log_data = {'error': str(e)}

        self._last_control = control
        return control, log_data

    def _run_step_inner(self, input_data, timestamp):
        # ── Send init (opendrive + route) on first tick ──
        if not self._opendrive_sent:
            opendrive_data = input_data.get('opendrive')
            if opendrive_data is not None:
                init_msg = {
                    'cmd': 'init',
                    'opendrive': opendrive_data[1].get('opendrive', '')
                        if isinstance(opendrive_data[1], dict) else opendrive_data[1],
                }
                if hasattr(self, '_buffered_route'):
                    init_msg['route'] = self._buffered_route
                if self.carla_actor is not None:
                    init_msg['vehicle_id'] = self.carla_actor.id
                resp = self._send_cmd(init_msg)
                if resp.get('status') != 'ok':
                    logger.error(f"init failed: {resp}")
                self._opendrive_sent = True
                self._route_sent = True

        if not self._route_sent and hasattr(self, '_buffered_route'):
            init_msg = {'cmd': 'init', 'route': self._buffered_route}
            if self.carla_actor is not None:
                init_msg['vehicle_id'] = self.carla_actor.id
            self._send_cmd(init_msg)
            self._route_sent = True

        # ── Pack sensor data ──
        game_time = int(timestamp * 1000)
        sensors_payload = {'cameras': {}}

        for key, val in input_data.items():
            _, data = val

            if key in self._camera_ids:
                img = data[:, :, :3] if data.shape[2] == 4 else data
                _, jpeg_buf = cv2.imencode('.jpg', img,
                                           [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
                sensors_payload['cameras'][key] = jpeg_buf.tobytes()

            elif key == 'LIDAR':
                sensors_payload['lidar'] = data

            elif key == 'imu':
                sensors_payload['imu'] = data

            elif key == 'gnss':
                sensors_payload['gnss'] = data

            elif key == 'speed':
                sensors_payload['speed'] = data

        # ── Send tick ──
        resp = self._send_cmd({
            'cmd': 'tick',
            'timestamp': game_time,
            'sensors': sensors_payload,
        })

        if resp.get('status') != 'ok':
            logger.warning(f"tick failed: {resp}")
            return self._last_control, {'error': resp.get('message', 'unknown')}

        # ── Unpack control ──
        ctrl = resp['control']
        control = carla.VehicleControl()
        control.throttle = float(ctrl.get('throttle', 0.0))
        control.steer = float(ctrl.get('steer', 0.0))
        control.brake = float(ctrl.get('brake', 0.0))
        control.hand_brake = bool(ctrl.get('hand_brake', False))
        control.reverse = bool(ctrl.get('reverse', False))
        control.manual_gear_shift = False

        return control, {}

    def destroy(self):
        """Tell container to reset for next scenario, close ZMQ."""
        try:
            self._send_cmd({'cmd': 'destroy'})
        except Exception as e:
            logger.warning(f"destroy command failed: {e}")
        finally:
            self._socket.close()
            self._ctx.term()

        if self._auto_stop:
            self._stop_container()

        logger.info(f"PylotProxy [{self.id}] destroyed.")

    # ── ZMQ helper ────────────────────────────────────────────────────────

    def _send_cmd(self, msg: dict) -> dict:
        """Send a msgpack message and return the response."""
        self._socket.send(msgpack.packb(msg, default=m.encode))
        raw = self._socket.recv()
        return msgpack.unpackb(raw, raw=False, object_hook=m.decode)
