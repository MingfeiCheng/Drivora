"""ApolloProxyAgent — runs Baidu Apollo as a Drivora ADS over CARLA sensors.

Design (see docs/apollo_integration_plan.md):
  - real-sensor perception: CARLA camera/lidar are forwarded to Apollo, which
    runs its own Perception (no ground-truth obstacles/traffic lights);
  - no-ground-truth localization (option 2): GNSS/IMU are forwarded and Apollo's
    Localization computes the pose;
  - per-tick run_step caches the latest sensor frames; a background thread
    publishes them to Apollo at the rates Apollo expects, and a control thread
    pulls Apollo's ControlCommand back.

Multi-ego: container name / ports are derived from the ego id, mirroring the
Pylot proxy.

This file adds a new agent only; it does not modify any existing Drivora code.
Running it requires a built Apollo, a generated HD map,
and matching sensor calibration — the empirically-tuned pieces are flagged in
``transform.py`` and the sensor publishers.
"""
import re
import json
import time
import threading
import traceback

import carla
from loguru import logger

from agent_corpus.atomic.base_agent import AutonomousAgent
from .transform import CarlaApolloTransform


def get_entry_point():
    return "ApolloProxyAgent"


def _ego_index(ego_id: str) -> int:
    m = re.search(r"(\d+)", ego_id or "")
    return int(m.group(1)) if m else 0


# Default sensor rig. Camera ids/resolutions must line up with the Apollo
# perception config + calibration (see transform.py / publishers/camera.py).
_DEFAULT_SENSORS = [
    {"type": "sensor.camera.rgb", "id": "front_6mm",
     "x": 1.5, "y": 0.0, "z": 1.7, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
     "width": 1920, "height": 1080, "fov": 30},
    {"type": "sensor.camera.rgb", "id": "front_12mm",
     "x": 1.5, "y": 0.0, "z": 1.7, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
     "width": 1920, "height": 1080, "fov": 15},
    {"type": "sensor.lidar.ray_cast", "id": "lidar128",
     "x": 0.0, "y": 0.0, "z": 1.9, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
     "range": 100, "rotation_frequency": 10, "channels": 128, "points_per_second": 1500000},
    {"type": "sensor.other.imu", "id": "imu",
     "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    {"type": "sensor.other.gnss", "id": "gnss",
     "x": 0.0, "y": 0.0, "z": 0.0},
    {"type": "sensor.speedometer", "id": "speed"},
]


class ApolloProxyAgent(AutonomousAgent):

    # background loop rates (s)
    PUBLISH_INTERVAL = 0.01
    CONTROL_INTERVAL = 0.01
    LOCALIZATION_WARMUP_SEC = 1.0
    ROUTING_TIMEOUT = 5.0
    SPEED_READY_THRESHOLD = 0.05

    # ----- setup -----
    def setup(self, path_to_conf_file):
        with open(path_to_conf_file, "r") as f:
            cfg = json.load(f)
        self._cfg = cfg

        idx = _ego_index(self.id)
        prefix = cfg.get("container_name_prefix", "drivora-apollo")
        self._container_name = f"{prefix}-{self.id}" if self.id else prefix
        self._bridge_port = cfg.get("bridge_port", 9090) + idx
        self._dreamview_port = cfg.get("dreamview_port", 8888) + idx
        self._gpu = str(cfg.get("gpu", 0))
        self._map_name = cfg.get("map_name", "san_mateo")
        self._apollo_root = cfg["apollo_root"]  # required: path to Apollo tree
        self._auto_stop = cfg.get("auto_stop_container", False)
        self._trigger_time = cfg.get("trigger_time", 0.0)
        self._sensors = cfg.get("sensors", _DEFAULT_SENSORS)

        self.transform = CarlaApolloTransform(cfg.get("calibration", {}))

        # shared state between run_step (writer) and the loops (readers)
        self._lock = threading.Lock()
        self._latest_input = None
        self._latest_ts = 0.0
        self._latest_control = carla.VehicleControl(throttle=0.0, brake=1.0)
        self._route_msg = None

        # routing FSM
        self._route_sent = False
        self._route_confirmed = False
        self._route_send_wall = 0.0
        self._ready = False
        self._control_enabled = False
        self._running = False

        # Build messenger lazily after the route is known? Apollo can boot
        # without a route, so start it now.
        from .bridge.messenger import ApolloMessenger
        from .bridge.format import ControlPadMessage
        self._ControlPadMessage = ControlPadMessage

        self.messenger = ApolloMessenger(
            idx=self.id,
            apollo_modules=cfg.get("apollo_modules", [
                "Transform", "Localization", "Perception",
                "Prediction", "Planning", "Control", "Routing",
            ]),
            publishers=cfg.get("publishers", [
                "publisher.camera", "publisher.lidar", "publisher.imu",
                "publisher.gnss", "publisher.chassis",
                "publisher.routing_request", "publisher.control_pad",
            ]),
            subscribers=["subscriber.control"],
            container_name=self._container_name,
            gpu=self._gpu,
            cpu=cfg.get("cpu", 24.0),
            apollo_root=self._apollo_root,
            map_name=self._map_name,
            dreamview_port=self._dreamview_port,
            bridge_port=self._bridge_port,
            start_modules=cfg.get("start_modules", True),
            start_dreamview=cfg.get("start_dreamview", False),
        )

        self._running = True
        self._pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._ctl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._pub_thread.start()

        logger.info(f"[Apollo {self.id}] warm-up {self.LOCALIZATION_WARMUP_SEC}s before control")
        time.sleep(self.LOCALIZATION_WARMUP_SEC)
        self._control_enabled = True
        self._ctl_thread.start()
        logger.info(f"[Apollo {self.id}] setup complete (container={self._container_name})")

    def sensors(self):
        return self._sensors

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._route_msg = self.transform.build_route(self._global_plan_world_coord, ts=0.0)

    # ----- per-tick -----
    def run_step(self, input_data, timestamp):
        with self._lock:
            self._latest_input = input_data
            self._latest_ts = timestamp
            control = self._latest_control
        return control, {}

    # ----- background publish -----
    def _publish_loop(self):
        pub = self.messenger.publish_message
        pub("publisher.control_pad", self._ControlPadMessage(timestamp=0.0, action=0))
        while self._running:
            try:
                self._publish_once(pub)
            except Exception:
                logger.opt(exception=True).warning(f"[Apollo {self.id}] publish error")
            time.sleep(self.PUBLISH_INTERVAL)

    def _publish_once(self, pub):
        with self._lock:
            input_data = self._latest_input
            ts = self._latest_ts
            last_control = self._latest_control
        if input_data is None or self.carla_actor is None:
            return

        ts = time.time()  # Apollo expects wall-clock-like monotonic stamps

        # ego proprioception (own GNSS/IMU/speedometer — not environment truth)
        ego_tf = self.carla_actor.get_transform()
        vel = self.carla_actor.get_velocity()
        speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5

        # chassis
        pub("publisher.chassis", self.transform.build_chassis(speed, last_control, ts))

        # sensors → Apollo Perception + Localization
        for spec in self._sensors:
            sid = spec["id"]
            if sid not in input_data:
                continue
            _, data = input_data[sid]
            stype = spec["type"]
            if stype == "sensor.camera.rgb":
                pub("publisher.camera", self.transform.build_camera(sid, data, ts))
            elif stype == "sensor.lidar.ray_cast":
                pub("publisher.lidar", self.transform.build_lidar(data, ts, lidar_name=sid))
            elif stype == "sensor.other.imu":
                pub("publisher.imu", self.transform.build_imu(data, ts))

        if "gnss" in input_data and "imu" in input_data:
            _, gnss = input_data["gnss"]
            _, imu = input_data["imu"]
            pub("publisher.gnss",
                self.transform.build_gnss(gnss, imu, ego_tf, vel, ts))

        self._routing_fsm(speed, ts, pub)

    def _routing_fsm(self, ego_speed, ts, pub):
        if self._ready or self._route_msg is None:
            return
        now = time.time()
        Pad = self._ControlPadMessage
        if not self._route_sent:
            self._route_send_wall = now
            self._route_msg.timestamp = ts
            pub("publisher.routing_request", self._route_msg)
            pub("publisher.control_pad", Pad(timestamp=ts, action=1))  # START
            self._route_sent = True
        elif not self._route_confirmed:
            if ego_speed > self.SPEED_READY_THRESHOLD:
                self._route_confirmed = True
            elif now - self._route_send_wall > self.ROUTING_TIMEOUT:
                self._route_send_wall = now
                self._route_msg.timestamp = ts
                pub("publisher.routing_request", self._route_msg)
                pub("publisher.control_pad", Pad(timestamp=ts, action=1))
        else:
            self._ready = True

    # ----- background control -----
    def _control_loop(self):
        sub = self.messenger.subscriber_pool["subscriber.control"]
        while self._running:
            try:
                t, b, s, r = sub.get_data()
                if self._control_enabled and self._latest_ts >= self._trigger_time:
                    ctrl = self.transform.apollo_control_to_carla(t, b, s, r)
                    with self._lock:
                        self._latest_control = ctrl
            except Exception:
                logger.opt(exception=True).warning(f"[Apollo {self.id}] control error")
            time.sleep(self.CONTROL_INTERVAL)

    # ----- teardown -----
    def destroy(self):
        self._running = False
        for th in (getattr(self, "_pub_thread", None), getattr(self, "_ctl_thread", None)):
            if th is not None:
                try:
                    th.join(timeout=3.0)
                except Exception:
                    pass
        try:
            self.messenger.shutdown()
        except Exception:
            traceback.print_exc()
        if self._auto_stop:
            try:
                self.messenger.container.stop_container()
            except Exception:
                pass
        logger.info(f"[Apollo {self.id}] destroyed")
