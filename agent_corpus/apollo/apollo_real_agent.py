"""ApolloRealAgent — Drivora ADS with REAL LiDAR perception via Apollo.

The full real-perception version of the Apollo Drivora agent. Connects to a
running Apollo (with all decision modules + RTK localization + lidar perception
+ in-container sensor injector), declares real sensors (128-beam lidar + imu +
gnss + speedometer), and per tick:
  - forwards the CARLA lidar point cloud through the in-container injector
    (TCP) so Apollo cnnseg detects real obstacles;
  - feeds GNSS odometry + corrected IMU through the bridge so RTK localization
    computes the pose + broadcasts /tf;
  - feeds chassis + routing (snapped to a routing-graph lane);
  - returns Apollo's ControlCommand for Drivora to apply.

Compared to ApolloDemoAgent (which injects pose + empty perception), this uses
the full real-sensor pipeline — ego avoids obstacles Apollo actually detects.

Config (JSON): apollo_host, bridge_port (default 9090), injector_port (9100),
calibration, cruise_speed (cap on planning's cruise speed, m/s; default 6.0).
"""
import os, math, time, json, socket, struct, threading
import numpy as np
import carla
from loguru import logger

from agent_corpus.atomic.base_agent import AutonomousAgent
from .transform import CarlaApolloTransform
from .bridge.cyber_bridge import CyberBridge
from .bridge import PUBLISHER_REGISTRY, SUBSCRIBER_REGISTRY
from .bridge import format as F
import agent_corpus.apollo.bridge.publishers  # noqa
import agent_corpus.apollo.bridge.subscribers  # noqa
import importlib
from scipy.spatial.transform import Rotation

_topo = importlib.import_module("apollo_modules.modules.routing.proto.topo_graph_pb2")
_hdr = importlib.import_module("apollo_modules.modules.common.proto.header_pb2")
_perc = importlib.import_module("apollo_modules.modules.perception.proto.perception_obstacle_pb2")
_tf_pb = importlib.import_module("apollo_modules.modules.transform.proto.transform_pb2")
_ROUTING_BIN = os.path.join(os.path.dirname(__file__), "map", "Town01", "routing_map.bin")


def get_entry_point():
    return "ApolloRealAgent"


class ApolloRealAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        with open(path_to_conf_file) as f:
            cfg = json.load(f)
        self._host = cfg.get("apollo_host", "172.17.0.2")
        self._bport = cfg.get("bridge_port", 9090)
        self._iport = cfg.get("injector_port", 9100)
        self._iport_cam = cfg.get("injector_port_camera", self._iport + 1)      # front_6mm  injector PROCESS
        self._iport_cam12 = cfg.get("injector_port_camera12", self._iport + 2)  # front_12mm injector PROCESS
        self.tf = CarlaApolloTransform(cfg.get("calibration", {"flip_y": True}))

        # load routing graph lanes for route snap
        g = _topo.Graph(); g.ParseFromString(open(_ROUTING_BIN, "rb").read())
        self._lanes = []
        for n in g.node:
            pts = [(p.x, p.y) for s in n.central_curve.segment if s.HasField("line_segment")
                   for p in s.line_segment.point]
            if len(pts) >= 2:
                self._lanes.append({"id": n.lane_id, "len": n.length or 0.0, "pts": pts})

        self._lock = threading.Lock()
        self._latest_ctrl = carla.VehicleControl(throttle=0.0, brake=1.0)
        self._latest_lidar = None     # Nx4 float32, velodyne128 frame
        self._latest_imu = None       # CARLA imu array
        self._route_msg = None
        self._ready = False
        self._route_sent_wall = 0.0
        self._route_first_wall = 0.0
        self._running = True
        self._start_wall = time.time()
        self._lidar_warmup_sec = cfg.get("lidar_warmup_sec", 5.0)  # delay lidar injection so planning can build reference line first
        # camera -> injector -> /apollo/sensor/camera/front_6mm/image (camera
        # perception + traffic-light detection). NOTE: meaningful camera
        # perception also needs front_6mm intrinsics/extrinsics calibration to
        # match this resolution/mount; transmission works regardless.
        self._enable_camera = bool(cfg.get("enable_camera", True))
        self._cam_w = int(cfg.get("camera_width", 1920))
        self._cam_h = int(cfg.get("camera_height", 1080))
        self._cam_fov = float(cfg.get("camera_fov", 60.0))
        self._cam_fov12 = float(cfg.get("camera_fov_12mm", 45.0))  # front_12mm = narrower (telephoto)
        self._cam_period = 1.0 / float(cfg.get("camera_hz", 10.0))
        # throttle gain: Apollo plans ~11 m/s but its throttle command only holds
        # ~5 m/s on CARLA's mkz (real-MKZ control calibration mismatch). Scale up.
        self._throttle_gain = float(cfg.get("throttle_gain", 1.0))

        # bridge — feed GNSS+IMU to RTK Localization (standalone full_pipeline
        # recipe; RTK computes pose + broadcasts /tf natively, no manual /tf).
        self.b = CyberBridge(self._host, self._bport)
        self.imu_pub = PUBLISHER_REGISTRY.get("publisher.imu")(idx="i", bridge=self.b)
        self.gnss_pub = PUBLISHER_REGISTRY.get("publisher.gnss")(idx="g", bridge=self.b)
        self.ch_pub = PUBLISHER_REGISTRY.get("publisher.chassis")(idx="c", bridge=self.b)
        self.rt_pub = PUBLISHER_REGISTRY.get("publisher.routing_request")(idx="r", bridge=self.b)
        self.pad_pub = PUBLISHER_REGISTRY.get("publisher.control_pad")(idx="p", bridge=self.b)
        self.ctl_sub = SUBSCRIBER_REGISTRY.get("subscriber.control")(idx="s", bridge=self.b)
        self.b.spin()

        # injector socket for bulk lidar (bypasses cyber_bridge bulk limit)
        self.inj = socket.create_connection((self._host, self._iport))
        self.inj.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # SEPARATE socket for camera: a 6MB jpeg frame on the shared lidar socket
        # intermittently starves lidar -> perception detections drop to 0 (flicker).
        # The injector listen()s with a thread per connection, so a 2nd socket
        # runs camera in its own handler thread -> lidar stays steady.
        self.inj_cam = None
        self.inj_cam12 = None
        if self._enable_camera:
            self.inj_cam = socket.create_connection((self._host, self._iport_cam))
            self.inj_cam.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # front_12mm — Apollo's FusionCameraDetectionComponent expects BOTH
            # front_6mm AND front_12mm; feeding only one leaves the component
            # starved on the missing channel and destabilises camera detections.
            self.inj_cam12 = socket.create_connection((self._host, self._iport_cam12))
            self.inj_cam12.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Only the control receiver runs in a thread (lightweight, just reads
        # subscriber). Publishing happens in run_step (driven by Drivora's tick)
        # to avoid bridge-socket send contention with Drivora's CARLA main loop.
        self._ctl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._ctl_thread.start()
        self._tick = 0
        self._last_pub_wall = 0.0  # throttle GNSS/IMU/chassis to 10Hz (Drivora ticks 20Hz; standalone full_pipeline = 10Hz)
        self._last_lidar_wall = 0.0  # throttle lidar->injector to 10Hz (lidar.rotation_frequency=10)
        self._prev_lidar = None      # accumulate consecutive half-sweeps into a full 360° frame
        self._last_cam_wall = 0.0    # throttle camera->injector to camera_hz
        logger.info(f"[ApolloReal {self.id}] connected bridge {self._host}:{self._bport}, injector :{self._iport}"
                    f"{' +camera' if self._enable_camera else ''}")

    def sensors(self):
        # real sensor rig (declared to Drivora's AgentWrapper)
        rig = [
            # Match REAL velodyne128 params (guardstrikelab reference) so cnnseg —
            # trained on real velodyne128 — gets the point pattern it expects.
            # upper_fov/lower_fov 2.0/-26.8 = real velodyne128 FOV (was wrongly
            # 10/-25); z=2.4 matches the velodyne128_novatel extrinsics z≈2.31
            # (was 1.9); rotation_frequency=20=sim fps (full sweeps).
            {"type": "sensor.lidar.ray_cast", "id": "lidar128",
             "x": 0.0, "y": 0.0, "z": 2.4, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "range": 50, "rotation_frequency": 20, "channels": 128,
             "points_per_second": 320000, "upper_fov": 2.0, "lower_fov": -26.8},
            {"type": "sensor.other.imu", "id": "imu",
             "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            {"type": "sensor.other.gnss", "id": "gnss",
             "x": 0.0, "y": 0.0, "z": 0.0},
            {"type": "sensor.speedometer", "id": "speed"},
        ]
        if self._enable_camera:
            # front-facing RGB -> Apollo front_6mm (camera obstacle + traffic light)
            # mounted coincident with the lidar (x=0,z=1.9) so front_6mm
            # extrinsics = pure optical rotation, zero translation (exact).
            rig.append({
                "type": "sensor.camera.rgb", "id": "camera_front",
                "x": 0.0, "y": 0.0, "z": 1.9, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "width": self._cam_w, "height": self._cam_h, "fov": self._cam_fov,
            })
            rig.append({
                "type": "sensor.camera.rgb", "id": "camera_front12",
                "x": 0.0, "y": 0.0, "z": 1.9, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "width": self._cam_w, "height": self._cam_h, "fov": self._cam_fov12,
            })
        return rig

    def _snap_lane(self, mx, my):
        """Nearest routing-graph lane to apollo-frame (mx,my); returns (lane, idx)."""
        best, bd, bi = None, 1e18, 0
        for ln in self._lanes:
            for i, (px, py) in enumerate(ln["pts"]):
                d = (px - mx) ** 2 + (py - my) ** 2
                if d < bd:
                    bd, best, bi = d, ln, i
        return best, bi

    @staticmethod
    def _s_of(ln, idx):
        n = len(ln["pts"])
        return max(0.0, min(ln.get("len", 0.0) or 0.0, (idx / max(1, n - 1)) * (ln.get("len", 0.0) or 0.0)))

    def _build_route(self):
        t = self.carla_actor.get_transform()
        mx, my, _ = self.tf.loc_carla_to_apollo(t.location.x, t.location.y, t.location.z)
        a, ai = self._snap_lane(mx, my)
        if not a:
            return None
        # Destination = end of Drivora's global plan (scenario route) if available,
        # so Apollo routes THROUGH junctions to the goal instead of stopping at the
        # spawn lane's end. Falls back to spawn-lane-end when no plan is set.
        dest = None
        gp = getattr(self, "_global_plan_world_coord", None)
        if gp:
            try:
                last = gp[-1][0]
                loc = getattr(last, "location", last)
                dx, dy, _ = self.tf.loc_carla_to_apollo(loc.x, loc.y, 0.0)
                db, dbi = self._snap_lane(dx, dy)
                # Use the scenario destination whenever it's a DISTINCT point —
                # a different lane, OR the same lane at a different index (s).
                # The old `db["id"] != a["id"]` discarded same-lane routes (every
                # short straight route!), making Apollo route to the LANE END
                # instead of wp1 -> Dreamview showed a far endpoint and the ego
                # stopped early when reach_destination fired at the real wp1.
                if db and (db["id"] != a["id"] or dbi != ai):
                    dest = (db, dbi)
            except Exception:
                dest = None
        p0 = a["pts"][ai]
        a1 = a["pts"][min(ai + 1, len(a["pts"]) - 1)]
        h = math.atan2(a1[1] - p0[1], a1[0] - p0[0])
        wps = [F.Waypoint(F.Lane(a["id"], self._s_of(a, ai), 0.0),
                          F.Location(p0[0], p0[1], 0, 0, h, 0))]
        if dest:
            b, bi = dest
            pe = b["pts"][bi]
            wps.append(F.Waypoint(F.Lane(b["id"], self._s_of(b, bi), 0.0),
                                  F.Location(pe[0], pe[1], 0, 0, h, 0)))
        else:
            pe = a["pts"][-1]
            wps.append(F.Waypoint(F.Lane(a["id"], max(1.0, (a.get("len", 0.0) or 2.0) - 2.0), 0.0),
                                  F.Location(pe[0], pe[1], 0, 0, h, 0)))
        return F.RouteMessage(timestamp=0.0, waypoints=wps)

    def _publish_loop(self):
        # high-rate: localization (inject) + /tf + chassis + empty perception
        # placeholder; routing FSM. Real lidar goes via run_step -> injector.
        i = 0
        while self._running:
            try:
                if self.carla_actor is None:
                    time.sleep(0.02); continue
                if self._route_msg is None:
                    self._route_msg = self._build_route()
                t = self.carla_actor.get_transform(); v = self.carla_actor.get_velocity()
                ts = time.time()
                mx, my, _ = self.tf.loc_carla_to_apollo(t.location.x, t.location.y, t.location.z)
                hd = self.tf.yaw_carla_to_apollo(t.rotation.yaw)
                spd = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
                with self._lock:
                    last = self._latest_ctrl
                self.loc_pub.publish(F.LocalizationMessage(ts, F.Location(mx, my, 0, 0, hd, 0), hd,
                                     self.tf.vec_carla_to_apollo(v.x, v.y, v.z),
                                     F.Vector(0, 0, 0), F.Vector(0, 0, 0)))
                # /tf world->novatel (perception lidar needs this)
                adj = (hd - math.pi/2 + math.pi) % (2*math.pi) - math.pi
                qx, qy, qz, qw = Rotation.from_euler("z", adj).as_quat()
                tfs = _tf_pb.TransformStampeds(header=_hdr.Header(timestamp_sec=ts, frame_id="world"))
                tr = tfs.transforms.add()
                tr.header.timestamp_sec = ts; tr.header.frame_id = "world"; tr.child_frame_id = "novatel"
                tr.transform.translation.x = mx; tr.transform.translation.y = my; tr.transform.translation.z = 0.0
                tr.transform.rotation.qx = qx; tr.transform.rotation.qy = qy
                tr.transform.rotation.qz = qz; tr.transform.rotation.qw = qw
                self.b.publish("/tf", tfs.SerializeToString())
                self.ch_pub.publish(F.ChassisMessage(ts, spd, last.throttle*100, last.brake*100,
                                                     last.steer*100, last.reverse))
                # routing FSM
                if not self._ready and self._route_msg is not None:
                    if spd > 0.3:
                        self._ready = True
                    elif ts - self._route_sent_wall > 2.0:
                        self._route_msg.timestamp = ts
                        self.rt_pub.last_publish_time = None; self.rt_pub.publish(self._route_msg)
                        self.pad_pub.last_publish_time = None; self.pad_pub.publish(F.ControlPadMessage(ts, 1))
                        self._route_sent_wall = ts
                i += 1
                if i == 1 or i % 500 == 0:
                    logger.info(f"[ApolloReal {self.id}] publish_loop tick={i} ego=({mx:.1f},{my:.1f}) hd={hd:.2f} route_sent={self._route_sent_wall>0}")
            except Exception:
                logger.opt(exception=True).warning(f"[ApolloReal {self.id}] publish err")
            time.sleep(0.01)

    def _control_loop(self):
        while self._running:
            try:
                t, br, s, r = self.ctl_sub.get_data()
                if self.ctl_sub.control_data is not None:
                    with self._lock:
                        self._latest_ctrl = self.tf.apollo_control_to_carla(t, br, s, r, self._throttle_gain)
            except Exception:
                pass
            time.sleep(0.01)

    def run_step(self, input_data, timestamp):
        # Drivora ticks at 20Hz; standalone full_pipeline (proven to drive)
        # publishes at 10Hz. Publishing at 20Hz overflows Apollo prediction's
        # /apollo/localization/pose reader buffer (drop_message warnings)
        # eventually starving prediction -> planning -> control (estop). So we
        # throttle GNSS/IMU/chassis to ~10Hz wall, and lidar to ~10Hz wall.
        ts = time.time()
        do_pub = (ts - self._last_pub_wall) >= 0.095
        do_lidar = (ts - self._last_lidar_wall) >= 0.095
        do_cam = self._enable_camera and (ts - self._last_cam_wall) >= self._cam_period
        try:
            if self._route_msg is None:
                self._route_msg = self._build_route()
            t = self.carla_actor.get_transform(); v = self.carla_actor.get_velocity()
            mx, my, _ = self.tf.loc_carla_to_apollo(t.location.x, t.location.y, t.location.z)
            hd = self.tf.yaw_carla_to_apollo(t.rotation.yaw)
            spd = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
            with self._lock:
                last = self._latest_ctrl
            vel = self.tf.vec_carla_to_apollo(v.x, v.y, v.z)
            if do_pub:
                # match standalone full_pipeline: imu first (RTK needs imu within
                # 20ms before GNSS), then GNSS, then chassis. Don't bypass the
                # publisher throttle; let it enforce the upstream rate cap.
                self.imu_pub.publish(F.IMUMessage(ts, F.Vector(0,0,0), F.Vector(0,0,0), F.Vector(0,0,hd)))
                self.gnss_pub.publish(F.GNSSMessage(ts, 0, 0, 0, mx, my, 0.0, hd, vel))
                self.ch_pub.publish(F.ChassisMessage(ts, spd, last.throttle*100, last.brake*100,
                                                     last.steer*100, last.reverse))
                self._last_pub_wall = ts
            # routing FSM: SEND routing first (must happen at least once even
            # if ego has spawn-velocity on tick 1), then mark ready only once
            # we've sent AND ego is moving under its own power.
            #
            # IMPORTANT: control_conf.enable_persistent_estop defaults to TRUE
            # in Apollo. If a previous run left control in estop, it stays in
            # estop until DrivingAction.RESET=2 arrives. Also, planning may
            # produce empty trajectories during the first few seconds (waiting
            # for reference line / routing / localization to converge) — each
            # empty trajectory re-sets estop_=true. So we send pad RESET at a
            # high cadence (every 5 publish-ticks ≈ 0.5s) for the first 10s,
            # then back off to the 2s interval once ego is moving.
            if not self._ready and self._route_msg is not None:
                already_sent = self._route_sent_wall > 0.0
                # Mark ready once the ego has been moving for >3s since the FIRST
                # routing send, then STOP re-sending routing + RESET/START. The
                # old logic re-sent every 2s and gated readiness on ">5s since
                # last send", so readiness never triggered -> RESET was spammed
                # forever, which kept Apollo planning re-initialising and capped
                # the ego at ~5 m/s. Keep _route_first_wall fixed at first send.
                if already_sent and spd > 0.3 and (ts - self._route_first_wall) > 3.0:
                    self._ready = True
                    logger.info(f"[ApolloReal {self.id}] route ready -> stop re-send, let it cruise (spd={spd:.2f})")
                else:
                    # only (re)send while not yet moving: first send, or every 2s if stalled
                    if (not already_sent) or (spd < 0.3 and (ts - self._route_sent_wall) > 2.0):
                        self._route_msg.timestamp = ts
                        self.rt_pub.last_publish_time = None; self.rt_pub.publish(self._route_msg)
                        if not already_sent:
                            self._route_first_wall = ts
                        self._route_sent_wall = ts
                        logger.info(f"[ApolloReal {self.id}] sent routing tick={self._tick}")
                    # clear estop only while stalled; once moving, stop (RESET
                    # spam was resetting planning and killing cruise speed)
                    if spd < 0.3:
                        self.pad_pub.last_publish_time = None; self.pad_pub.publish(F.ControlPadMessage(ts, 2))
                        self.pad_pub.last_publish_time = None; self.pad_pub.publish(F.ControlPadMessage(ts, 1))
            self._tick += 1
            if self._tick == 1 or self._tick % 200 == 0:
                logger.info(f"[ApolloReal {self.id}] run_step tick={self._tick} pose=({mx:.1f},{my:.1f},{hd:.2f}) spd={spd:.2f} ctrl=th{last.throttle:.2f} br{last.brake:.2f}")
        except Exception:
            logger.opt(exception=True).warning(f"[ApolloReal {self.id}] run_step err")
        # forward CARLA lidar -> injector at ~10Hz wall (matches lidar
        # rotation_frequency=10Hz; faster would just send duplicate sweeps)
        if do_lidar and input_data is not None and "lidar128" in input_data:
            _, pts = input_data["lidar128"]
            if pts is not None and len(pts) > 0:
                # CARLA delivers a 180° HALF sweep per frame (consecutive frames
                # alternate front/back halves) -> objects fall in only every other
                # frame -> detection alternated N,0. Stitch the latest two frames
                # into a full 360° sweep so every published cloud is complete.
                if self._prev_lidar is not None:
                    pp = np.concatenate([pts, self._prev_lidar], axis=0).copy()
                else:
                    pp = pts.copy()
                self._prev_lidar = pts.copy()
                pp[:, 1] = -pp[:, 1]
                # filter EGO self-points: lidar is at (0,0,1.9) on the ego;
                # points within the ego bbox (≈ ±2.5 fwd × ±1 lat × below lidar)
                # would otherwise be detected by cnnseg as a phantom obstacle
                # AT the ego, triggering "collision with obstacle" -> stop.
                mask = ~((np.abs(pp[:, 0]) < 2.5) & (np.abs(pp[:, 1]) < 1.2) & (pp[:, 2] < -0.3))
                pp = pp[mask]
                if len(pp) > 40000:
                    # deterministic even stride (NOT random): a stable point
                    # pattern frame-to-frame so cnnseg detections don't flicker
                    # in/out (random subsampling changed which points existed
                    # each sweep -> intermittent obstacles).
                    pp = pp[:: (len(pp) + 39999) // 40000]
                pp = np.ascontiguousarray(pp, dtype="<f4")
                try:
                    self.inj.sendall(b"L" + struct.pack("<d", ts) + struct.pack("<I", len(pp)) + pp.tobytes())
                    self._last_lidar_wall = ts
                except Exception:
                    pass
        # forward CARLA camera -> injector at camera_hz. CARLA delivers BGRA
        # (H,W,4); Apollo wants rgb8 (H,W,3) so drop alpha + BGR->RGB.
        if do_cam and self.inj_cam is not None and input_data is not None and "camera_front" in input_data:
            _, img = input_data["camera_front"]
            if img is not None and img.ndim == 3 and img.shape[2] >= 3:
                rgb = np.ascontiguousarray(img[:, :, [2, 1, 0]], dtype=np.uint8)
                h, w = rgb.shape[0], rgb.shape[1]
                try:
                    # separate socket so the big jpeg frame never blocks lidar
                    self.inj_cam.sendall(b"C" + struct.pack("<d", ts) + struct.pack("<II", h, w) + rgb.tobytes())
                    self._last_cam_wall = ts
                except Exception:
                    pass
            # front_12mm on its own socket/process
            if self.inj_cam12 is not None and "camera_front12" in input_data:
                _, img2 = input_data["camera_front12"]
                if img2 is not None and img2.ndim == 3 and img2.shape[2] >= 3:
                    rgb2 = np.ascontiguousarray(img2[:, :, [2, 1, 0]], dtype=np.uint8)
                    h2, w2 = rgb2.shape[0], rgb2.shape[1]
                    try:
                        self.inj_cam12.sendall(b"C" + struct.pack("<d", ts) + struct.pack("<II", h2, w2) + rgb2.tobytes())
                    except Exception:
                        pass
        with self._lock:
            return self._latest_ctrl, {}

    def destroy(self):
        self._running = False
        th = getattr(self, "_ctl_thread", None)
        if th:
            try: th.join(timeout=2.0)
            except Exception: pass
        try: self.inj.close()
        except Exception: pass
        try:
            if self.inj_cam is not None: self.inj_cam.close()
        except Exception: pass
        try:
            if self.inj_cam12 is not None: self.inj_cam12.close()
        except Exception: pass
        try: self.b.conn.close(); self.b.stop()
        except Exception: pass
        logger.info(f"[ApolloReal {self.id}] destroyed")
