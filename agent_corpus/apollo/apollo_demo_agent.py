"""ApolloDemoAgent — Drivora ADS that drives via a *running* Apollo on Town01.

This is the "wire it all up" agent: it plugs Baidu Apollo into Drivora's
scenario_runner so the full pipeline (Drivora CARLA scenario → Apollo brain →
control → Drivora video/oracle) works end-to-end.

For this integration it uses the proven recipe from the standalone harness:
  - connect to an ALREADY-RUNNING Apollo bridge (config: apollo_host/bridge_port)
    on the converted Town01 map — does NOT start its own container;
  - inject localization from the ego's own pose (option-1, robust in real time);
  - publish an empty /apollo/perception/obstacles to unblock prediction→planning
    (real lidar perception is a separate WIP, blocked by cyber_bridge bulk limits);
  - route along the nearest routing-graph lane to the ego's spawn.

The full-sensor design lives in apollo_proxy_agent.py; this focuses on a working
closed loop through Drivora.
"""
import os
import math
import time
import json
import threading

import numpy as np
import carla
from loguru import logger

from agent_corpus.atomic.base_agent import AutonomousAgent
from .transform import CarlaApolloTransform
from .bridge.cyber_bridge import CyberBridge
from .bridge import PUBLISHER_REGISTRY, SUBSCRIBER_REGISTRY
from .bridge import format as F
import agent_corpus.apollo.bridge.publishers  # noqa: F401 (register)
import agent_corpus.apollo.bridge.subscribers  # noqa: F401
import importlib

_topo = importlib.import_module("apollo_modules.modules.routing.proto.topo_graph_pb2")
_hdr = importlib.import_module("apollo_modules.modules.common.proto.header_pb2")
_perc = importlib.import_module("apollo_modules.modules.perception.proto.perception_obstacle_pb2")

_ROUTING_BIN = os.path.join(os.path.dirname(__file__), "map", "Town01", "routing_map.bin")


def get_entry_point():
    return "ApolloDemoAgent"


def _graph_lanes():
    g = _topo.Graph()
    g.ParseFromString(open(_ROUTING_BIN, "rb").read())
    lanes = []
    for n in g.node:
        pts = [(p.x, p.y) for s in n.central_curve.segment if s.HasField("line_segment")
               for p in s.line_segment.point]
        if len(pts) >= 2:
            lanes.append({"id": n.lane_id, "len": n.length or 0.0, "pts": pts})
    return lanes


class ApolloDemoAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        with open(path_to_conf_file, "r") as f:
            cfg = json.load(f)
        self._host = cfg.get("apollo_host", "172.17.0.3")
        self._port = cfg.get("bridge_port", 9090)
        self.tf = CarlaApolloTransform(cfg.get("calibration", {"flip_y": True}))
        self._lanes = _graph_lanes()

        self._lock = threading.Lock()
        self._latest_control = carla.VehicleControl(throttle=0.0, brake=1.0)
        self._route_msg = None
        self._ready = False
        self._route_sent_wall = 0.0
        self._running = True

        # connect to the already-running Apollo bridge (no container management)
        self.bridge = CyberBridge(self._host, self._port)
        self.loc_pub = PUBLISHER_REGISTRY.get("publisher.localization")(idx="l", bridge=self.bridge)
        self.ch_pub = PUBLISHER_REGISTRY.get("publisher.chassis")(idx="c", bridge=self.bridge)
        self.pad_pub = PUBLISHER_REGISTRY.get("publisher.control_pad")(idx="p", bridge=self.bridge)
        self.rt_pub = PUBLISHER_REGISTRY.get("publisher.routing_request")(idx="r", bridge=self.bridge)
        self.ctl_sub = SUBSCRIBER_REGISTRY.get("subscriber.control")(idx="s", bridge=self.bridge)
        self.bridge.add_publisher("/apollo/perception/obstacles", "apollo.perception.PerceptionObstacles")
        self._pseq = 0
        self.bridge.spin()

        self._pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._ctl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._pub_thread.start()
        self._ctl_thread.start()
        logger.info(f"[ApolloDemo {self.id}] connected to Apollo {self._host}:{self._port}")

    def sensors(self):
        # localization is injected from the ego pose; no sensors needed for the
        # decision loop. A speedometer is handy but optional.
        return [{"type": "sensor.speedometer", "id": "speed"}]

    def _build_route(self):
        """Snap the ego's current apollo position to the nearest routing-graph
        lane and route to its far end."""
        t = self.carla_actor.get_transform()
        mx, my, _ = self.tf.loc_carla_to_apollo(t.location.x, t.location.y, t.location.z)
        best, bestd = None, 1e9
        for ln in self._lanes:
            for (px, py) in ln["pts"]:
                d = (px - mx) ** 2 + (py - my) ** 2
                if d < bestd:
                    bestd, best = d, ln
        if best is None:
            return None
        p0 = best["pts"][0]
        p1 = best["pts"][1]
        pe = best["pts"][-1]
        heading = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        return F.RouteMessage(timestamp=0.0, waypoints=[
            F.Waypoint(F.Lane(best["id"], 0.0, 0.0),
                       F.Location(p0[0], p0[1], 0, 0, heading, 0)),
            F.Waypoint(F.Lane(best["id"], max(1.0, best["len"] - 2.0), 0.0),
                       F.Location(pe[0], pe[1], 0, 0, heading, 0)),
        ])

    def _publish_loop(self):
        i = 0
        while self._running:
            try:
                if self.carla_actor is None:
                    time.sleep(0.02)
                    continue
                if self._route_msg is None:
                    self._route_msg = self._build_route()
                t = self.carla_actor.get_transform()
                v = self.carla_actor.get_velocity()
                ts = time.time()
                mx, my, _ = self.tf.loc_carla_to_apollo(t.location.x, t.location.y, t.location.z)
                h = self.tf.yaw_carla_to_apollo(t.rotation.yaw)
                with self._lock:
                    last = self._latest_control
                self.loc_pub.publish(F.LocalizationMessage(ts, F.Location(mx, my, 0, 0, h, 0), h,
                                     self.tf.vec_carla_to_apollo(v.x, v.y, v.z),
                                     F.Vector(0, 0, 0), F.Vector(0, 0, 0)))
                self.ch_pub.publish(F.ChassisMessage(ts,
                                    math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z),
                                    last.throttle*100, last.brake*100, last.steer*100, last.reverse))
                if i % 10 == 0:  # empty perception @ ~10Hz unblocks prediction->planning
                    po = _perc.PerceptionObstacles(header=_hdr.Header(
                        timestamp_sec=ts, module_name="drivora", sequence_num=self._pseq))
                    self._pseq += 1
                    self.bridge.publish("/apollo/perception/obstacles", po.SerializeToString())
                # routing FSM: (re)send until ego confirmed moving
                if not self._ready and self._route_msg is not None:
                    spd = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
                    if spd > 0.3:
                        self._ready = True
                    elif ts - self._route_sent_wall > 2.0:
                        self._route_msg.timestamp = ts
                        self.rt_pub.last_publish_time = None
                        self.rt_pub.publish(self._route_msg)
                        self.pad_pub.last_publish_time = None
                        self.pad_pub.publish(F.ControlPadMessage(ts, 1))
                        self._route_sent_wall = ts
            except Exception:
                logger.opt(exception=True).warning(f"[ApolloDemo {self.id}] publish error")
            i += 1
            time.sleep(0.01)

    def _control_loop(self):
        while self._running:
            try:
                t, b, s, r = self.ctl_sub.get_data()
                if self.ctl_sub.control_data is not None:
                    with self._lock:
                        self._latest_control = self.tf.apollo_control_to_carla(t, b, s, r)
            except Exception:
                pass
            time.sleep(0.01)

    def run_step(self, input_data, timestamp):
        with self._lock:
            return self._latest_control, {}

    def destroy(self):
        self._running = False
        for th in (getattr(self, "_pub_thread", None), getattr(self, "_ctl_thread", None)):
            if th:
                try:
                    th.join(timeout=2.0)
                except Exception:
                    pass
        try:
            self.bridge.conn.close()
            self.bridge.stop()
        except Exception:
            pass
        logger.info(f"[ApolloDemo {self.id}] destroyed")
