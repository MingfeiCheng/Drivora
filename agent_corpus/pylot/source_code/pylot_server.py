"""
PylotServer — Container-side ZMQ server for Drivora.

Receives sensor data (cameras, LiDAR, IMU, GNSS, speed) from the host-side
PylotProxyAgent via ZMQ REP, feeds them into the ERDOS dataflow pipeline,
and returns the computed vehicle control.

Protocol (msgpack over ZMQ REQ/REP):
  → { cmd: "ping" }
  ← { status: "ok" }

  → { cmd: "get_sensors" }
  ← { status: "ok", sensors: [...] }

  → { cmd: "init", opendrive: str, route: [...], vehicle_id: int }
  ← { status: "ok" }

  → { cmd: "tick", timestamp: int, sensors: { cameras: {name: jpeg_bytes}, lidar: ndarray, imu: [...], gnss: [...], speed: {speed: float} } }
  ← { status: "ok", control: { throttle, steer, brake, hand_brake, reverse } }

  → { cmd: "destroy" }
  ← { status: "ok" }
"""

import os
import sys
import time
import logging
import argparse
import traceback
from collections import deque

import zmq
import numpy as np
import msgpack
import msgpack_numpy as m
m.patch()

# cv2 is imported from the base image's existing opencv install.
# Do NOT install opencv-python-headless on top — it conflicts.
import cv2

from loguru import logger
from absl import flags

import erdos

import pylot.flags
import pylot.component_creator
import pylot.operator_creator
import pylot.perception.messages
import pylot.utils
from pylot.drivers.sensor_setup import LidarSetup, RGBCameraSetup
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.point_cloud import PointCloud
from pylot.localization.messages import GNSSMessage, IMUMessage
from pylot.planning.messages import WaypointsMessage
from pylot.planning.waypoints import Waypoints

FLAGS = flags.FLAGS

# Flag defined in ERDOSAgent.py but not in pylot/flags.py — define it here.
flags.DEFINE_bool(
    'perfect_localization', False,
    'Set to True to receive ego-vehicle locations from the simulator')

# ── Camera constants (must match create_camera_setups) ──────────────────
CENTER_CAMERA_LOCATION = pylot.utils.Location(0.0, 0.0, 2.0)
CENTER_CAMERA_NAME = 'center_camera'
LANE_CAMERA_LOCATION = pylot.utils.Location(1.3, 0.0, 1.8)
LANE_CAMERA_NAME = 'lane_camera'
TL_CAMERA_NAME = 'traffic_lights_camera'


# ═══════════════════════════════════════════════════════════════════════════
#  ERDOS pipeline construction (standalone, no CARLA dependency)
# ═══════════════════════════════════════════════════════════════════════════

def create_camera_setups():
    camera_setups = {}
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION, pylot.utils.Rotation())
    center_camera_setup = RGBCameraSetup(
        CENTER_CAMERA_NAME, FLAGS.camera_image_width, FLAGS.camera_image_height, transform, 90)
    camera_setups[CENTER_CAMERA_NAME] = center_camera_setup

    if not FLAGS.simulator_traffic_light_detection:
        tl_camera_setup = RGBCameraSetup(
            TL_CAMERA_NAME, FLAGS.camera_image_width, FLAGS.camera_image_height, transform, 45)
        camera_setups[TL_CAMERA_NAME] = tl_camera_setup

    if FLAGS.execution_mode == 'challenge-sensors':
        lane_transform = pylot.utils.Transform(LANE_CAMERA_LOCATION, pylot.utils.Rotation(pitch=-15))
        lane_camera_setup = RGBCameraSetup(LANE_CAMERA_NAME, 1280, 720, lane_transform, 90)
        camera_setups[LANE_CAMERA_NAME] = lane_camera_setup

    return camera_setups


def create_lidar_setup():
    lidar_transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION, pylot.utils.Rotation())
    return LidarSetup('lidar', 'sensor.lidar.ray_cast', lidar_transform, range=8500, legacy=False)


def using_lidar():
    return not (FLAGS.simulator_obstacle_detection and FLAGS.simulator_traffic_light_detection)


def create_data_flow():
    """Creates the ERDOS dataflow graph — identical to agent.py but standalone."""
    streams_to_send_top_on = []
    camera_setups = create_camera_setups()
    camera_streams = {name: erdos.IngestStream() for name in camera_setups}

    global_trajectory_stream = erdos.IngestStream()
    open_drive_stream = erdos.IngestStream()
    point_cloud_stream = erdos.IngestStream()
    imu_stream = erdos.IngestStream()
    gnss_stream = erdos.IngestStream()
    route_stream = erdos.IngestStream()
    time_to_decision_loop_stream = erdos.LoopStream()

    if FLAGS.localization:
        pose_stream = pylot.operator_creator.add_localization(imu_stream, gnss_stream, route_stream)
    else:
        pose_stream = erdos.IngestStream()

    # Obstacle detection
    perfect_obstacles_stream = erdos.IngestStream()
    if FLAGS.simulator_obstacle_detection:
        obstacles_stream = perfect_obstacles_stream
    elif any('efficientdet' in model_name for model_name in FLAGS.obstacle_detection_model_names):
        obstacles_stream = pylot.operator_creator.add_efficientdet_obstacle_detection(
            camera_streams[CENTER_CAMERA_NAME], time_to_decision_loop_stream)[0]
        if not (FLAGS.evaluate_obstacle_detection or FLAGS.evaluate_obstacle_tracking):
            streams_to_send_top_on.append(perfect_obstacles_stream)
    else:
        obstacles_stream = pylot.operator_creator.add_obstacle_detection(
            camera_streams[CENTER_CAMERA_NAME], time_to_decision_loop_stream)[0]
        if not (FLAGS.evaluate_obstacle_detection or FLAGS.evaluate_obstacle_tracking):
            streams_to_send_top_on.append(perfect_obstacles_stream)

    # Traffic light detection
    perfect_traffic_lights_stream = erdos.IngestStream()
    if FLAGS.simulator_traffic_light_detection:
        traffic_lights_stream = perfect_traffic_lights_stream
        camera_streams[TL_CAMERA_NAME] = erdos.IngestStream()
        streams_to_send_top_on.append(camera_streams[TL_CAMERA_NAME])
    else:
        traffic_lights_stream = pylot.operator_creator.add_traffic_light_detector(
            camera_streams[TL_CAMERA_NAME], time_to_decision_loop_stream)
        traffic_lights_stream = pylot.operator_creator.add_obstacle_location_finder(
            traffic_lights_stream, point_cloud_stream, pose_stream, camera_setups[TL_CAMERA_NAME])
        streams_to_send_top_on.append(perfect_traffic_lights_stream)

    vehicle_id_stream = erdos.IngestStream()
    if not (FLAGS.perfect_obstacle_tracking or FLAGS.perfect_localization):
        streams_to_send_top_on.append(vehicle_id_stream)

    # Tracking
    obstacles_tracking_stream = pylot.component_creator.add_obstacle_tracking(
        camera_streams[CENTER_CAMERA_NAME], camera_setups[CENTER_CAMERA_NAME],
        obstacles_stream, depth_stream=point_cloud_stream,
        vehicle_id_stream=vehicle_id_stream, pose_stream=pose_stream,
        ground_obstacles_stream=perfect_obstacles_stream,
        time_to_decision_stream=time_to_decision_loop_stream)

    # Lanes
    if FLAGS.execution_mode == 'challenge-sensors':
        lanes_stream = pylot.operator_creator.add_lanenet_detection(camera_streams[LANE_CAMERA_NAME])
    else:
        lanes_stream = erdos.IngestStream()
        streams_to_send_top_on.append(lanes_stream)

    # Prediction
    prediction_stream, _, _ = pylot.component_creator.add_prediction(
        obstacles_tracking_stream, vehicle_id_stream,
        time_to_decision_loop_stream, pose_stream=pose_stream)

    # Planning
    waypoints_stream = pylot.component_creator.add_planning(
        None, pose_stream, prediction_stream, traffic_lights_stream,
        lanes_stream, open_drive_stream, global_trajectory_stream,
        time_to_decision_loop_stream)

    # Control
    control_stream = pylot.component_creator.add_control(pose_stream, waypoints_stream)
    extract_control_stream = erdos.ExtractStream(control_stream)

    pylot.component_creator.add_evaluation(vehicle_id_stream, pose_stream, imu_stream)

    time_to_decision_stream = pylot.operator_creator.add_time_to_decision(
        pose_stream, obstacles_stream)
    time_to_decision_loop_stream.set(time_to_decision_stream)

    return (camera_streams, pose_stream, route_stream,
            global_trajectory_stream, open_drive_stream, point_cloud_stream,
            imu_stream, gnss_stream, extract_control_stream,
            perfect_obstacles_stream, perfect_traffic_lights_stream,
            vehicle_id_stream, streams_to_send_top_on)


# ═══════════════════════════════════════════════════════════════════════════
#  PylotServer
# ═══════════════════════════════════════════════════════════════════════════

class PylotServer:

    def __init__(self, config_path: str, zmq_port: int = 12667):
        self._zmq_port = zmq_port
        self._last_yaw = 0
        self._last_point_cloud = None

        # ── Parse pylot flags ──
        pylot.utils.set_tf_loglevel(logging.ERROR)
        flags.FLAGS([__file__, f'--flagfile={config_path}'])

        # ── Camera / LiDAR setups ──
        self._camera_setups = create_camera_setups()
        self._lidar_setup = create_lidar_setup()

        # ── Build ERDOS dataflow ──
        (self._camera_streams, self._pose_stream, self._route_stream,
         self._global_trajectory_stream, self._open_drive_stream,
         self._point_cloud_stream, self._imu_stream, self._gnss_stream,
         self._control_stream,
         self._perfect_obstacles_stream, self._perfect_traffic_lights_stream,
         self._vehicle_id_stream,
         streams_to_send_top_on) = create_data_flow()

        self._node_handle = erdos.run_async()

        for stream in streams_to_send_top_on:
            stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

        # ── ZMQ REP socket ──
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REP)
        self._socket.bind(f"tcp://0.0.0.0:{zmq_port}")

        logger.info(f"PylotServer ready — listening on tcp://0.0.0.0:{zmq_port}")

    # ── Sensor config query (so proxy knows what sensors to attach) ──────
    def get_sensor_config(self):
        """Return the sensor specs the proxy should create in CARLA."""
        sensors = []

        # Cameras
        for cs in self._camera_setups.values():
            sensors.append({
                'type': cs.camera_type,
                'x': cs.transform.location.x,
                'y': cs.transform.location.y,
                'z': cs.transform.location.z,
                'roll': cs.transform.rotation.roll,
                'pitch': cs.transform.rotation.pitch,
                'yaw': cs.transform.rotation.yaw,
                'width': cs.width,
                'height': cs.height,
                'fov': cs.fov,
                'id': cs.name,
            })

        # LiDAR
        if using_lidar():
            ls = self._lidar_setup
            sensors.append({
                'type': 'sensor.lidar.ray_cast',
                'x': ls.transform.location.x,
                'y': ls.transform.location.y,
                'z': ls.transform.location.z,
                'roll': ls.transform.rotation.roll,
                'pitch': ls.transform.rotation.pitch,
                'yaw': ls.transform.rotation.yaw,
                'id': 'LIDAR',
            })

        # Pseudo-sensors
        sensors.append({'type': 'sensor.opendrive_map', 'reading_frequency': 20, 'id': 'opendrive'})
        sensors.append({'type': 'sensor.other.gnss', 'x': 0, 'y': 0, 'z': 0, 'id': 'gnss'})
        sensors.append({
            'type': 'sensor.other.imu', 'x': 0, 'y': 0, 'z': 0,
            'roll': 0, 'pitch': 0, 'yaw': 0, 'id': 'imu',
        })
        sensors.append({'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'speed'})

        return sensors

    # ── Command handlers ─────────────────────────────────────────────────

    def _handle_get_sensors(self):
        return {'status': 'ok', 'sensors': self.get_sensor_config()}

    def _handle_init(self, data):
        ts = erdos.Timestamp(coordinates=[0])

        # OpenDrive map (sent once)
        if 'opendrive' in data and not self._open_drive_stream.is_closed():
            opendrive_str = data['opendrive']
            logger.info(f"Received opendrive: {len(opendrive_str)} chars, starts with: {opendrive_str[:100]}...")

            # Validate: can carla.Map parse this?
            try:
                from carla import Map as CarlaMap
                test_map = CarlaMap('test', opendrive_str)
                logger.info(f"OpenDrive parsed OK — map name: {test_map.name}")
            except Exception as e:
                logger.error(f"OpenDrive parsing FAILED: {e}")
                return {'status': 'error', 'message': f'OpenDrive parse error: {e}'}

            self._open_drive_stream.send(erdos.Message(ts, opendrive_str))
            self._open_drive_stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

        # Route (sent once)
        if 'route' in data and not self._global_trajectory_stream.is_closed():
            waypoints = deque()
            road_options = deque()
            for wp in data['route']:
                loc = pylot.utils.Location(wp['x'], wp['y'], wp['z'])
                rot = pylot.utils.Rotation(pitch=wp.get('pitch', 0),
                                           yaw=wp.get('yaw', 0),
                                           roll=wp.get('roll', 0))
                waypoints.append(pylot.utils.Transform(loc, rot))
                road_options.append(pylot.utils.RoadOption(wp['road_option']))
            self._global_trajectory_stream.send(
                WaypointsMessage(ts, Waypoints(waypoints, road_options=road_options)))
            self._global_trajectory_stream.send(
                erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

        # Vehicle ID (for perfect tracking, optional)
        if 'vehicle_id' in data and not self._vehicle_id_stream.is_closed():
            self._vehicle_id_stream.send(erdos.Message(ts, data['vehicle_id']))
            self._vehicle_id_stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

        logger.info("Init complete — received opendrive + route.")
        return {'status': 'ok'}

    def _handle_tick(self, data):
        game_time = data['timestamp']
        ts = erdos.Timestamp(coordinates=[game_time])
        sensors = data['sensors']

        # ── 1. Cameras (JPEG-encoded) ──
        for cam_name, jpeg_bytes in sensors.get('cameras', {}).items():
            if cam_name not in self._camera_streams:
                continue
            img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            frame = CameraFrame(img, 'BGR', self._camera_setups[cam_name])
            self._camera_streams[cam_name].send(
                pylot.perception.messages.FrameMessage(ts, frame))
            self._camera_streams[cam_name].send(erdos.WatermarkMessage(ts))

        # ── 2. LiDAR ──
        if 'lidar' in sensors and sensors['lidar'] is not None:
            pc_data = np.asarray(sensors['lidar'], dtype=np.float32)[:, :3]
            point_cloud = PointCloud(pc_data, self._lidar_setup)
            if self._last_point_cloud is not None:
                point_cloud.merge(self._last_point_cloud)
            self._point_cloud_stream.send(
                pylot.perception.messages.PointCloudMessage(ts, point_cloud))
            self._point_cloud_stream.send(erdos.WatermarkMessage(ts))
            self._last_point_cloud = PointCloud(pc_data, self._lidar_setup)

        # ── 3. Pose (computed from speed / IMU / GNSS) ──
        speed_data = sensors.get('speed')
        imu_data = sensors.get('imu')
        gnss_data = sensors.get('gnss')

        if not FLAGS.localization:
            pose = self._compute_pose(speed_data, imu_data, gnss_data, ts)
            self._pose_stream.send(erdos.Message(ts, pose))
            self._pose_stream.send(erdos.WatermarkMessage(ts))
        else:
            # Localization operator computes pose from IMU + GNSS
            if imu_data is not None:
                acc = pylot.utils.Vector3D(imu_data[0], imu_data[1], imu_data[2])
                gyro = pylot.utils.Vector3D(imu_data[3], imu_data[4], imu_data[5])
                compass = imu_data[6]
                self._imu_stream.send(IMUMessage(ts, None, acc, gyro, compass))
                self._imu_stream.send(erdos.WatermarkMessage(ts))
            if gnss_data is not None:
                lat, lon, alt = gnss_data[0], gnss_data[1], gnss_data[2]
                location = pylot.utils.Location.from_gps(lat, lon, alt)
                transform = pylot.utils.Transform(location, pylot.utils.Rotation())
                self._gnss_stream.send(GNSSMessage(ts, transform, alt, lat, lon))
                self._gnss_stream.send(erdos.WatermarkMessage(ts))
            # Send noisy pose on route stream for EKF initialisation
            pose = self._compute_pose(speed_data, imu_data, gnss_data, ts)
            self._route_stream.send(erdos.Message(ts, pose))
            self._route_stream.send(erdos.WatermarkMessage(ts))

        # ── 4. Read control from ERDOS ──
        control_msg = self._read_control()
        control = {
            'throttle':   control_msg.throttle   if control_msg else 0.0,
            'steer':      control_msg.steer      if control_msg else 0.0,
            'brake':      control_msg.brake      if control_msg else 1.0,
            'hand_brake': control_msg.hand_brake if control_msg else False,
            'reverse':    control_msg.reverse    if control_msg else False,
        }
        return {'status': 'ok', 'control': control}

    def _handle_destroy(self):
        """Tear down and rebuild the ERDOS pipeline for the next scenario.

        CUDA and TF are already warm so this is much faster than cold start.
        The ZMQ server loop stays alive — no container restart needed.
        """
        logger.info("Destroying ERDOS pipeline for rebuild...")
        self._node_handle.shutdown()
        erdos.reset()
        self._last_yaw = 0
        self._last_point_cloud = None

        # Rebuild pipeline
        logger.info("Rebuilding ERDOS pipeline...")
        (self._camera_streams, self._pose_stream, self._route_stream,
         self._global_trajectory_stream, self._open_drive_stream,
         self._point_cloud_stream, self._imu_stream, self._gnss_stream,
         self._control_stream,
         self._perfect_obstacles_stream, self._perfect_traffic_lights_stream,
         self._vehicle_id_stream,
         streams_to_send_top_on) = create_data_flow()

        self._node_handle = erdos.run_async()
        for stream in streams_to_send_top_on:
            stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

        logger.info("ERDOS pipeline rebuilt — ready for next scenario.")
        return {'status': 'ok'}

    # ── Helpers ───────────────────────────────────────────────────────────

    def _compute_pose(self, speed_data, imu_data, gnss_data, timestamp):
        forward_speed = speed_data.get('speed', 0.0) if speed_data else 0.0
        if gnss_data is not None:
            lat, lon, alt = gnss_data[0], gnss_data[1], gnss_data[2]
            location = pylot.utils.Location.from_gps(lat, lon, alt)
        else:
            location = pylot.utils.Location(0, 0, 0)

        if imu_data is not None and not np.isnan(imu_data[6]):
            compass = np.degrees(imu_data[6])
            yaw = compass - 90 if compass < 270 else compass - 450
            self._last_yaw = yaw
        else:
            yaw = self._last_yaw

        transform = pylot.utils.Transform(location, pylot.utils.Rotation(yaw=yaw))
        velocity = pylot.utils.Vector3D(
            forward_speed * np.cos(np.radians(yaw)),
            forward_speed * np.sin(np.radians(yaw)), 0)
        return pylot.utils.Pose(transform, forward_speed, velocity, timestamp.coordinates[0])

    def _read_control(self, max_tries=200):
        for _ in range(max_tries):
            msg = self._control_stream.try_read()
            if msg is not None and not isinstance(msg, erdos.WatermarkMessage):
                return msg
        logger.warning("Control read timed out — returning None")
        return None

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        logger.info("PylotServer entering main loop...")
        while True:
            try:
                raw = self._socket.recv()
                msg = msgpack.unpackb(raw, raw=False, object_hook=m.decode)
                cmd = msg.get('cmd')

                if cmd == 'ping':
                    resp = {'status': 'ok'}
                elif cmd == 'get_sensors':
                    resp = self._handle_get_sensors()
                elif cmd == 'init':
                    resp = self._handle_init(msg)
                elif cmd == 'tick':
                    resp = self._handle_tick(msg)
                elif cmd == 'destroy':
                    resp = self._handle_destroy()
                elif cmd == 'shutdown':
                    # Hard shutdown — actually exits the server
                    logger.info("Shutdown requested — exiting.")
                    resp = {'status': 'ok'}
                    self._socket.send(msgpack.packb(resp, default=m.encode))
                    self._node_handle.shutdown()
                    erdos.reset()
                    break
                else:
                    resp = {'status': 'error', 'message': f'Unknown command: {cmd}'}

                self._socket.send(msgpack.packb(resp, default=m.encode))

            except Exception as e:
                logger.error(f"Server error: {e}")
                traceback.print_exc()
                try:
                    self._socket.send(msgpack.packb(
                        {'status': 'error', 'message': str(e)}, default=m.encode))
                except Exception:
                    pass

        logger.info("PylotServer shut down.")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pylot ERDOS Server')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to pylot flagfile (e.g. configs/challenge_map.conf)')
    parser.add_argument('--port', type=int, default=12667,
                        help='ZMQ REP port')
    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, 'pylot_server.log'), level='DEBUG',
               rotation='10 MB', retention='5 days')

    server = PylotServer(config_path=args.config, zmq_port=args.port)
    server.run()
