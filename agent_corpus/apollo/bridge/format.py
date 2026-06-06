"""Intermediate message dataclasses passed from the agent to publishers.

These are *bridge-internal* containers (not Apollo protobufs). Each publisher's
``_process_data`` converts one of these into the corresponding Apollo proto.

Every message MUST expose a ``timestamp`` attribute (seconds, float): the base
Publisher uses it for frequency limiting.

Sensor messages (Camera/Lidar/IMU/GNSS) are new for Drivora's real-sensor
perception path and have no equivalent in ground-truth-perception setups (which
inject perfect perception).
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Vector:
    x: float
    y: float
    z: float


@dataclass
class Lane:
    id: str
    s: float
    l: float


@dataclass
class Location:
    x: float
    y: float
    z: float
    pitch: float
    yaw: float  # heading
    roll: float


@dataclass
class Waypoint:
    lane: Lane
    location: Location


# ----- ego state / control plumbing -----

@dataclass
class ChassisMessage:
    timestamp: float
    speed_mps: float
    throttle_percentage: float  # [0, 100]
    brake_percentage: float
    steering_percentage: float
    reverse: bool


@dataclass
class RouteMessage:
    timestamp: float
    waypoints: List[Waypoint]


@dataclass
class ControlPadMessage:
    # 0 - stop, 1 - start, 2 - reset
    timestamp: float
    action: int


# ----- raw sensor payloads (real-sensor perception) -----

@dataclass
class CameraMessage:
    """One camera frame. ``image`` is HxWx3 uint8 RGB."""
    timestamp: float
    camera_name: str          # e.g. "front_6mm" — must match Apollo channel/calib
    image: np.ndarray
    frame_id: str = "novatel"


@dataclass
class LidarMessage:
    """One lidar sweep. ``points`` is Nx4 float32 (x, y, z, intensity) in the
    Apollo sensor frame (already transformed from CARLA)."""
    timestamp: float
    points: np.ndarray
    lidar_name: str = "lidar128"
    frame_id: str = "novatel"


@dataclass
class IMUMessage:
    timestamp: float
    angular_velocity: Vector       # rad/s, vehicle/IMU frame
    linear_acceleration: Vector    # m/s^2, includes gravity convention per Apollo
    euler_angles: Vector           # roll, pitch, yaw (rad)


@dataclass
class LocalizationMessage:
    """Injected map-frame pose (optional/debug; see publishers/localization.py)."""
    timestamp: float
    location: Location
    heading: float                 # rad, map frame
    velocity: Vector
    acceleration: Vector
    angular_velocity: Vector


@dataclass
class GNSSMessage:
    """GNSS fix fed to Apollo Localization for self-localization (option 2).

    Carries both the geodetic fix and the already-projected map-frame pose so
    publishers can emit whichever Apollo channel they target (best_pose /
    odometry / ins_stat) without re-projecting.
    """
    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    # map-frame (UTM-aligned with the HD map) — filled by the transform layer
    map_x: float = 0.0
    map_y: float = 0.0
    map_z: float = 0.0
    heading: float = 0.0           # rad, map frame
    linear_velocity: Optional[Vector] = field(default=None)
