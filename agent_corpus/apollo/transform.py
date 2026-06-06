"""CARLA ↔ Apollo coordinate / sensor conversions.

Two coordinate conventions are bridged:

  CARLA world : left-handed, metres, yaw in degrees, clockwise-positive.
  Apollo map  : right-handed (ENU-like), metres, heading in radians,
                counter-clockwise-positive, origin tied to the HD map's UTM zone.

The fixed rigid transform between the two depends entirely on how the HD map
was generated from the CARLA town (see docs/apollo_integration_plan.md, 难点
A/B). The defaults below implement the common ``y-flip`` convention but MUST be
re-calibrated against the actual generated map on a straight road before any
closed-loop run — this is the most error-prone part of the integration.

All ``build_*`` helpers return the bridge-internal dataclasses from
``bridge.format`` (not Apollo protobufs).
"""
import math
import numpy as np

from .bridge.format import (
    Vector, Location, Lane, Waypoint, RouteMessage,
    ChassisMessage, CameraMessage, LidarMessage, IMUMessage, GNSSMessage,
)


class CarlaApolloTransform:
    def __init__(self, calib: dict = None):
        """``calib`` overrides the default convention. Keys (all optional):
        flip_y(bool), yaw_offset_deg(float), map_origin_x/y(float),
        utm_origin_x/y(float), utm_zone(int).
        """
        calib = calib or {}
        self.flip_y = calib.get("flip_y", True)
        self.yaw_offset = math.radians(calib.get("yaw_offset_deg", 0.0))
        # CARLA-map → Apollo-map translation (metres). Determined during map gen.
        self.map_origin_x = calib.get("map_origin_x", 0.0)
        self.map_origin_y = calib.get("map_origin_y", 0.0)
        # HD-map UTM origin, used to turn map-frame xy into the lat/lon Apollo's
        # gnss driver carries (and back). Set from the generated map.
        self.utm_origin_x = calib.get("utm_origin_x", 0.0)
        self.utm_origin_y = calib.get("utm_origin_y", 0.0)

    # ---- primitive conversions ----
    def loc_carla_to_apollo(self, x, y, z):
        ay = -y if self.flip_y else y
        return (x + self.map_origin_x, ay + self.map_origin_y, z)

    def yaw_carla_to_apollo(self, yaw_deg: float) -> float:
        """CARLA yaw (deg, CW) → Apollo heading (rad, CCW)."""
        h = -math.radians(yaw_deg) if self.flip_y else math.radians(yaw_deg)
        h += self.yaw_offset
        return (h + math.pi) % (2 * math.pi) - math.pi

    def vec_carla_to_apollo(self, vx, vy, vz):
        return Vector(x=vx, y=(-vy if self.flip_y else vy), z=vz)

    # ---- ego state ----
    def build_chassis(self, speed_mps: float, last_control, ts: float) -> ChassisMessage:
        return ChassisMessage(
            timestamp=ts,
            speed_mps=float(speed_mps),
            throttle_percentage=float(getattr(last_control, "throttle", 0.0)) * 100.0,
            brake_percentage=float(getattr(last_control, "brake", 0.0)) * 100.0,
            steering_percentage=float(getattr(last_control, "steer", 0.0)) * 100.0,
            reverse=bool(getattr(last_control, "reverse", False)),
        )

    # ---- sensors ----
    def build_camera(self, camera_name: str, image: np.ndarray, ts: float) -> CameraMessage:
        # Drivora delivers RGBA (CARLA) or RGB; strip alpha, keep RGB order.
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        return CameraMessage(timestamp=ts, camera_name=camera_name,
                             image=np.ascontiguousarray(image, dtype=np.uint8))

    def build_lidar(self, points: np.ndarray, ts: float, lidar_name: str = "lidar128") -> LidarMessage:
        """``points`` is CARLA Nx4 (x, y, z, intensity). Flip y into Apollo frame."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 4).copy()
        if self.flip_y:
            pts[:, 1] = -pts[:, 1]
        return LidarMessage(timestamp=ts, points=pts, lidar_name=lidar_name)

    def build_imu(self, imu_data, ts: float) -> IMUMessage:
        """CARLA IMU sensor array: [ax, ay, az, gx, gy, gz, compass]."""
        a = np.asarray(imu_data, dtype=np.float64).ravel()
        ax, ay, az, gx, gy, gz = a[0], a[1], a[2], a[3], a[4], a[5]
        compass = a[6] if a.size > 6 else 0.0
        sgn = -1.0 if self.flip_y else 1.0
        return IMUMessage(
            timestamp=ts,
            linear_acceleration=Vector(x=ax, y=sgn * ay, z=az),
            angular_velocity=Vector(x=gx, y=sgn * gy, z=gz),
            # CARLA compass: rad clockwise from north; convert to map heading.
            euler_angles=Vector(x=0.0, y=0.0, z=self._compass_to_heading(compass)),
        )

    def _compass_to_heading(self, compass_rad: float) -> float:
        h = (math.pi / 2.0) - compass_rad  # north-CW → east-CCW (approx)
        return (h + math.pi) % (2 * math.pi) - math.pi

    def build_gnss(self, gnss_data, imu_data, ego_transform, speed_vec, ts: float) -> GNSSMessage:
        """Build the GNSS message for Apollo localization.

        ``gnss_data`` is CARLA's [lat, lon, alt]. We also carry the map-frame
        pose derived from the ego transform so the gnss publisher can emit
        odometry without re-projecting. (Using the ego transform here is *not*
        ground-truth perception — it is the simulated ego pose that a real GNSS
        receiver would observe; localization still runs in Apollo.)
        """
        g = np.asarray(gnss_data, dtype=np.float64).ravel()
        lat, lon, alt = (g[0], g[1], g[2]) if g.size >= 3 else (0.0, 0.0, 0.0)

        loc = ego_transform.location
        rot = ego_transform.rotation
        mx, my, mz = self.loc_carla_to_apollo(loc.x, loc.y, loc.z)
        heading = self.yaw_carla_to_apollo(rot.yaw)
        vel = self.vec_carla_to_apollo(speed_vec.x, speed_vec.y, speed_vec.z)

        return GNSSMessage(
            timestamp=ts, latitude=lat, longitude=lon, altitude=alt,
            map_x=mx, map_y=my, map_z=mz, heading=heading, linear_velocity=vel,
        )

    # ---- route ----
    def build_route(self, world_waypoints, ts: float) -> RouteMessage:
        """``world_waypoints`` is Drivora's global plan: list of
        (carla.Transform, road_option). Lane id/s are left empty → Apollo routing
        snaps the (x, y, heading) pose to the nearest lane on the HD map.
        """
        wps = []
        for transform, _road_option in world_waypoints:
            loc = transform.location
            rot = transform.rotation
            mx, my, mz = self.loc_carla_to_apollo(loc.x, loc.y, loc.z)
            wps.append(Waypoint(
                lane=Lane(id="", s=0.0, l=0.0),
                location=Location(x=mx, y=my, z=mz, pitch=rot.pitch,
                                  yaw=self.yaw_carla_to_apollo(rot.yaw), roll=rot.roll),
            ))
        return RouteMessage(timestamp=ts, waypoints=wps)

    # ---- control (Apollo → CARLA) ----
    @staticmethod
    def apollo_control_to_carla(throttle, brake, steer, reverse, throttle_gain=1.0):
        """Apollo: throttle/brake in [0,1], steer in [-1,1] (left positive).
        CARLA VehicleControl: steer in [-1,1] (right positive) → negate.
        throttle_gain compensates Apollo's real-MKZ throttle calibration vs
        CARLA's vehicle dynamics: Apollo PLANS full cruise (~11 m/s) but its
        throttle command only holds ~5 m/s on CARLA's mkz, so scale throttle up."""
        import carla
        return carla.VehicleControl(
            throttle=float(max(0.0, min(1.0, throttle * throttle_gain))),
            brake=float(max(0.0, min(1.0, brake))),
            steer=float(max(-1.0, min(1.0, -steer))),  # VERIFY sign on real map
            reverse=bool(reverse),
            hand_brake=False,
            manual_gear_shift=False,
        )
