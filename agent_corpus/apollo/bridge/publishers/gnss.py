"""GNSS publisher → Apollo (RTK) Localization.

Option 2 (fully no-ground-truth localization): instead of injecting
``/apollo/localization/pose`` directly, we feed the inputs that Apollo's RTK
localization consumes and let it compute the pose:

  - ``/apollo/sensor/gnss/odometry``  (apollo.localization.Gps)   — map-frame pose
  - ``/apollo/sensor/gnss/ins_stat``  (apollo.drivers.gnss.InsStat) — fix quality

The map-frame pose (map_x/map_y/heading) is produced by the transform layer by
projecting CARLA GNSS lat/lon onto the HD map's UTM frame. This is the single
highest-risk piece of the whole integration: if the projection/origin/heading
convention is wrong, localization diverges and planning fails.

VERIFY (needs a live Apollo): proto module paths below, the ins_status/pos_type
magic values RTK localization treats as a valid fix, and the orientation
convention of the odometry Pose.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation

from apollo_modules.modules.common.proto.header_pb2 import Header
from apollo_modules.modules.common.proto.geometry_pb2 import PointENU, Point3D, Quaternion
from apollo_modules.modules.localization.proto.gps_pb2 import Gps
from apollo_modules.modules.localization.proto.pose_pb2 import Pose
from apollo_modules.modules.drivers.gnss.proto.ins_pb2 import InsStat

from .base import MultiChannelPublisher
from ..registry import PUBLISHER_REGISTRY

# Well-known "good RTK fix" markers used by sim bridges (DoppelTest/LGSVL).
_INS_STATUS_GOOD = 2     # apollo.drivers.gnss.InsStat.GOOD
_POS_TYPE_NARROW_INT = 56  # NARROW_INT


@PUBLISHER_REGISTRY.register("publisher.gnss")
class GnssPublisher(MultiChannelPublisher):
    channels = [
        ("/apollo/sensor/gnss/odometry", "apollo.localization.Gps"),
        ("/apollo/sensor/gnss/ins_stat", "apollo.drivers.gnss.InsStat"),
    ]
    proto_types = [Gps, InsStat]   # not linked in cyber_bridge → must RegisterDesc
    frequency = 100.0

    def _process_data(self, message) -> dict:
        header = Header(timestamp_sec=message.timestamp, module_name="drivora",
                        sequence_num=self.frame_count)

        # heading (map frame, rad) → quaternion about z, with Apollo's -pi/2 offset
        adjusted = (message.heading - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
        qx, qy, qz, qw = Rotation.from_euler("z", adjusted, degrees=False).as_quat()

        vel = message.linear_velocity
        gps = Gps(
            header=header,
            localization=Pose(
                position=PointENU(x=message.map_x, y=message.map_y, z=message.map_z),
                orientation=Quaternion(qx=qx, qy=qy, qz=qz, qw=qw),
                linear_velocity=Point3D(
                    x=(vel.x if vel else 0.0),
                    y=(vel.y if vel else 0.0),
                    z=(vel.z if vel else 0.0),
                ),
                heading=message.heading,
            ),
        )
        ins = InsStat(header=header, ins_status=_INS_STATUS_GOOD, pos_type=_POS_TYPE_NARROW_INT)
        return {
            "/apollo/sensor/gnss/odometry": gps,
            "/apollo/sensor/gnss/ins_stat": ins,
        }
