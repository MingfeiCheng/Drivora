"""Localization publisher → /apollo/localization/pose.

OPTIONAL / DEBUG capability. The production design (option 2) does NOT inject
pose — Apollo's Localization module computes it from GNSS/IMU. This publisher
exists to (a) de-risk bring-up by isolating routing/planning/control from the
GNSS-localization step, and (b) support a "perfect localization" mode if ever
wanted. It is not in the default publisher set.

Input: an injected pose in the Apollo map frame (LocalizationMessage).
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation

from apollo_modules.modules.common.proto.header_pb2 import Header
from apollo_modules.modules.common.proto.geometry_pb2 import Point3D, PointENU, Quaternion
from apollo_modules.modules.localization.proto.localization_pb2 import LocalizationEstimate
from apollo_modules.modules.localization.proto.pose_pb2 import Pose

from .base import Publisher
from ..registry import PUBLISHER_REGISTRY


def _inv_quat_rotate(q: Quaternion, v: np.ndarray) -> np.ndarray:
    rot = Rotation.from_quat([q.qx, q.qy, q.qz, q.qw])
    return rot.inv().as_matrix().dot(v)


@PUBLISHER_REGISTRY.register("publisher.localization")
class LocalizationPublisher(Publisher):
    channel = "/apollo/localization/pose"
    msg_type = "apollo.localization.LocalizationEstimate"
    msg_cls = LocalizationEstimate
    frequency = 100.0

    def _process_data(self, message):
        loc = message.location
        position = PointENU(x=loc.x, y=loc.y, z=loc.z)
        heading = message.heading
        adjusted = (heading - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
        qx, qy, qz, qw = Rotation.from_euler("z", adjusted, degrees=False).as_quat()
        orientation = Quaternion(qx=qx, qy=qy, qz=qz, qw=qw)

        lin_acc = Point3D(x=message.acceleration.x, y=message.acceleration.y, z=message.acceleration.z)
        ang_vel = Point3D(x=message.angular_velocity.x, y=message.angular_velocity.y,
                          z=message.angular_velocity.z)
        acc_vrf = _inv_quat_rotate(orientation, np.array([lin_acc.x, lin_acc.y, lin_acc.z]))
        ang_vrf = _inv_quat_rotate(orientation, np.array([ang_vel.x, ang_vel.y, ang_vel.z]))

        return LocalizationEstimate(
            header=Header(timestamp_sec=message.timestamp, module_name="drivora",
                          sequence_num=self.frame_count),
            pose=Pose(
                position=position,
                heading=heading,
                orientation=orientation,
                linear_velocity=Point3D(x=message.velocity.x, y=message.velocity.y, z=message.velocity.z),
                linear_acceleration=lin_acc,
                angular_velocity=ang_vel,
                linear_acceleration_vrf=Point3D(x=acc_vrf[0], y=acc_vrf[1], z=acc_vrf[2]),
                angular_velocity_vrf=Point3D(x=ang_vrf[0], y=ang_vrf[1], z=ang_vrf[2]),
            ),
        )
