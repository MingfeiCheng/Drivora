"""IMU publisher → Apollo Localization.

Publishes ``/apollo/sensor/gnss/corrected_imu`` (apollo.localization.CorrectedImu)
which RTK localization fuses with the GNSS odometry to produce
``/apollo/localization/pose``. We do NOT inject the pose ourselves (option 2:
fully no-ground-truth localization).

VERIFY (needs a live Apollo): CorrectedImu/Pose field set required by the
localization module in 7.0, sign/frame conventions of the IMU axes, and gravity
handling in ``linear_acceleration``.
"""
from apollo_modules.modules.common.proto.header_pb2 import Header
from apollo_modules.modules.common.proto.geometry_pb2 import Point3D
from apollo_modules.modules.localization.proto.imu_pb2 import CorrectedImu
from apollo_modules.modules.localization.proto.pose_pb2 import Pose

from .base import Publisher
from ..registry import PUBLISHER_REGISTRY


@PUBLISHER_REGISTRY.register("publisher.imu")
class ImuPublisher(Publisher):
    channel = "/apollo/sensor/gnss/corrected_imu"
    msg_type = "apollo.localization.CorrectedImu"
    msg_cls = CorrectedImu
    proto_types = [CorrectedImu]   # not linked in cyber_bridge → must RegisterDesc
    frequency = 100.0

    def _process_data(self, message):
        av = message.angular_velocity
        la = message.linear_acceleration
        ea = message.euler_angles
        return CorrectedImu(
            header=Header(timestamp_sec=message.timestamp, module_name="drivora",
                          sequence_num=self.frame_count),
            imu=Pose(
                linear_acceleration=Point3D(x=la.x, y=la.y, z=la.z),
                angular_velocity=Point3D(x=av.x, y=av.y, z=av.z),
                euler_angles=Point3D(x=ea.x, y=ea.y, z=ea.z),
                heading=ea.z,
            ),
        )
