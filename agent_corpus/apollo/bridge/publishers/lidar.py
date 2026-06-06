"""Lidar point-cloud publisher → Apollo Perception (lidar detection).

Forwards a CARLA lidar sweep (already transformed into the Apollo sensor frame
by the transform layer) as ``apollo.drivers.PointCloud``.

VERIFY (needs a live Apollo): channel name for the chosen lidar model
(``/apollo/sensor/lidar128/compensator/PointCloud2`` for the 128-beam config),
the frame the lidar detector expects, and whether per-point ``timestamp`` is
required.
"""
from apollo_modules.modules.common.proto.header_pb2 import Header
from apollo_modules.modules.drivers.proto.pointcloud_pb2 import PointCloud, PointXYZIT

from .base import Publisher
from ..registry import PUBLISHER_REGISTRY


@PUBLISHER_REGISTRY.register("publisher.lidar")
class LidarPublisher(Publisher):
    channel = "/apollo/sensor/lidar128/compensator/PointCloud2"
    msg_type = "apollo.drivers.PointCloud"
    msg_cls = PointCloud
    proto_types = [PointCloud]   # not linked in cyber_bridge → must RegisterDesc
    frequency = 10.0

    def _process_data(self, message):
        pts = message.points  # Nx4 float32: x, y, z, intensity
        cloud = PointCloud(
            header=Header(timestamp_sec=message.timestamp, module_name="drivora",
                          frame_id=message.frame_id, sequence_num=self.frame_count),
            frame_id=message.frame_id,
            is_dense=False,
            measurement_time=message.timestamp,
            height=1,
            width=int(pts.shape[0]),
        )
        for p in pts:
            cloud.point.add(
                x=float(p[0]), y=float(p[1]), z=float(p[2]),
                intensity=int(p[3]) if len(p) > 3 else 0,
                timestamp=int(message.timestamp * 1e9),
            )
        return cloud
