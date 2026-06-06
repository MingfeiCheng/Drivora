"""Camera image publisher → Apollo Perception (camera detection).

Real-sensor perception: forwards CARLA RGB frames to Apollo. The channel name
encodes the camera (e.g. front_6mm) and MUST match both Apollo's perception
config and the camera intrinsics/extrinsics calibration in
``/apollo/modules/perception/data`` + ``/apollo/modules/calibration/data``.

VERIFY (needs a live Apollo): exact channel names, image ``encoding`` Apollo's
camera detector expects (rgb8 vs bgr8), and whether raw ``Image`` or
``CompressedImage`` is consumed for the chosen perception pipeline.
"""
from apollo_modules.modules.common.proto.header_pb2 import Header
from apollo_modules.modules.drivers.proto.sensor_image_pb2 import Image

from .base import Publisher
from ..registry import PUBLISHER_REGISTRY


@PUBLISHER_REGISTRY.register("publisher.camera")
class CameraPublisher(Publisher):
    # Channel is templated per camera_name at publish time; default registers
    # the common front camera so the writer exists up front.
    channel = "/apollo/sensor/camera/front_6mm/image"
    msg_type = "apollo.drivers.Image"
    msg_cls = Image
    proto_types = [Image]   # not linked in cyber_bridge → must RegisterDesc
    frequency = 30.0

    def __init__(self, idx, bridge):
        super().__init__(idx, bridge)
        self._registered_channels = {self.channel}

    def _channel_for(self, camera_name: str) -> str:
        return f"/apollo/sensor/camera/{camera_name}/image"

    def publish(self, message):
        # Ensure a writer exists for this camera's channel (multi-camera rigs).
        channel = self._channel_for(message.camera_name)
        if channel not in self._registered_channels:
            self.bridge.add_publisher(channel, self.msg_type)
            self._registered_channels.add(channel)

        img = message.image  # HxWx3 uint8 RGB
        proto = Image(
            header=Header(timestamp_sec=message.timestamp, module_name="drivora",
                          frame_id=message.frame_id, sequence_num=self.frame_count),
            frame_id=message.frame_id,
            measurement_time=message.timestamp,
            height=int(img.shape[0]),
            width=int(img.shape[1]),
            encoding="rgb8",  # VERIFY against Apollo camera detector expectation
            step=int(img.shape[1] * img.shape[2]),
            data=img.tobytes(),
        )
        self.frame_count += 1
        self.bridge.publish(channel, proto.SerializeToString())
