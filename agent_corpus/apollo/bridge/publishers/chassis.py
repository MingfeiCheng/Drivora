from apollo_modules.modules.common.proto.header_pb2 import Header
from apollo_modules.modules.canbus.proto.chassis_pb2 import Chassis

from .base import Publisher
from ..registry import PUBLISHER_REGISTRY


@PUBLISHER_REGISTRY.register("publisher.chassis")
class ChassisPublisher(Publisher):
    channel = "/apollo/canbus/chassis"
    msg_type = "apollo.canbus.Chassis"
    msg_cls = Chassis
    frequency = 100.0

    def _process_data(self, message):
        speed_mps = message.speed_mps
        if message.reverse:
            gear = Chassis.GearPosition.GEAR_REVERSE
            speed_mps = -speed_mps
        else:
            gear = Chassis.GearPosition.GEAR_DRIVE
        return Chassis(
            header=Header(timestamp_sec=message.timestamp, module_name="drivora",
                          sequence_num=self.frame_count),
            engine_started=True,
            driving_mode=Chassis.DrivingMode.COMPLETE_AUTO_DRIVE,
            gear_location=gear,
            speed_mps=speed_mps,
            throttle_percentage=message.throttle_percentage,
            brake_percentage=message.brake_percentage,
            steering_percentage=message.steering_percentage,
        )
