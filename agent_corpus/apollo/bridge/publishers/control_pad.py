from apollo_modules.modules.common.proto.header_pb2 import Header
from apollo_modules.modules.canbus.proto.chassis_pb2 import Chassis
from apollo_modules.modules.control.proto.pad_msg_pb2 import PadMessage, DrivingAction

from .base import Publisher
from ..registry import PUBLISHER_REGISTRY


@PUBLISHER_REGISTRY.register("publisher.control_pad")
class ControlPadPublisher(Publisher):
    channel = "/apollo/control/pad"
    msg_type = "apollo.control.PadMessage"
    msg_cls = PadMessage
    frequency = 1000.0  # do not rate-limit

    def _process_data(self, message):
        if message.action == 0:
            action = DrivingAction.STOP
        elif message.action == 1:
            action = DrivingAction.START
        else:
            action = DrivingAction.RESET
        return PadMessage(
            header=Header(timestamp_sec=message.timestamp, module_name="drivora",
                          sequence_num=self.frame_count),
            driving_mode=Chassis.DrivingMode.COMPLETE_AUTO_DRIVE,
            action=action,
        )
