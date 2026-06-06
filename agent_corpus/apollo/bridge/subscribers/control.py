from apollo_modules.modules.control.proto.control_cmd_pb2 import ControlCommand
from apollo_modules.modules.canbus.proto.chassis_pb2 import Chassis

from .base import Subscriber
from ..registry import SUBSCRIBER_REGISTRY


@SUBSCRIBER_REGISTRY.register("subscriber.control")
class ControlSubscriber(Subscriber):
    channel = "/apollo/control"
    msg_type = "apollo.control.ControlCommand"
    msg_cls = ControlCommand

    def __init__(self, idx, bridge):
        self.control_data = None
        self.throttle_percentage = 0.0   # [0, 1]
        self.brake_percentage = 0.0
        self.steering_percentage = 0.0   # [-1, 1]
        self.reverse = False
        super().__init__(idx, bridge)

    def _callback(self, data):
        self.control_data = data
        self.throttle_percentage = data.throttle / 100.0
        self.brake_percentage = data.brake / 100.0
        self.steering_percentage = data.steering_target / 100.0
        self.reverse = data.gear_location == Chassis.GearPosition.GEAR_REVERSE

    def get_data(self):
        return (self.throttle_percentage, self.brake_percentage,
                self.steering_percentage, self.reverse)
