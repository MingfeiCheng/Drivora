"""Publisher base classes.

A Publisher converts a bridge-internal message (see ``format.py``) into an
Apollo protobuf and writes it to a CyberBridge channel, with simple
timestamp-based frequency limiting.
"""
import traceback

from loguru import logger

from ..cyber_bridge import CyberBridge


class Publisher(object):
    channel: str = "/default"
    msg_type: str = "default"
    msg_cls = None
    frequency: float = 100.0
    # proto classes to register with the bridge via OP_REGISTER_DESC because
    # they are NOT linked into Apollo's cyber_bridge (sensor/gnss types). Leave
    # empty for the natively-supported types.
    proto_types = []

    def __init__(self, idx, bridge: CyberBridge):
        self.idx = idx
        self.bridge = bridge
        self.frame_count = 0
        self.last_publish_time = None
        for cls in self.proto_types:
            self.bridge.register_message_descriptors(cls)
        self._register()

    def _register(self):
        self.bridge.add_publisher(self.channel, self.msg_type)

    def _process_data(self, message):
        raise NotImplementedError

    def publish(self, message):
        period = 1.0 / self.frequency
        try:
            proto = self._process_data(message)
        except Exception as e:
            logger.warning(f"Publisher {self.channel} build failed: {e}")
            traceback.print_exc()
            return

        ts = message.timestamp  # every message MUST have a timestamp
        if self.last_publish_time is not None and ts - self.last_publish_time < period:
            return
        self.last_publish_time = ts

        if proto is not None:
            try:
                self.bridge.publish(self.channel, proto.SerializeToString())
                self.frame_count += 1
            except Exception as e:
                logger.warning(f"Publisher {self.channel} send failed: {e}")
                traceback.print_exc()


class MultiChannelPublisher(object):
    """Publisher that writes to several Apollo channels from one message.

    Used for GNSS → Apollo Localization, which needs odometry + ins_stat (and
    optionally corrected_imu) emitted together.

    Subclasses define ``channels`` as a list of (channel, msg_type) and
    implement ``_process_data`` returning a dict {channel: proto_or_None}.
    """
    channels = []           # list[tuple[str, str]]
    frequency: float = 100.0
    proto_types = []        # see Publisher.proto_types

    def __init__(self, idx, bridge: CyberBridge):
        self.idx = idx
        self.bridge = bridge
        self.frame_count = 0
        self.last_publish_time = None
        for cls in self.proto_types:
            self.bridge.register_message_descriptors(cls)
        for channel, msg_type in self.channels:
            self.bridge.add_publisher(channel, msg_type)

    def _process_data(self, message) -> dict:
        raise NotImplementedError

    def publish(self, message):
        period = 1.0 / self.frequency
        ts = message.timestamp
        if self.last_publish_time is not None and ts - self.last_publish_time < period:
            return
        try:
            protos = self._process_data(message)
        except Exception as e:
            logger.warning(f"MultiChannelPublisher {self.idx} build failed: {e}")
            traceback.print_exc()
            return
        self.last_publish_time = ts
        for channel, proto in protos.items():
            if proto is None:
                continue
            try:
                self.bridge.publish(channel, proto.SerializeToString())
            except Exception as e:
                logger.warning(f"Publisher {channel} send failed: {e}")
        self.frame_count += 1
