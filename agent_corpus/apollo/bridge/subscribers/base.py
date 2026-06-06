from ..cyber_bridge import CyberBridge


class Subscriber(object):
    channel: str = "/default"
    msg_type: str = "default"
    msg_cls = None

    def __init__(self, idx, bridge: CyberBridge):
        self.idx = idx
        self.bridge = bridge
        self.frame_count = 0
        self._register()

    def _register(self):
        self.bridge.add_subscriber(self.channel, self.msg_type, self.msg_cls, self._callback)

    def _callback(self, data):
        raise NotImplementedError

    def get_data(self):
        return None
