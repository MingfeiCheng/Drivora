"""Minimal client for Apollo's ``cyber_bridge`` TCP protocol.

Rewritten for Drivora. The wire format (length-prefixed frames, op codes) is
defined by Apollo's ``modules/contrib/cyber_bridge/client.cc``.

Frame layout for every payload chunk: 4-byte little-endian length, then bytes.
"""
import socket

from collections import defaultdict
from threading import Thread
from typing import DefaultDict, List, Set


def _to_bytes(s: str) -> bytes:
    return bytes(s, "ascii")


class BridgeOp:
    """cyber_bridge operation codes (single byte, big-endian)."""
    RegisterDesc = (1).to_bytes(1, byteorder="big")
    AddReader = (2).to_bytes(1, byteorder="big")
    AddWriter = (3).to_bytes(1, byteorder="big")
    Publish = (4).to_bytes(1, byteorder="big")


class CyberBridge:
    conn: socket.socket
    subscribers: DefaultDict[str, List]
    publishable_channel: Set[str]
    spinning: bool
    t: Thread

    def __init__(self, host: str, port: int = 9090) -> None:
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((host, port))
        self.conn.setblocking(False)
        self.subscribers = defaultdict(list)
        self.publishable_channel = set()
        self.spinning = False

    @staticmethod
    def _prepare_bytes(data: bytes) -> bytes:
        """Prefix ``data`` with its 4-byte little-endian length."""
        result = bytes()
        for s in (0, 8, 16, 24):
            result += ((len(data) >> s).to_bytes(4, byteorder="big")[-1]).to_bytes(1, byteorder="big")
        result += data
        return result

    def add_subscriber(self, channel: str, message_type: str, message_cls, callback):
        data = BridgeOp.AddReader
        data += self._prepare_bytes(_to_bytes(channel))
        data += self._prepare_bytes(_to_bytes(message_type))
        self.conn.send(data)

        def cb_wrapper(raw: bytes):
            parsed = message_cls()
            parsed.ParseFromString(raw)
            callback(parsed)

        self.subscribers[channel].append(cb_wrapper)

    def register_message_descriptors(self, message_cls):
        """Dynamically register a protobuf type (+ its dependencies) with the
        cyber_bridge via OP_REGISTER_DESC.

        Apollo's cyber_bridge only natively handles the message types linked
        into it (localization/perception/canbus/control/routing). Sensor types
        such as ``apollo.localization.Gps`` / ``apollo.drivers.gnss.InsStat`` /
        ``apollo.localization.CorrectedImu`` are NOT linked, so AddWriter for
        them silently drops messages. Sending their FileDescriptorProtos (in
        dependency order) makes the bridge able to handle them.

        Wire format (see contrib/cyber_bridge/client.cc handle_register_desc):
            [op=1][count:u32 LE]( [size:u32 LE][serialized FileDescriptorProto] )*
        """
        # collect file descriptors in dependency-first order
        ordered = []
        seen = set()

        def visit(file_desc):
            if file_desc.name in seen:
                return
            seen.add(file_desc.name)
            for dep in file_desc.dependencies:
                visit(dep)
            ordered.append(file_desc)

        visit(message_cls.DESCRIPTOR.file)
        blobs = [fd.serialized_pb for fd in ordered]

        data = BridgeOp.RegisterDesc
        data += len(blobs).to_bytes(4, byteorder="little")
        for b in blobs:
            data += len(b).to_bytes(4, byteorder="little")
            data += b
        self.conn.send(data)

    def add_publisher(self, channel: str, message_type: str):
        if channel in self.publishable_channel:
            return
        data = BridgeOp.AddWriter
        data += self._prepare_bytes(_to_bytes(channel))
        data += self._prepare_bytes(_to_bytes(message_type))
        self.conn.send(data)
        self.publishable_channel.add(channel)

    @staticmethod
    def _get_32_le(b: bytes) -> int:
        assert len(b) == 4, f"Expecting 4 bytes, got {len(b)}"
        return b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24

    def on_read(self, data: bytes):
        if not data:
            return
        if data[0] == int.from_bytes(BridgeOp.Publish, "big"):
            self.receive_publish(data)

    def receive_publish(self, data: bytes):
        if not self.spinning:
            return
        offset = 1
        topic_length = self._get_32_le(data[offset:offset + 4])
        offset += 4
        topic = data[offset:offset + topic_length].decode("ascii")
        offset += topic_length
        message_size = self._get_32_le(data[offset:offset + 4])
        offset += 4
        msg = data[offset:offset + message_size]
        for subscriber in self.subscribers[topic]:
            subscriber(msg)

    def publish(self, channel: str, data: bytes):
        assert isinstance(data, bytes)
        msg = BridgeOp.Publish
        msg += self._prepare_bytes(_to_bytes(channel))
        msg += self._prepare_bytes(data)
        self.conn.send(msg)

    def _spin(self):
        while self.spinning:
            try:
                data = self.conn.recv(65527)
                self.on_read(data)
            except Exception:
                # Non-blocking socket raises when no data is available; ignore.
                pass

    def spin(self):
        if self.spinning:
            return
        self.spinning = True
        self.t = Thread(target=self._spin, daemon=True)
        self.t.start()

    def stop(self):
        self.spinning = False
        try:
            self.t.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass
