"""Thin Dreamview websocket client (HD map / vehicle / module control).

Rewritten for Drivora. Only used to configure the HD map + vehicle
calibration and to query module status; the actual sensor/control traffic goes
through CyberBridge, not Dreamview.
"""
import json
import time

from websocket import create_connection
from loguru import logger


def _titleize(name: str) -> str:
    """``san_mateo`` -> ``San Mateo`` (Dreamview display convention)."""
    return " ".join(s[0].upper() + s[1:] for s in name.split("_") if s)


class Dreamview:
    def __init__(self, ip: str, port: int) -> None:
        self.url = f"ws://{ip}:{port}/websocket"
        self.ws = create_connection(self.url)

    def reconnect(self):
        self.ws.close()
        self.ws = create_connection(self.url)

    def send_data(self, data: dict):
        self.ws.send(json.dumps(data))

    # ----- HD map -----
    def set_hd_map(self, hd_map: str):
        mapped = _titleize(hd_map)
        self.ws.send(json.dumps({"type": "HMIAction", "action": "CHANGE_MAP", "value": mapped}))
        if self.get_current_map() != mapped:
            raise RuntimeError(
                f"HD Map '{mapped}' not set. Verify /apollo/modules/map/data/{hd_map} "
                f"exists and restart Dreamview."
            )

    def get_current_map(self):
        try:
            self.reconnect()
        except ConnectionRefusedError as e:
            logger.error(f"Cannot query current HD map: {e}")
            return None
        data = json.loads(self.ws.recv())
        while data["type"] != "HMIStatus":
            data = json.loads(self.ws.recv())
        return data["data"]["currentMap"]

    # ----- vehicle calibration -----
    def set_vehicle(self, vehicle: str):
        mapped = _titleize(vehicle)
        self.ws.send(json.dumps({"type": "HMIAction", "action": "CHANGE_VEHICLE", "value": mapped}))
        if self.get_current_vehicle() != mapped:
            raise RuntimeError(
                f"Vehicle calibration '{mapped}' not set. Verify "
                f"/apollo/modules/calibration/data/{vehicle} exists."
            )

    def get_current_vehicle(self):
        try:
            self.reconnect()
        except ConnectionRefusedError as e:
            logger.error(f"Cannot query current vehicle: {e}")
            return None
        data = json.loads(self.ws.recv())
        while data["type"] != "HMIStatus":
            data = json.loads(self.ws.recv())
        return data["data"]["currentVehicle"]

    # ----- mode / modules -----
    def set_setup_mode(self, mode: str):
        self.ws.send(json.dumps({"type": "HMIAction", "action": "CHANGE_MODE", "value": mode}))

    def enable_module(self, module: str, wait_time: float = 3.0):
        tries = 0
        while not self.check_module_status(module):
            tries += 1
            if tries > 60:
                raise RuntimeError(f"Apollo module {module} cannot be started.")
            self.ws.send(json.dumps({"type": "HMIAction", "action": "START_MODULE", "value": module}))
            time.sleep(wait_time)
            wait_time += 0.5

    def disable_module(self, module: str, wait_time: float = 3.0):
        tries = 0
        while self.check_module_status(module):
            tries += 1
            if tries > 60:
                raise RuntimeError(f"Apollo module {module} cannot be stopped.")
            self.ws.send(json.dumps({"type": "HMIAction", "action": "STOP_MODULE", "value": module}))
            time.sleep(wait_time)
            wait_time += 1

    def _get_module_status(self):
        self.reconnect()
        data = json.loads(self.ws.recv())
        while data["type"] != "HMIStatus":
            data = json.loads(self.ws.recv())
        return data["data"]["modules"]

    def check_module_status(self, module: str) -> bool:
        for name, status in self._get_module_status().items():
            if name == module and not status:
                return False
        return True
