"""
Integration test: mock PylotServer + real PylotProxyAgent.
Verifies the full ZMQ communication chain without needing ERDOS or Docker.

Usage:
    .venvs/pylot/bin/python agent_corpus/pylot/test_integration.py
"""

import threading
import time
import json
import os
import sys
import numpy as np

import zmq
import msgpack
import msgpack_numpy as m
m.patch()


# ── Mock PylotServer (simulates the container) ──────────────────────────

MOCK_SENSORS = [
    {
        'type': 'sensor.camera.rgb',
        'x': 0.0, 'y': 0.0, 'z': 2.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'width': 800, 'height': 600, 'fov': 90,
        'id': 'center_camera',
    },
    {
        'type': 'sensor.lidar.ray_cast',
        'x': 0.0, 'y': 0.0, 'z': 2.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'id': 'LIDAR',
    },
    {'type': 'sensor.opendrive_map', 'reading_frequency': 20, 'id': 'opendrive'},
    {'type': 'sensor.other.gnss', 'x': 0, 'y': 0, 'z': 0, 'id': 'gnss'},
    {'type': 'sensor.other.imu', 'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0, 'id': 'imu'},
    {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'speed'},
]


def mock_server(port=12668):
    """Run a mock PylotServer that responds to all commands."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{port}")

    tick_count = 0
    print(f"[MockServer] Listening on port {port}")

    while True:
        raw = sock.recv()
        msg = msgpack.unpackb(raw, raw=False, object_hook=m.decode)
        cmd = msg.get('cmd')

        if cmd == 'ping':
            resp = {'status': 'ok'}
            print(f"[MockServer] ping -> ok")

        elif cmd == 'get_sensors':
            resp = {'status': 'ok', 'sensors': MOCK_SENSORS}
            print(f"[MockServer] get_sensors -> {len(MOCK_SENSORS)} sensors")

        elif cmd == 'init':
            has_opendrive = 'opendrive' in msg
            has_route = 'route' in msg
            route_len = len(msg.get('route', []))
            print(f"[MockServer] init -> opendrive={has_opendrive}, route_len={route_len}")
            resp = {'status': 'ok'}

        elif cmd == 'tick':
            tick_count += 1
            ts = msg.get('timestamp', 0)
            sensors = msg.get('sensors', {})
            cam_count = len(sensors.get('cameras', {}))
            has_lidar = 'lidar' in sensors
            has_imu = 'imu' in sensors
            has_gnss = 'gnss' in sensors
            has_speed = 'speed' in sensors

            # Check camera data size
            cam_sizes = {k: len(v) for k, v in sensors.get('cameras', {}).items()}

            print(f"[MockServer] tick #{tick_count} ts={ts} cameras={cam_sizes} "
                  f"lidar={has_lidar} imu={has_imu} gnss={has_gnss} speed={has_speed}")

            # Return mock control
            resp = {
                'status': 'ok',
                'control': {
                    'throttle': 0.5,
                    'steer': 0.1 * (tick_count % 5 - 2),
                    'brake': 0.0,
                    'hand_brake': False,
                    'reverse': False,
                }
            }

        elif cmd == 'destroy':
            print(f"[MockServer] destroy -> shutting down")
            resp = {'status': 'ok'}
            sock.send(msgpack.packb(resp, default=m.encode))
            break
        else:
            resp = {'status': 'error', 'message': f'Unknown: {cmd}'}

        sock.send(msgpack.packb(resp, default=m.encode))

    sock.close()
    ctx.term()
    print(f"[MockServer] Done. Processed {tick_count} ticks.")


# ── Test: PylotProxyAgent talks to MockServer ────────────────────────────

def test_proxy_agent():
    import carla
    import enum
    from agent_corpus.pylot.pylot_proxy_agent import PylotProxyAgent

    # carla 0.9.15 pip package may not expose RoadOption directly
    class RoadOption(enum.IntEnum):
        VOID = -1
        LEFT = 1
        RIGHT = 2
        STRAIGHT = 3
        LANEFOLLOW = 4
        CHANGELANELEFT = 5
        CHANGELANERIGHT = 6

    PORT = 12668

    # Start mock server in background thread
    server_thread = threading.Thread(target=mock_server, args=(PORT,), daemon=True)
    server_thread.start()
    time.sleep(0.5)  # Let server bind

    # Write temp config — set container_name to empty so it skips docker start
    config_path = '/tmp/pylot_test_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'container_host': 'localhost',
            'container_port': PORT,
            'jpeg_quality': 80,
            'container_name': '',
        }, f)

    # ── 1. Create and setup agent ──
    print("\n=== Test: setup ===")
    agent = PylotProxyAgent()
    agent.setup(config_path)
    print(f"  Sensors declared: {len(agent.sensors())}")
    for s in agent.sensors():
        print(f"    - {s['id']} ({s['type']})")

    # ── 2. Set global plan ──
    print("\n=== Test: set_global_plan ===")
    mock_plan_gps = [
        ({'lat': 0.0, 'lon': 0.0, 'z': 0.0}, 'LANEFOLLOW'),
        ({'lat': 0.001, 'lon': 0.001, 'z': 0.0}, 'LEFT'),
    ]
    mock_plan_world = [
        (carla.Transform(carla.Location(x=10, y=20, z=0), carla.Rotation()), RoadOption.LANEFOLLOW),
        (carla.Transform(carla.Location(x=30, y=40, z=0), carla.Rotation()), RoadOption.LEFT),
    ]
    agent.set_global_plan(mock_plan_gps, mock_plan_world)
    print(f"  Route buffered: {len(agent._buffered_route)} waypoints")

    # ── 3. Simulate ticks ──
    print("\n=== Test: run_step (5 ticks) ===")
    for i in range(5):
        # Mock sensor data
        input_data = {
            'center_camera': (i, np.random.randint(0, 255, (600, 800, 4), dtype=np.uint8)),
            'LIDAR': (i, np.random.randn(1000, 4).astype(np.float32)),
            'imu': (i, np.array([0.1, 0.0, 9.8, 0.0, 0.0, 0.0, 1.57], dtype=np.float64)),
            'gnss': (i, np.array([48.0, 11.0, 300.0], dtype=np.float64)),
            'speed': (i, {'speed': 5.0 + i}),
        }
        # First tick also sends opendrive
        if i == 0:
            input_data['opendrive'] = (i, {'opendrive': '<OpenDRIVE>mock</OpenDRIVE>'})

        timestamp = 0.05 * i  # 20 FPS
        control, log_data = agent.run_step(input_data, timestamp)
        print(f"  tick {i}: throttle={control.throttle:.2f} steer={control.steer:.2f} brake={control.brake:.2f}")

    # ── 4. Destroy ──
    print("\n=== Test: destroy ===")
    agent.destroy()

    server_thread.join(timeout=2)
    print("\n=== ALL TESTS PASSED ===")


if __name__ == '__main__':
    test_proxy_agent()
