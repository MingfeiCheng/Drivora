"""In-container native cyber sensor injector (runs INSIDE the Apollo container).

cyber_bridge cannot carry bulk sensor data (point clouds / images) from an
external client — large frames break the connection and the type-descriptor
forwarding is unreliable. This node instead publishes the sensor channels with
NATIVE cyber Writers from inside the container, and receives the raw sensor
frames from the host (CARLA side) over a lightweight TCP socket. This is the
proper way to feed simulated sensors to Apollo perception.

Channels published:
  /apollo/sensor/lidar128/compensator/PointCloud2     (apollo.drivers.PointCloud)
  /apollo/sensor/camera/front_6mm/image               (apollo.drivers.Image)        — perception
  /apollo/sensor/camera/front_6mm/image/compressed    (apollo.drivers.CompressedImage) — Dreamview camera panel

Wire protocol (host -> this node), repeated:
  1 byte  type: b'L' lidar | b'C' camera
  8 bytes double little-endian: timestamp (s)
  type L: 4 bytes uint32 LE npoints, then npoints*4 float32 (x,y,z,intensity)
  type C: 4 bytes H, 4 bytes W (uint32 LE), then H*W*3 uint8 (RGB)

Run inside container:
  source /apollo/cyber/setup.bash
  python3 /apollo/drivora_injector.py [port]
"""
import sys, socket, struct, threading, time

sys.path.insert(0, "/apollo/bazel-bin")
from cyber.python.cyber_py3 import cyber
from modules.drivers.proto.pointcloud_pb2 import PointCloud
from modules.drivers.proto.sensor_image_pb2 import Image, CompressedImage
from modules.common.proto.header_pb2 import Header
import numpy as np
try:
    import cv2  # jpeg-encode the compressed channel (Dreamview camera view)
except Exception:
    cv2 = None

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9100
# Which sensor(s) this PROCESS handles. Run one sensor per process (own python
# interpreter -> own GIL) so heavy camera jpeg encoding never starves the lidar
# PointCloud build (shared-process GIL contention dropped lidar frames -> the
# perception detection flicker). Usage: drivora_injector.py <port> [lidar|camera|lidar,camera]
SENSORS = set((sys.argv[2] if len(sys.argv) > 2 else "lidar,camera").split(","))
# camera channel name (argv[3]) so one process per camera: front_6mm / front_12mm
CAM_NAME = sys.argv[3] if len(sys.argv) > 3 else "front_6mm"
LIDAR_CH = "/apollo/sensor/lidar128/compensator/PointCloud2"
CAMERA_CH = f"/apollo/sensor/camera/{CAM_NAME}/image"
# Dreamview's camera panel (perception_camera_updater) reads FLAGS_image_short_topic,
# a CompressedImage. Perception itself uses the raw CAMERA_CH above.
CAMERA_COMPRESSED_CH = f"/apollo/sensor/camera/{CAM_NAME}/image/compressed"


def recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def main():
    cyber.init()
    node = cyber.Node("drivora_sensor_injector_%d" % PORT)
    lidar_w = node.create_writer(LIDAR_CH, PointCloud) if "lidar" in SENSORS else None
    cam_w = node.create_writer(CAMERA_CH, Image) if "camera" in SENSORS else None
    comp_w = node.create_writer(CAMERA_COMPRESSED_CH, CompressedImage) if ("camera" in SENSORS and cv2 is not None) else None
    seq = {"l": 0, "c": 0}
    print(f"[injector] sensors={sorted(SENSORS)} writers up; listening on 0.0.0.0:{PORT}"
          f"{' (no cv2 -> no compressed)' if (cv2 is None and 'camera' in SENSORS) else ''}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(4)

    def handle(conn):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while cyber.ok():
            hdr = recvall(conn, 9)
            if not hdr:
                break
            mtype = hdr[0:1]
            ts = struct.unpack("<d", hdr[1:9])[0]
            if mtype == b"L":
                meta = recvall(conn, 4)
                npts = struct.unpack("<I", meta)[0]
                payload = recvall(conn, npts * 16)  # 4 float32 per point
                if payload is None:
                    break
                if lidar_w is None:
                    continue
                pts = np.frombuffer(payload, dtype="<f4").reshape(-1, 4)
                pc = PointCloud(
                    header=Header(timestamp_sec=ts, module_name="drivora",
                                  frame_id="velodyne128", sequence_num=seq["l"]),
                    frame_id="velodyne128", is_dense=False, measurement_time=ts,
                    height=1, width=int(pts.shape[0]))
                ns = int(ts * 1e9)
                for p in pts:
                    pc.point.add(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                                 intensity=int(p[3]) & 0xFF, timestamp=ns)
                lidar_w.write(pc)
                seq["l"] += 1
            elif mtype == b"C":
                meta = recvall(conn, 8)
                h, w = struct.unpack("<II", meta)
                payload = recvall(conn, h * w * 3)
                if payload is None:
                    break
                if cam_w is None:
                    continue
                # raw image -> perception
                img = Image(
                    header=Header(timestamp_sec=ts, module_name="drivora",
                                  frame_id=CAM_NAME, sequence_num=seq["c"]),
                    frame_id=CAM_NAME, measurement_time=ts,
                    height=h, width=w, encoding="rgb8", step=w * 3, data=payload)
                cam_w.write(img)
                # compressed jpeg -> Dreamview camera panel
                if comp_w is not None:
                    try:
                        rgb = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3)
                        bgr = rgb[:, :, ::-1]  # cv2.imencode expects BGR
                        ok, buf = cv2.imencode(".jpg", bgr)
                        if ok:
                            comp = CompressedImage(
                                header=Header(timestamp_sec=ts, module_name="drivora",
                                              frame_id=CAM_NAME, sequence_num=seq["c"]),
                                frame_id=CAM_NAME, format="jpeg",
                                measurement_time=ts, data=buf.tobytes())
                            comp_w.write(comp)
                    except Exception:
                        pass
                seq["c"] += 1
        conn.close()

    while cyber.ok():
        try:
            conn, _ = srv.accept()
        except Exception:
            break
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
