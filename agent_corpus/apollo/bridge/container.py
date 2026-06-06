"""Apollo Docker container lifecycle for Drivora.

Self-contained container helper. The Apollo source tree (``apollo_root``) is
bind-mounted by Apollo's own ``dev_start_ctn.sh``; point it at your built Apollo
source tree (the only shared asset).

Responsibilities:
  - create/start the container (via Apollo's dev_start script)
  - connect the cyber_bridge
  - start the required Apollo modules (Transform/Localization/Perception/
    Prediction/Planning/Control/Routing) via ``cyber_launch``
  - start/stop/copy cyber_recorder bags
"""
import os
import time
import subprocess

from typing import List, Optional
from loguru import logger

from .cyber_bridge import CyberBridge
from .dreamview import Dreamview

# launch files for the full real-sensor stack
MODULE_LAUNCHES = {
    "Transform": "modules/transform/launch/static_transform.launch",
    "Localization": "modules/localization/launch/localization.launch",
    "Perception": "modules/perception/production/launch/perception_all.launch",
    "Prediction": "modules/prediction/launch/prediction.launch",
    "Planning": "modules/planning/launch/planning.launch",
    "Control": "modules/control/launch/control.launch",
    "Routing": "modules/routing/launch/routing.launch",
}


class ApolloContainer:
    def __init__(
        self,
        name: str,
        modules: Optional[List[str]] = None,
        gpu: str = "0",
        cpu: str = "24.0",
        apollo_root: str = "/apollo",
        map_name: str = "san_mateo",
        dreamview_port: int = 8888,
        bridge_port: int = 9090,
        map_dreamview: bool = False,
    ) -> None:
        self.user = name
        self.name = name
        self.APOLLO_MODULES = modules or [
            "Transform", "Localization", "Perception",
            "Prediction", "Planning", "Control", "Routing",
        ]
        self.hd_map = map_name
        self.apollo_root = apollo_root
        self.map_dreamview = map_dreamview

        self.dreamview: Optional[Dreamview] = None
        self.dreamview_port = dreamview_port
        self.bridge: Optional[CyberBridge] = None
        self.bridge_port = bridge_port

        self.cpu_usage = cpu
        self.gpu_usage = gpu

        self.create_container()

    # ----- introspection -----
    @property
    def host(self) -> str:
        import docker
        ctn = docker.from_env().containers.get(self.name)
        ip = ctn.attrs["NetworkSettings"]["IPAddress"]
        return ip or "localhost"

    @property
    def is_container_running(self) -> bool:
        try:
            import docker
            return docker.from_env().containers.get(self.name).status == "running"
        except Exception:
            return False

    @property
    def is_bridge_running(self) -> bool:
        try:
            b = CyberBridge(self.host, self.bridge_port)
            b.conn.close()
            return True
        except Exception:
            return False

    @property
    def is_modules_running(self) -> bool:
        if self.dreamview is None:
            return False
        return all(self.dreamview.check_module_status(m) for m in self.APOLLO_MODULES)

    # ----- container -----
    def create_container(self):
        import docker
        client = docker.from_env()
        try:
            client.containers.get(self.name)
            return  # already exists
        except Exception:
            pass

        logger.info(f"Create Apollo container {self.name}")
        start_script = os.path.join(self.apollo_root, "docker", "scripts", "dev_start_ctn.sh")
        if not os.path.isfile(start_script):
            raise FileNotFoundError(
                f"Apollo dev start script not found: {start_script}. "
                f"Set apollo_root to a valid Apollo source tree "
                f"and apply apollo_bridge/patches/dev_start_ctn.sh."
            )
        options = "-y -l" + (" -md" if self.map_dreamview else "")
        subprocess.run(
            f"bash {start_script} {options}",
            env={
                "CURR_DIR": os.path.join(self.apollo_root, "docker", "scripts"),
                "APOLLO_ROOT_DIR": self.apollo_root,
                "USER": self.user,
            },
            shell=True,
        )

    def start_container(self):
        if self.is_container_running:
            return
        logger.info(f"Start Apollo container {self.name}")
        subprocess.run(f"docker start {self.name}", shell=True)
        time.sleep(0.5)

    def stop_container(self):
        if self.is_container_running:
            logger.info(f"Stop Apollo container {self.name}")
            subprocess.run(f"docker stop {self.name}", shell=True)

    # ----- dreamview -----
    def start_dreamview_server(self) -> bool:
        """Bring up the Dreamview web UI (HTTP on :8888 inside the container)
        via bootstrap.sh. Reachable from the host at ``<container_ip>:8888``.

        This is the piece that was missing: ``map_dreamview`` (-md) only mounts
        map volumes for dev_start; it does NOT launch Dreamview.
        """
        logger.info(f"Start Apollo Dreamview on {self.name} (:{self.dreamview_port})")
        cmd = (
            f"docker exec -d --user {self.user} {self.name} "
            f'bash -lc "source /apollo/cyber/setup.bash 2>/dev/null; '
            f'/apollo/scripts/bootstrap.sh start"'
        )
        subprocess.run(cmd, shell=True)
        # poll the HTTP port
        for _ in range(20):
            r = subprocess.run(
                f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 3 "
                f"http://{self.host}:{self.dreamview_port}",
                shell=True, capture_output=True, text=True)
            if r.stdout.strip() == "200":
                logger.info(f"Dreamview up: http://{self.host}:{self.dreamview_port}")
                return True
            time.sleep(1.0)
        logger.warning("Dreamview did not report HTTP 200 within timeout")
        return False

    # ----- bridge -----
    def start_bridge(self) -> bool:
        if not self.is_bridge_running:
            for _ in range(10):
                try:
                    subprocess.run(
                        f"docker exec --user {self.user} -d {self.name} ./scripts/bridge.sh",
                        shell=True,
                    )
                    self.bridge = CyberBridge(self.host, self.bridge_port)
                    break
                except (ConnectionRefusedError, AssertionError):
                    time.sleep(1.0)
        else:
            self.bridge = CyberBridge(self.host, self.bridge_port)
        if self.bridge is not None:
            logger.info(f"Connected Apollo cyber bridge: {self.host}:{self.bridge_port}")
        return self.is_bridge_running

    def stop_bridge(self):
        if self.bridge is None:
            return
        logger.info(f"Stop Apollo bridge: {self.host}:{self.bridge_port}")
        try:
            self.bridge.conn.close()
        except Exception:
            pass
        self.bridge.stop()

    # ----- modules (cyber_launch, no Dreamview required) -----
    def start_modules_script(self):
        logger.info(f"Start Apollo modules: {self.APOLLO_MODULES}")
        for m in self.APOLLO_MODULES:
            launch = MODULE_LAUNCHES.get(m)
            if launch is None:
                logger.warning(f"No launch file mapped for module '{m}', skipping.")
                continue
            cmd = (
                f'docker exec --user {self.user} -d {self.name} '
                f'bash -c "source /apollo/scripts/apollo_base.sh && '
                f'export CUDA_VISIBLE_DEVICES={self.gpu_usage} '
                f'NVIDIA_VISIBLE_DEVICES={self.gpu_usage} && '
                f'cyber_launch start {launch}"'
            )
            subprocess.run(cmd, shell=True)

    def stop_modules_script(self):
        for m in self.APOLLO_MODULES:
            launch = MODULE_LAUNCHES.get(m)
            if launch is None:
                continue
            cmd = (
                f'docker exec --user {self.user} {self.name} '
                f'bash -c "source /apollo/scripts/apollo_base.sh && '
                f'cyber_launch stop {launch}"'
            )
            subprocess.run(cmd, shell=True)

    # ----- recorder -----
    def start_recorder(self, record_folder: str, record_id: str):
        logger.info(f"Start Apollo recorder: {record_folder}/{record_id}")
        for c in (
            f"sh -c 'find /apollo -name \"cyber_recorder.log.INFO.*\" -delete'",
            f"rm -rf {record_folder}/{record_id}",
            f"mkdir -p {record_folder}/{record_id}",
        ):
            subprocess.run(f"docker exec --user {self.user} {self.name} {c}", shell=True)
        recorder = "/apollo/bazel-bin/cyber/tools/cyber_recorder/cyber_recorder"
        subprocess.run(
            f"docker exec -d --user {self.user} {self.name} "
            f"{recorder} record -o {record_folder}/{record_id}/recording -a &",
            shell=True,
        )
        time.sleep(1.0)

    def stop_recorder(self):
        logger.info("Stop Apollo recorder.")
        subprocess.run(
            f"docker exec --user {self.user} {self.name} "
            f"python3 /apollo/scripts/record_bag.py --stop --stop_signal SIGINT",
            shell=True,
        )
        time.sleep(1.0)

    def copy_record(self, record_folder: str, record_id: str, target_folder: str, delete: bool = False):
        subprocess.run(
            f"docker cp {self.name}:{record_folder}/{record_id} {target_folder}", shell=True
        )
        if delete:
            subprocess.run(
                f"docker exec --user {self.user} {self.name} rm -rf {record_folder}/{record_id}",
                shell=True,
            )

    # ----- housekeeping -----
    def clean_cache(self):
        """Remove runtime artefacts only; never touch /apollo/.cache (bazel)."""
        for c in (
            "rm -rf /apollo/data /apollo/records",
            "find /apollo -name '*.log.*' -delete 2>/dev/null || true",
            "mkdir -p /apollo/data/bag /apollo/data/log /apollo/data/core /apollo/records",
            f"chown -R {self.user} /apollo/data /apollo/records",
        ):
            subprocess.run(
                f"docker exec --user root {self.name} sh -c '{c}'",
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
