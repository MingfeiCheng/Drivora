"""ApolloBackendManager — Python lifecycle/health manager for Apollo ADS backends.

Replaces the brittle bash multi-container orchestration in scripts/random/apollo.sh
with a managed pool of Apollo containers, one per CARLA worker. It owns:

  * lifecycle      — idempotent bring-up (reuse healthy, recreate broken) + shutdown
  * health checks  — cyber_bridge port, sensor-injector ports, critical modules up
  * fault recovery — restart(idx) a dead backend so the fuzzer can requeue its job
  * per-worker IO  — writes real_config_<i>.json (apollo_host = that container's IP)
                     and exposes the list for the fuzzer's worker_agent_configs

It still calls scripts/apollo/start_apollo.sh for the heavy container creation
(that script is idempotent), but adds the management layer bash lacked: cleanup,
verified-up health checks with retry, per-worker port/config, and restart.

Generic contract (so the framework stays ADS-agnostic): the agent config names
this class via `agent.backend.entry_point`; the runner instantiates it, calls
`bring_up() -> List[str]` (per-worker config paths) and keeps it for health/
restart/shutdown during the campaign.
"""
import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

# Substrings of the mainboard DAG cmdlines that MUST be running for a usable ADS.
CRITICAL_DAGS = [
    "perception",      # dag_streaming_perception(.dag)
    "planning",
    "control",
    "prediction",
    "routing",
    "transform",       # static_transform
    "localization",    # dag_streaming_rtk_localization
]


@dataclass
class ApolloBackend:
    idx: int
    container_name: str
    gpu: str
    config_path: str
    dreamview_publish: bool = False
    host: Optional[str] = None
    bridge_port: int = 9090
    injector_port: int = 9100


class ApolloBackendManager:
    """Manages N Apollo containers (one per CARLA worker, paired by index)."""

    def __init__(
        self,
        num: int,
        run_tag: str,
        drivora_root: str,
        apollo_gpus: str = "1",
        dreamview_worker0: bool = False,
        template_config: Optional[str] = None,
        start_timeout: float = 360.0,
        health_retries: int = 1,
        topology: str = "multi_seed",
        reuse_existing: bool = True,
        log_dir: Optional[str] = None,
        dreamview_host_port: Optional[int] = None,
        **_ignored,
    ):
        # optional: expose EACH worker's Dreamview on a distinct host port via an
        # in-process TCP relay (worker i -> localhost:<base+i>). Workers don't
        # publish to host (avoids the :8888 singleton conflict), so this gives
        # per-container browser access without -md / without editing Apollo.
        self.dreamview_host_port = int(dreamview_host_port) if dreamview_host_port else None
        self._proxy_socks = []
        self.num = int(num)
        self.topology = topology
        # where to persist each backend's start_apollo bring-up log (per-run dir,
        # passed by the framework). Falls back to /tmp for standalone use.
        self.log_dir = log_dir or "/tmp"
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception:
            self.log_dir = "/tmp"
        # reuse_existing=True (resume): adopt this run's own healthy containers.
        # False (fresh run): recreate them clean so we never inherit a stale or
        # another run's container. Containers are run-scoped via run_tag, so a
        # different run_tag is always a different set — we only ever touch ours.
        self.reuse_existing = bool(reuse_existing)
        self.run_tag = run_tag
        self.drivora_root = drivora_root
        if isinstance(apollo_gpus, (list, tuple)):
            self.gpus = [str(g).strip() for g in apollo_gpus if str(g).strip()]
        else:
            self.gpus = [g.strip() for g in str(apollo_gpus).split(",") if g.strip()]
        self.dreamview_worker0 = bool(dreamview_worker0)
        self.template_config = template_config or os.path.join(
            drivora_root, "agent_corpus/apollo/config/real_config.json")
        self.start_apollo = os.path.join(drivora_root, "scripts/apollo/start_apollo.sh")
        self.start_timeout = start_timeout
        self.health_retries = health_retries
        # dev_start_ctn.sh names the container AND the in-container user after this
        # name, and useradd caps usernames at 32 chars. apollo_<run_tag>_<i> easily
        # exceeds that (-> useradd fails -> every `docker exec --user` fails ->
        # nothing starts). Use a SHORT but run-unique name: apollo_<hash(run_tag)>_<i>.
        import hashlib
        run_hash = hashlib.md5(run_tag.encode()).hexdigest()[:8]
        self.run_hash = run_hash
        self.backends: List[ApolloBackend] = []
        for i in range(self.num):
            self.backends.append(ApolloBackend(
                idx=i,
                container_name=f"apollo_{run_hash}_{i}",
                gpu=self.gpus[i % len(self.gpus)] if self.gpus else "0",
                config_path=os.path.join(
                    drivora_root, f"agent_corpus/apollo/config/real_config_{i}.json"),
                dreamview_publish=(i == 0 and self.dreamview_worker0),
            ))

    # ---- low-level helpers ------------------------------------------------
    @staticmethod
    def _sh(cmd: List[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def _rm_container(self, name: str):
        self._sh(["docker", "rm", "-f", name], timeout=60.0)

    def _container_state(self, name: str) -> str:
        """'running' | 'exists' (stopped/created) | 'absent'."""
        r = self._sh(["docker", "inspect", "-f", "{{.State.Running}}", name], timeout=20.0)
        if r.returncode != 0:
            return "absent"
        return "running" if r.stdout.strip() == "true" else "exists"

    @staticmethod
    def _port_open(host: str, port: int, timeout: float = 3.0) -> bool:
        if not host:
            return False
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _running_dags(self, container: str) -> str:
        r = self._sh(["docker", "exec", container, "bash", "-lc",
                      "ps aux 2>/dev/null | grep mainboard | grep -v grep"], timeout=20.0)
        return r.stdout or ""

    # ---- health -----------------------------------------------------------
    def health_check(self, b: ApolloBackend) -> bool:
        """True iff bridge+injector ports open AND all critical modules running."""
        if not b.host:
            b.host = self._read_host(b)
        if not self._port_open(b.host, b.bridge_port):
            logger.warning(f"[apollo {b.idx}] bridge {b.host}:{b.bridge_port} not open")
            return False
        if not self._port_open(b.host, b.injector_port):
            logger.warning(f"[apollo {b.idx}] injector {b.host}:{b.injector_port} not open")
            return False
        dags = self._running_dags(b.container_name)
        missing = [d for d in CRITICAL_DAGS if d not in dags]
        if missing:
            logger.warning(f"[apollo {b.idx}] modules missing: {missing}")
            return False
        return True

    def wait_healthy(self, b: ApolloBackend, timeout: float = 90.0, interval: float = 5.0) -> bool:
        """Poll health_check until healthy or timeout (modules can lag after start)."""
        end = time.time() + timeout
        while time.time() < end:
            if self.health_check(b):
                return True
            time.sleep(interval)
        return self.health_check(b)

    def _read_host(self, b: ApolloBackend) -> Optional[str]:
        try:
            with open(b.config_path) as f:
                cfg = json.load(f)
            b.bridge_port = int(cfg.get("bridge_port", b.bridge_port))
            b.injector_port = int(cfg.get("injector_port", b.injector_port))
            return cfg.get("apollo_host")
        except Exception:
            return None

    # ---- lifecycle --------------------------------------------------------
    def _start_one(self, b: ApolloBackend, fresh: bool = False):
        if fresh:
            self._rm_container(b.container_name)
        # seed this worker's config from the template, then start_apollo patches it
        if not os.path.isfile(b.config_path):
            self._sh(["cp", "-f", self.template_config, b.config_path])
        env = os.environ.copy()
        env.update({
            "APOLLO_CTN": b.container_name,   # also becomes the in-container user
            "USE_DREAMVIEW": "true" if b.dreamview_publish else "false",
            "APOLLO_GPU": str(b.gpu),
            "REAL_CONFIG": b.config_path,
        })
        logger.info(f"[apollo {b.idx}] start_apollo.sh ctn={b.container_name} "
                    f"gpu={b.gpu} dreamview_publish={b.dreamview_publish}")
        r = subprocess.run(["bash", self.start_apollo], env=env,
                           capture_output=True, text=True, timeout=self.start_timeout)
        # always persist the FULL bring-up log for debugging (per-run dir)
        log_path = os.path.join(self.log_dir, f"start_apollo_{b.container_name}.log")
        try:
            with open(log_path, "w") as f:
                f.write((r.stdout or "") + "\n--- STDERR ---\n" + (r.stderr or ""))
        except Exception:
            pass
        if r.returncode != 0:
            logger.error(f"[apollo {b.idx}] start_apollo.sh rc={r.returncode} "
                         f"(full log: {log_path})\n{r.stdout[-1200:]}")
        b.host = self._read_host(b)

    def _ensure_backend(self, b: ApolloBackend):
        """Auto-discover: reuse if running+healthy, repair if running+unhealthy,
        recreate if stopped, create if absent. Raises if it can't be made healthy."""
        state = self._container_state(b.container_name)
        if state == "running" and self.reuse_existing:
            b.host = self._read_host(b)
            if self.health_check(b):
                logger.info(f"[apollo {b.idx}] discovered healthy {b.container_name} -> reuse")
                return
            logger.info(f"[apollo {b.idx}] {b.container_name} running but unhealthy -> repair")
        elif state != "absent":
            # stopped, OR running on a fresh (non-resume) run -> recreate clean so
            # this run owns a fresh container and never inherits/steals another's.
            reason = "fresh run (no resume)" if state == "running" else "stopped"
            logger.info(f"[apollo {b.idx}] {b.container_name} {reason} -> recreate")
            self._rm_container(b.container_name)
        else:
            logger.info(f"[apollo {b.idx}] {b.container_name} absent -> create")
        for attempt in range(self.health_retries + 1):
            self._start_one(b, fresh=(attempt > 0))   # attempt 0 reuses/creates, then recreate
            if self.wait_healthy(b):
                return
            logger.warning(f"[apollo {b.idx}] unhealthy after attempt {attempt}; "
                           f"{'recreating' if attempt < self.health_retries else 'giving up'}")
        raise RuntimeError(f"Apollo backend {b.idx} ({b.container_name}) failed health check")

    def bring_up(self) -> List[str]:
        """Auto-discover-or-create all backends, verify health, return config paths."""
        for b in self.backends:
            self._ensure_backend(b)
            logger.info(f"[apollo {b.idx}] UP host={b.host} bridge={b.bridge_port} "
                        f"injector={b.injector_port} cfg={b.config_path}")
        if self.dreamview_host_port:
            for b in self.backends:
                self._start_dreamview_proxy(self.dreamview_host_port + b.idx, b)
        self._log_summary()
        return [b.config_path for b in self.backends]

    def dreamview_url(self, idx: int):
        """Browser URL for worker idx's Dreamview, or None if not exposed."""
        if self.dreamview_host_port is None or idx >= len(self.backends):
            return None
        return f"http://localhost:{self.dreamview_host_port + idx}"

    def _log_summary(self):
        bar = "=" * 64
        logger.info(bar)
        logger.info(f"Apollo backends UP ({len(self.backends)}, topology={self.topology}):")
        for b in self.backends:
            url = self.dreamview_url(b.idx)
            dv = f"dreamview {url}" if url else f"dreamview (container only) http://{b.host}:8888"
            logger.info(f"  worker {b.idx}: {b.container_name} @ {b.host}  |  {dv}")
        logger.info(bar)

    def _start_dreamview_proxy(self, host_port: int, b: ApolloBackend):
        """In-process TCP relay: host:<host_port> -> <b.host>:8888 so this worker's
        Dreamview is browsable on localhost without publishing docker host ports."""
        import threading

        def _relay(src, dst):
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except OSError:
                pass
            finally:
                for s in (src, dst):
                    try:
                        s.close()
                    except OSError:
                        pass

        def _serve(host_port, target_host):
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                srv.bind(("0.0.0.0", host_port))
                srv.listen(32)
            except OSError as e:
                logger.warning(f"[apollo {b.idx}] dreamview proxy bind :{host_port} failed: {e}")
                return
            self._proxy_socks.append(srv)
            logger.info(f"[apollo {b.idx}] Dreamview: http://localhost:{host_port} "
                        f"-> {target_host}:8888")
            while True:
                try:
                    cli, _ = srv.accept()
                    up = socket.create_connection((target_host, 8888), timeout=10)
                except OSError:
                    break
                threading.Thread(target=_relay, args=(cli, up), daemon=True).start()
                threading.Thread(target=_relay, args=(up, cli), daemon=True).start()

        threading.Thread(target=_serve, args=(host_port, b.host), daemon=True).start()

    def restart(self, idx: int) -> bool:
        """Recreate one backend (for fault recovery). Returns health."""
        b = self.backends[idx]
        logger.warning(f"[apollo {b.idx}] RESTART requested")
        self._start_one(b, fresh=True)
        ok = self.wait_healthy(b)
        logger.info(f"[apollo {b.idx}] restart -> {'healthy' if ok else 'STILL UNHEALTHY'}")
        return ok

    def ensure_healthy(self, worker_idx: int) -> bool:
        """Ensure the ADS backend(s) used by this CARLA worker are alive (restart
        if down). multi_seed: just backend[worker_idx]; multi_ads: ALL backends
        (the single sim drives all N ADS)."""
        targets = self.backends if self.topology == "multi_ads" else [self.backends[worker_idx]]
        ok = True
        for b in targets:
            if not self.health_check(b):
                ok = self.restart(b.idx) and ok
        return ok

    def worker_config_paths(self) -> List[str]:
        return [b.config_path for b in self.backends]

    def shutdown(self, remove: bool = False):
        for s in self._proxy_socks:
            try:
                s.close()
            except OSError:
                pass
        self._proxy_socks = []
        if not remove:
            return  # leave containers running for reuse across runs
        for b in self.backends:
            self._rm_container(b.container_name)
            logger.info(f"[apollo {b.idx}] removed {b.container_name}")
