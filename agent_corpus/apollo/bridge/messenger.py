"""One-stop manager for an Apollo instance: container + bridge + pub/sub pool.

Self-contained manager. Unlike ground-truth setups (perfect perception), the
default publisher set here forwards raw sensors so Apollo runs its own
perception + localization.
"""
from typing import List, Optional
from loguru import logger

from .container import ApolloContainer
from .registry import PUBLISHER_REGISTRY, SUBSCRIBER_REGISTRY

# Importing this package triggers @PUBLISHER_REGISTRY.register / @SUBSCRIBER_REGISTRY.register
from . import publishers as _publishers  # noqa: F401
from . import subscribers as _subscribers  # noqa: F401


class ApolloMessenger:
    def __init__(
        self,
        idx,
        apollo_modules: List[str],
        publishers: List[str],
        subscribers: List[str],
        container_name: str,
        gpu: str = "0",
        cpu: float = 24.0,
        apollo_root: str = "/apollo",
        map_name: str = "san_mateo",
        dreamview_port: int = 8888,
        bridge_port: int = 9090,
        map_dreamview: bool = False,
        start_modules: bool = True,
        start_dreamview: bool = False,
    ):
        self.idx = idx
        self.publishers = publishers
        self.subscribers = subscribers
        self.container_name = container_name

        self.publisher_pool = {}
        self.subscriber_pool = {}

        self.container = ApolloContainer(
            container_name,
            modules=apollo_modules,
            gpu=gpu, cpu=str(cpu),
            apollo_root=apollo_root,
            map_name=map_name,
            dreamview_port=dreamview_port,
            bridge_port=bridge_port,
            map_dreamview=map_dreamview,
        )

        self.container.start_container()
        self.container.clean_cache()
        if start_dreamview:
            self.container.start_dreamview_server()
        if start_modules:
            self.container.start_modules_script()

        if not self.container.start_bridge():
            raise RuntimeError(f"Apollo cyber_bridge not available for {container_name}")

        self.register_publishers()
        self.register_subscribers()
        self.container.bridge.spin()

    def register_publishers(self):
        for name in self.publishers:
            if name in self.publisher_pool:
                raise RuntimeError(f"Publisher {name} already exists")
            cls = PUBLISHER_REGISTRY.get(name)
            self.publisher_pool[name] = cls(idx=name, bridge=self.container.bridge)
        logger.info(f"Registered publishers: {self.publishers}")

    def register_subscribers(self):
        for name in self.subscribers:
            if name in self.subscriber_pool:
                raise RuntimeError(f"Subscriber {name} already exists")
            cls = SUBSCRIBER_REGISTRY.get(name)
            self.subscriber_pool[name] = cls(idx=name, bridge=self.container.bridge)
        logger.info(f"Registered subscribers: {self.subscribers}")

    def publish_message(self, name: str, message):
        try:
            self.publisher_pool[name].publish(message)
        except Exception as e:
            logger.warning(f"Error publishing {name}: {e}")

    def shutdown(self):
        self.container.stop_bridge()
        self.publisher_pool.clear()
        self.subscriber_pool.clear()

    # ----- recorder passthrough -----
    def recorder_operator(self, operation: str, record_folder: Optional[str] = None,
                          scenario_id: Optional[str] = None):
        if operation == "start":
            self.container.start_recorder(record_folder, scenario_id)
        elif operation == "stop":
            self.container.stop_recorder()
        else:
            raise RuntimeError(f"Unsupported recorder operation: {operation}")

    def move_recording(self, record_folder: str, scenario_id: str, local_folder: str, delete: bool = True):
        self.container.copy_record(record_folder, scenario_id, local_folder, delete=delete)
