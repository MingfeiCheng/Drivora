"""Self-contained Apollo CyberBridge layer for Drivora.

Self-contained implementation; no cross-repo imports. Provides:
  - CyberBridge        : raw TCP pub/sub against Apollo's cyber_bridge
  - ApolloContainer    : Apollo Docker lifecycle (bind-mounts the apollo tree)
  - ApolloMessenger    : one-stop container + bridge + publisher/subscriber pool
  - PUBLISHER_REGISTRY / SUBSCRIBER_REGISTRY : local registries (do not touch
    Drivora's global registry/)
"""
from .registry import PUBLISHER_REGISTRY, SUBSCRIBER_REGISTRY  # noqa: F401
