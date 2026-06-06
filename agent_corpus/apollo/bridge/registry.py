"""Local registries for Apollo publishers/subscribers.

Deliberately independent from Drivora's top-level ``registry/`` package so that
adding the Apollo agent cannot collide with or perturb the fuzzer/runner
registries used by the rest of Drivora.
"""


class _Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry = {}

    def register(self, name=None):
        def decorator(cls):
            key = name or cls.__name__
            if key in self._registry:
                raise KeyError(f"'{key}' already registered in '{self.name}'")
            self._registry[key] = cls
            return cls
        return decorator

    def get(self, key):
        if key in self._registry:
            return self._registry[key]
        raise KeyError(f"'{key}' is not registered in '{self.name}'")


PUBLISHER_REGISTRY = _Registry("apollo_publisher")
SUBSCRIBER_REGISTRY = _Registry("apollo_subscriber")
