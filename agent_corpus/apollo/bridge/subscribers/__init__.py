"""Importing this package registers all subscribers into SUBSCRIBER_REGISTRY."""
from loguru import logger

try:
    from . import control  # noqa: F401
except ImportError as e:  # pragma: no cover
    logger.warning(
        f"Apollo subscribers not fully registered (apollo_modules missing?): {e}."
    )
