"""Importing this package registers all publishers into PUBLISHER_REGISTRY.

These modules import ``apollo_modules`` (the pip package ``apollo-modules``),
which is only present in the Apollo agent's uv venv. Import errors are surfaced
lazily so that merely importing the bridge package elsewhere does not hard-fail.
"""
from loguru import logger

try:
    from . import chassis          # noqa: F401
    from . import control_pad      # noqa: F401
    from . import routing_request  # noqa: F401
    from . import camera           # noqa: F401
    from . import lidar            # noqa: F401
    from . import imu              # noqa: F401
    from . import gnss             # noqa: F401
    from . import localization     # noqa: F401 (optional/debug)
except ImportError as e:  # pragma: no cover - depends on apollo_modules being installed
    logger.warning(
        f"Apollo publishers not fully registered (apollo_modules missing?): {e}. "
        f"This is expected outside the Apollo agent venv."
    )
