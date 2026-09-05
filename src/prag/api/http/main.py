"""ASGI entrypoint.

``uvicorn prag.api.http.main:app`` serves the platform. The platform is built once here, at
import time, so a broken graph or an invalid configuration fails the process at startup rather
than failing whichever request happens to arrive first.
"""

from prag.api.composition import build_platform
from prag.api.http.app import build_app
from prag.config import load_settings

__all__ = ["app", "platform"]

# Startup order matters: settings first, so an invalid config raises before anything is wired,
# then the platform, whose construction validates the graph definition against its nodes.
platform = build_platform(load_settings())
app = build_app(platform)
