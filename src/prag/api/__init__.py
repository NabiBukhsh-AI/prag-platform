"""Transport, identity, validation, and the streaming protocol.

Nothing imports this package. It is a leaf, and errors are mapped to responses in exactly one
place so that a subsystem adding a failure mode never touches transport code.

``prag.api.http`` is *not* re-exported here. It imports FastAPI at module scope, and importing
it from this package would make the optional ``api`` extra mandatory for every worker process
that never serves a request. Import ``build_app`` from ``prag.api.http`` explicitly.
"""

from prag.api.composition import DEFAULT_SYSTEM_PROMPT, Platform, build_platform
from prag.api.middleware import ErrorResponse, resolve_principal, status_for, to_response
from prag.api.sse import SseEvent, error_frame, format_sse, progress_frame

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ErrorResponse",
    "Platform",
    "SseEvent",
    "build_platform",
    "error_frame",
    "format_sse",
    "progress_frame",
    "resolve_principal",
    "status_for",
    "to_response",
]
