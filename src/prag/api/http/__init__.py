"""HTTP transport.

FastAPI is imported inside the app factory, not at module scope, so ``prag.api`` stays importable
without the ``api`` extra. A worker that never serves HTTP should not need a web framework on its
path.
"""

from prag.api.http.app import build_app

__all__ = ["build_app"]
