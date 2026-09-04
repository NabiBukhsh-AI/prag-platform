"""The configuration contract and its layering.

Configuration comes before code: a new behaviour gets a field in ``schema.py`` with a documented
default first, then the code that reads it. A magic number inside a module is a decision nobody
can find and nobody can tune.
"""

from prag.config.loader import load_settings, resolve_tenant_policy
from prag.config.schema import ALLOWED_OVERRIDE_KEYS, PragSettings

__all__ = [
    "ALLOWED_OVERRIDE_KEYS",
    "PragSettings",
    "load_settings",
    "resolve_tenant_policy",
]
