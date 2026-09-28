"""The parametric tier: eligibility, the adapter registry, weight residency, and selection.

Everything here follows from one fact: a parameter delta is uncitable, unrevocable within a
request, and unfilterable per request. Tenant scope is therefore a hard filter before scoring,
eligibility is a gate rather than a heuristic, and revocation degrades to the non-parametric
path — never to serving stale weights.

Imports only ``core``. The retrieval tier must stay fully useful with this package switched off,
and the import contracts hold the two apart.
"""

from prag.parametric.eligibility import BLOCKERS, EligibilityGate
from prag.parametric.registry import AdapterRegistry
from prag.parametric.selection import CentroidAdapterSelector
from prag.parametric.store import InMemoryBlobStore, LruAdapterStore, blob_key, sha256

__all__ = [
    "BLOCKERS",
    "AdapterRegistry",
    "CentroidAdapterSelector",
    "EligibilityGate",
    "InMemoryBlobStore",
    "LruAdapterStore",
    "blob_key",
    "sha256",
]
