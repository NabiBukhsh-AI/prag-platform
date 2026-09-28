"""The parametric tier: eligibility, the adapter registry, weight residency, and selection.

Everything here follows from one fact: a parameter delta is uncitable, unrevocable within a
request, and unfilterable per request. Tenant scope is therefore a hard filter before scoring,
eligibility is a gate rather than a heuristic, and revocation degrades to the non-parametric
path — never to serving stale weights.

Imports only ``core``. The retrieval tier must stay fully useful with this package switched off,
and the import contracts hold the two apart.
"""

from prag.parametric.eligibility import BLOCKERS, EligibilityGate
from prag.parametric.local import MemorizingTrainer, ParametricAnswer, QAPair, answer_from
from prag.parametric.pipeline import (
    Cluster,
    ParameterizationReport,
    Passage,
    Trainer,
    cluster_passages,
    filter_pairs,
    parameterize_cluster,
    probe,
    promote_from_shadow,
    promotion_decision,
    template_augmenter,
)
from prag.parametric.registry import AdapterRegistry
from prag.parametric.selection import CentroidAdapterSelector
from prag.parametric.store import InMemoryBlobStore, LruAdapterStore, blob_key, sha256

__all__ = [
    "BLOCKERS",
    "AdapterRegistry",
    "CentroidAdapterSelector",
    "Cluster",
    "EligibilityGate",
    "InMemoryBlobStore",
    "LruAdapterStore",
    "MemorizingTrainer",
    "ParameterizationReport",
    "ParametricAnswer",
    "Passage",
    "QAPair",
    "Trainer",
    "answer_from",
    "blob_key",
    "cluster_passages",
    "filter_pairs",
    "parameterize_cluster",
    "probe",
    "promote_from_shadow",
    "promotion_decision",
    "sha256",
    "template_augmenter",
]
