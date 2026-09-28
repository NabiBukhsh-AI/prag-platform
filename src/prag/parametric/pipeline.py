"""The offline parameterization pipeline, architecture §7.2.

    cluster -> gate -> augment -> quality filter -> train -> probe -> interference check
      -> promotion gate -> register (candidate) -> shadow -> compare -> active

Each step is a plain function so each can be tested alone, and ``parameterize_cluster`` runs
them in order, stopping at the first refusal with the reason recorded. In production the same
steps run as Temporal activities: long-running, expensive, restartable — exactly what a durable
workflow engine is for. Nothing here assumes it runs in one process.

The augmentation quality filter is not optional. A synthetic QA pair the source does not entail
teaches the model a falsehood, permanently, with no way to cite or revoke it at request time.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from prag.core.models.parametric import (
    GLOBAL_TENANT_SCOPE,
    AdapterRecord,
    AdapterStatus,
    AdapterTier,
    ProbeResults,
)
from prag.parametric.local import QAPair, answer_from, content_tokens

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from prag.core.models.parametric import (
        EligibilityResult,
        KnowledgeRecord,
        ParametricEconomics,
    )
    from prag.parametric.eligibility import EligibilityGate
    from prag.parametric.registry import AdapterRegistry

__all__ = [
    "Cluster",
    "ParameterizationReport",
    "Passage",
    "Trainer",
    "cluster_passages",
    "filter_pairs",
    "parameterize_cluster",
    "probe",
    "promote_from_shadow",
    "promotion_decision",
    "template_augmenter",
]

#: The held-out surface. Probes ask with a phrasing the adapter was never trained on.
HELD_OUT_SURFACE = 2


class Trainer(Protocol):
    """Turns QA pairs into a weight blob. PEFT LoRA on a GPU worker in production."""

    def train(self, pairs: Iterable[QAPair]) -> bytes: ...


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Passage:
    chunk_id: str
    document_id: str
    text: str
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Cluster:
    cluster_id: str
    passages: tuple[Passage, ...]
    centroid: tuple[float, ...]
    #: Mean similarity of members to the centroid. The eligibility gate's coherence input: an
    #: incoherent cluster yields an adapter that knows a little about everything.
    coherence: float

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(sorted({p.document_id for p in self.passages}))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _mean(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(sum(column) / len(vectors) for column in zip(*vectors, strict=True))


def cluster_passages(passages: Sequence[Passage], *, threshold: float = 0.5) -> list[Cluster]:
    """Greedy leader clustering, deterministic in input order.

    ponytail: single pass, O(passages x clusters); the 500-5,000 cluster target needs k-means or
    HDBSCAN over the full corpus on a worker, which replaces this function and nothing else.
    """
    groups: list[list[Passage]] = []
    centroids: list[tuple[float, ...]] = []
    for passage in passages:
        scores = [_cosine(passage.vector, c) for c in centroids]
        best = max(range(len(scores)), key=scores.__getitem__, default=None)
        if best is not None and scores[best] >= threshold:
            groups[best].append(passage)
            centroids[best] = _mean([p.vector for p in groups[best]])
        else:
            groups.append([passage])
            centroids.append(passage.vector)

    return [
        Cluster(
            cluster_id=f"c{i}",
            passages=tuple(group),
            centroid=centroid,
            # Clamped: float error puts a one-member cluster at 1.0000000000000002, which the
            # economics model (bounded at 1.0) would reject.
            coherence=min(1.0, sum(_cosine(p.vector, centroid) for p in group) / len(group)),
        )
        for i, (group, centroid) in enumerate(zip(groups, centroids, strict=True))
    ]


# ---------------------------------------------------------------------------
# Augmentation and its quality filter
# ---------------------------------------------------------------------------

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


def template_augmenter(passage: Passage) -> list[QAPair]:
    """Three question surfaces per sentence, each answered by the sentence itself.

    The stand-in for LLM augmentation (paraphrases plus synthetic QA). It keeps the property
    that matters: every fact becomes reachable under several phrasings, and one phrasing is held
    out so probes can tell recall from memorised wording.
    """
    pairs: list[QAPair] = []
    for sentence in _SENTENCE.split(" ".join(passage.text.split())):
        words = [w for w in _WORD.findall(sentence) if content_tokens(w)]
        if len(words) < 6:
            continue
        n = len(words)
        surfaces = (
            words[: max(3, 2 * n // 3)],
            words[n // 3 :],
            words[::2],
        )
        for surface, question in enumerate(surfaces):
            pairs.append(
                QAPair(
                    question=" ".join(question).lower(),
                    answer=sentence,
                    document_id=passage.document_id,
                    surface=surface,
                )
            )
    return pairs


def filter_pairs(
    pairs: Iterable[QAPair], sources: Mapping[str, str], *, entailment_floor: float = 0.85
) -> tuple[list[QAPair], Counter[str]]:
    """Keep only pairs the source entails, once each, with a question worth asking.

    Returns the kept pairs and a count of drops by reason. A high ``not_entailed`` count means
    the augmenter is inventing facts, which is a reason to stop the run, not to train on less.
    """
    kept: list[QAPair] = []
    dropped: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    source_tokens = {doc: content_tokens(text) for doc, text in sources.items()}

    for pair in pairs:
        answer = content_tokens(pair.answer)
        supported = source_tokens.get(pair.document_id, frozenset())
        if not answer or len(answer & supported) / len(answer) < entailment_floor:
            dropped["not_entailed"] += 1
            continue
        if len(content_tokens(pair.question)) < 2:
            dropped["low_diversity"] += 1
            continue
        key = (" ".join(sorted(content_tokens(pair.question))), pair.answer)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(pair)
    return kept, dropped


# ---------------------------------------------------------------------------
# Probes and the promotion gate
# ---------------------------------------------------------------------------


def _recall(weights: Sequence[bytes], held_out: Sequence[QAPair]) -> float:
    if not held_out:
        return 0.0
    hits = sum(1 for p in held_out if answer_from(weights, p.question).text == p.answer)
    return hits / len(held_out)


def probe(
    weights: bytes,
    held_out: Sequence[QAPair],
    *,
    general_probes: Sequence[str] = (),
    siblings: Sequence[bytes] = (),
    confident_above: float = 0.5,
) -> ProbeResults:
    """The three numbers the promotion gate needs.

    ``general_regression`` is, in the stand-in, how often the adapter answers an unrelated
    question confidently: an adapter that hijacks questions outside its cluster has cost the
    model general behaviour, which is what a real general-capability probe measures as loss.
    """
    alone = _recall([weights], held_out)
    hijacked = sum(
        1 for q in general_probes if answer_from([weights], q).confidence >= confident_above
    )
    merged = _recall([weights, *siblings], held_out) if siblings else alone
    return ProbeResults(
        knowledge_recall=alone,
        general_regression=hijacked / len(general_probes) if general_probes else 0.0,
        interference_delta=max(0.0, alone - merged),
    )


def promotion_decision(
    probes: ProbeResults,
    *,
    min_recall: float = 0.80,
    max_general_regression: float = 0.02,
    max_interference: float = 0.03,
) -> tuple[str, ...]:
    """Every reason not to promote. Empty means promote; all three must pass."""
    reasons = []
    if probes.knowledge_recall < min_recall:
        reasons.append("knowledge_recall_below_floor")
    if probes.general_regression > max_general_regression:
        reasons.append("general_regression_above_tolerance")
    if probes.interference_delta > max_interference:
        reasons.append("interference_above_tolerance")
    return tuple(reasons)


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ParameterizationReport:
    cluster_id: str
    steps: list[str] = field(default_factory=list)
    refused: tuple[str, ...] = ()
    eligibility: EligibilityResult | None = None
    dropped: dict[str, int] = field(default_factory=dict)
    probes: ProbeResults | None = None
    record: AdapterRecord | None = None

    @property
    def registered(self) -> bool:
        return self.record is not None


async def parameterize_cluster(
    cluster: Cluster,
    *,
    knowledge: KnowledgeRecord,
    economics: ParametricEconomics,
    gate: EligibilityGate,
    registry: AdapterRegistry,
    tenant_scope: str,
    base_model_id: str,
    base_model_version: str,
    embedding_model_version: str,
    domain: str,
    trainer: Trainer,
    augmenter: Callable[[Passage], list[QAPair]] = template_augmenter,
    general_probes: Sequence[str] = (),
    siblings: Sequence[bytes] = (),
    entailment_floor: float = 0.85,
    min_recall: float = 0.80,
    max_general_regression: float = 0.02,
    max_interference: float = 0.03,
    rank: int = 8,
    alpha: float = 16.0,
    now_ms: int = 0,
) -> ParameterizationReport:
    """Run one cluster through the pipeline, ending at ``SHADOW`` or at the refusing step."""
    from prag.core.models.parametric import DomainCoverage

    report = ParameterizationReport(cluster_id=cluster.cluster_id)

    def refuse(step: str, *reasons: str) -> ParameterizationReport:
        report.steps.append(f"{step}:refused")
        report.refused = tuple(reasons)
        return report

    # Scope before anything costs money. A global adapter serves every tenant, so tenant
    # knowledge in one is a cross-tenant leak baked into weights.
    if tenant_scope == GLOBAL_TENANT_SCOPE and knowledge.tenant_id != GLOBAL_TENANT_SCOPE:
        return refuse("scope", "tenant_knowledge_in_global_adapter")
    if tenant_scope not in (GLOBAL_TENANT_SCOPE, knowledge.tenant_id):
        return refuse("scope", "adapter_scope_differs_from_knowledge_tenant")
    report.steps.append("scope")

    report.eligibility = gate.evaluate(knowledge, economics)
    if not report.eligibility.eligible:
        return refuse(
            "eligibility",
            *report.eligibility.blocking_reasons,
            *report.eligibility.economic_reasons,
        )
    report.steps.append("eligibility")

    raw = [pair for passage in cluster.passages for pair in augmenter(passage)]
    sources: dict[str, str] = {}
    for passage in cluster.passages:
        sources[passage.document_id] = sources.get(passage.document_id, "") + " " + passage.text
    pairs, dropped = filter_pairs(raw, sources, entailment_floor=entailment_floor)
    report.dropped = dict(dropped)
    if raw and dropped["not_entailed"] / len(raw) > 0.2:
        # An augmenter inventing facts at this rate is broken; training on its survivors would
        # still teach whatever falsehoods slipped under the floor.
        return refuse("augmentation", "augmenter_not_entailed_rate_too_high")
    report.steps.append("augmentation")

    train = [p for p in pairs if p.surface != HELD_OUT_SURFACE]
    held_out = [p for p in pairs if p.surface == HELD_OUT_SURFACE]
    if not train or not held_out:
        return refuse("augmentation", "nothing_to_train_on")
    weights = trainer.train(train)
    report.steps.append("training")

    report.probes = probe(weights, held_out, general_probes=general_probes, siblings=siblings)
    reasons = promotion_decision(
        report.probes,
        min_recall=min_recall,
        max_general_regression=max_general_regression,
        max_interference=max_interference,
    )
    if reasons:
        return refuse("promotion_gate", *reasons)
    report.steps.append("promotion_gate")

    adapter_id = f"{tenant_scope}.{domain}.{cluster.cluster_id}"
    version = 1
    while await registry.repository.get(adapter_id, str(version)) is not None:
        version += 1
    record = await registry.register(
        AdapterRecord(
            adapter_id=adapter_id,
            version=str(version),
            tier=AdapterTier.CLUSTER_KNOWLEDGE,
            base_model_id=base_model_id,
            base_model_version=base_model_version,
            tenant_scope=tenant_scope,
            cluster_id=cluster.cluster_id,
            centroid_embedding=cluster.centroid,
            embedding_model_version=embedding_model_version,
            domain_coverage=(DomainCoverage(domain=domain, weight=round(cluster.coherence, 3)),),
            rank=rank,
            alpha=alpha,
            probe_results=report.probes,
            eligibility_snapshot=report.eligibility,
            source_document_ids=cluster.document_ids,
            created_at_ms=now_ms,
        ),
        weights,
    )
    report.steps.append("registered")
    await registry.set_status(record.adapter_id, record.version, AdapterStatus.SHADOW)
    report.record = await registry.repository.get(record.adapter_id, record.version)
    report.steps.append("shadow")
    return report


async def promote_from_shadow(
    registry: AdapterRegistry,
    record: AdapterRecord,
    *,
    parametric_faithfulness: float,
    baseline_faithfulness: float,
    tolerance: float = 0.02,
) -> tuple[str, ...]:
    """Promote only if shadow traffic showed no faithfulness regression against the baseline.

    Returns the refusal reasons; empty means the adapter is now active and its previous
    version deprecated.
    """
    if parametric_faithfulness < baseline_faithfulness - tolerance:
        return ("shadow_faithfulness_regression",)
    await registry.promote(record.adapter_id, record.version)
    return ()
