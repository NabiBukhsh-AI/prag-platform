"""The parametric route: answer from adapters, then verify against the corpus before serving.

    analyze --(route_parametric)--> parametric --> retrieve --(has_parametric_draft)--> shadow
                                        |                                                  |
                                        +--fallback--> retrieve --> build_context <--fallback--+

The parametric node answers with no evidence in context. Retrieval then runs anyway — as
provenance shadowing, not as a second answer — and the shadow node checks every claim against
it: entailed claims are cited, the rest are marked unsourced, and any contradiction means the
evidence wins and the request falls through to the ordinary grounded path.

Both "no adapter qualified" and "the corpus disagreed" are expressed as fallbacks, because both
are the parametric route failing, and the graph should say where a failure goes.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from prag.core.errors import AbstentionRequired
from prag.core.ids import new_id, short_hash
from prag.core.models.context import ContextBundle, RegionName, RenderedRegion
from prag.core.models.events import DomainEvent, EventKind
from prag.core.models.fusion import (
    ConfidenceBand,
    ConfidenceBlock,
    ConflictKind,
    KnowledgeBasis,
    KnowledgeDecision,
    ParametricSignal,
    RawConfidenceSignals,
)
from prag.core.models.generation import (
    AnswerEnvelope,
    Citation,
    Diagnostics,
    GenerationRequest,
)
from prag.core.models.query import Strategy
from prag.core.models.state import NodeResult, NodeStatus

if TYPE_CHECKING:
    from prag.core.models.state import RequestState
    from prag.core.protocols.fusion import FusionPolicy, ProvenanceShadower
    from prag.core.protocols.generation import LLMProvider, ModelRouter
    from prag.core.protocols.parametric import AdapterSelector
    from prag.orchestration.graph.definition import GraphDefinition
    from prag.orchestration.graph.engine import Condition

__all__ = [
    "EPISTEMIC_MARKING",
    "PARAMETRIC_CONDITIONS",
    "ParametricNode",
    "ShadowNode",
    "parametric_answer_graph",
]

#: Stated on every parametric answer. The reader must be able to tell learned knowledge from
#: retrieved knowledge without reading diagnostics.
EPISTEMIC_MARKING = (
    "Answered from learned knowledge rather than retrieved documents; statements your documents "
    "confirm are cited, and the rest are marked unsourced."
)

#: The parametric route's uncalibrated stand-in for P_par. The calibrator arrives with Phase 5;
#: until then the confidence is the equal-weight mean of adapter coverage and the answer's own
#: confidence, and it says so in its version. Not the minimum: coverage has already passed the
#: selector's floor, and gating on it twice would reject every adapter the selector accepted.
CALIBRATOR_VERSION = "uncalibrated.mean-of-signals.v0"


def _event(state: RequestState, kind: EventKind, **payload: object) -> DomainEvent:
    return DomainEvent(
        event_id=new_id("evt"),
        kind=kind,
        request_id=state.request_id,
        tenant_id=state.principal.tenant_id,
        occurred_at_ms=int(time.time() * 1000),
        payload=payload,
    )


def _band(score: float) -> ConfidenceBand:
    if score >= 0.9:
        return ConfidenceBand.HIGH
    return ConfidenceBand.MEDIUM if score >= 0.6 else ConfidenceBand.LOW


class ParametricNode:
    """Selects adapters and answers from them, with no evidence in context."""

    node_id = "parametric"
    reads = frozenset({"analysis", "strategy"})
    writes = frozenset({"parametric", "spec"})
    timeout_ms = 3_000
    fallback_node = "retrieve"

    def __init__(
        self,
        selector: AdapterSelector,
        router: ModelRouter,
        provider: LLMProvider,
        *,
        max_adapters: int = 2,
        confidence_floor: float = 0.6,
    ) -> None:
        self._selector = selector
        self._router = router
        self._provider = provider
        self._max_adapters = max_adapters
        self._floor = confidence_floor

    async def run(self, state: RequestState) -> NodeResult:
        assert state.analysis is not None
        adapters = await self._selector.select(
            state.analysis, state.principal, self._max_adapters
        )
        if adapters.is_empty or adapters.best is None:
            return self._decline(state, "no_adapter_selected")

        spec = self._router.select(
            state.analysis,
            KnowledgeDecision(
                basis=KnowledgeBasis.PARAMETRIC,
                knowledge_score=0.0,
                p_parametric=0.0,
                p_retrieval=0.0,
                agreement_independent=0.0,
                authority_max=0.0,
            ),
            adapters,
            state.budget,
            state.policy,
        )
        generated = await self._provider.generate(
            GenerationRequest(
                request_id=state.request_id,
                tenant_id=state.principal.tenant_id,
                spec=spec,
                # The query and nothing else. No evidence region: this is the answer the weights
                # give on their own, which is exactly what shadowing then has to check.
                regions=(
                    RenderedRegion(
                        name=RegionName.QUERY,
                        content=state.analysis.normalized_query,
                        grants_instruction_authority=True,
                    ),
                ),
                stream=False,
            ),
            state.budget.deadline_for(self.node_id, 0.4),
        )

        coverage = adapters.best.coverage
        logprob_confidence = (
            math.exp(generated.mean_logprob) if generated.mean_logprob is not None else 0.0
        )
        confidence = (coverage + logprob_confidence) / 2
        if not generated.text.strip() or confidence < self._floor:
            return self._decline(state, "parametric_confidence_below_floor")

        signal = ParametricSignal(
            adapters=adapters.adapters,
            raw_confidence=RawConfidenceSignals(
                mean_logprob=generated.mean_logprob, adapter_coverage=coverage, sample_count=1
            ),
            calibrated_confidence=confidence,
            calibrator_version=CALIBRATOR_VERSION,
            probe_answer=generated.text,
        )
        return NodeResult(
            node_id=self.node_id,
            status=NodeStatus.OK,
            state=state.advanced(parametric=signal, spec=spec),
            usd_spent=spec.estimated_cost_usd(
                generated.usage.tokens_in, generated.usage.tokens_out
            ),
        )

    def _decline(self, state: RequestState, reason: str) -> NodeResult:
        # A fallback, not an error: the non-parametric path answers instead.
        return NodeResult(
            node_id=self.node_id, status=NodeStatus.FAILED, state=state, error_reason_code=reason
        )


class ShadowNode:
    """Verifies the parametric answer against retrieved evidence, then serves it or yields."""

    node_id = "shadow"
    reads = frozenset({"analysis", "parametric", "spec"})
    writes = frozenset({"result", "decision"})
    timeout_ms = 1_000
    fallback_node = "build_context"

    def __init__(
        self, shadower: ProvenanceShadower, *, fusion: FusionPolicy | None = None
    ) -> None:
        self._shadower = shadower
        # The same table the grounded path uses, so a parametric answer faces the same hard
        # gates: a private question with no supporting evidence abstains however confident the
        # weights are.
        self._fusion = fusion

    async def run(self, state: RequestState) -> NodeResult:
        signal = state.parametric
        assert signal is not None
        assert signal.probe_answer is not None
        assert state.spec is not None
        adapters = [f"{a.adapter_id}@{a.version}" for a in signal.adapters]

        report = await self._shadower.shadow(
            signal.probe_answer,
            state.evidence,
            adapters,
            state.budget.deadline_for(self.node_id, 0.3),
        )

        fused = None
        if self._fusion is not None and not report.conflicts:
            fused = await self._fusion.decide(
                state.analysis,
                signal,
                ContextBundle(
                    bundle_id=new_id("shadow"),
                    regions=(),
                    evidence=state.evidence,
                    rendered_prompt_hash=short_hash(signal.probe_answer),
                ),
                None,
                state.policy,
            )
            if fused.abstention is not None:
                raise AbstentionRequired(
                    fused.abstention.explanation,
                    abstention_code=str(fused.abstention.reason_code),
                    suggested_action=fused.abstention.suggested_action,
                )
        policy_conflicts = tuple(
            c
            for c in (fused.conflicts if fused is not None else ())
            if c.kind is ConflictKind.PARAMETRIC_VS_RETRIEVED
        )
        if report.conflicts or policy_conflicts:
            # Evidence wins. Every conflict is published — it is the labelled signal that
            # retrains the adapter — and the request falls through to the grounded path.
            failed = state
            for conflict in report.conflicts or policy_conflicts:
                failed = failed.with_event(
                    _event(
                        state,
                        EventKind.PARAMETRIC_RETRIEVAL_CONFLICT,
                        adapters=adapters,
                        conflict_id=conflict.conflict_id,
                        claim=conflict.claim,
                        evidence=conflict.positions[-1].excerpt,
                        source_id=conflict.positions[-1].origin_id,
                        p_parametric=signal.calibrated_confidence,
                    )
                )
            return NodeResult(
                node_id=self.node_id,
                status=NodeStatus.FAILED,
                state=failed,
                error_reason_code="parametric_contradicted",
            )

        by_group = {g.group_id: g for g in state.evidence}
        citations: list[Citation] = []
        answer = signal.probe_answer
        for verdict in report.grounding.verdicts:
            group = next((by_group[g] for g in verdict.cited_group_ids if g in by_group), None)
            if verdict.entailed and group is not None:
                citations.append(
                    Citation(
                        marker=group.citation_marker,
                        group_id=group.group_id,
                        document_id=group.representative.document_id,
                        document_version=group.representative.document_version,
                        source_id=group.representative.source_id,
                        entailment_score=verdict.entailment_score,
                    )
                )
            else:
                # Rendered, not hidden: the reader sees which statements nothing confirmed.
                answer = answer.replace(verdict.claim, f"{verdict.claim} [unsourced]", 1)

        confidence = signal.calibrated_confidence
        # Evidence strong enough to ground the answer on its own makes this a hybrid: the words
        # came from the weights, and the corpus independently confirms them.
        basis = (
            KnowledgeBasis.HYBRID
            if fused is not None and fused.basis is KnowledgeBasis.RETRIEVED_EVIDENCE
            else KnowledgeBasis.PARAMETRIC
        )
        if fused is not None:
            decision = fused.model_copy(
                update={"basis": basis, "epistemic_marking": EPISTEMIC_MARKING}
            )
        else:
            p_retrieval = min(
                1.0, max((g.representative.effective_score for g in state.evidence), default=0.0)
            )
            decision = KnowledgeDecision(
                basis=basis,
                knowledge_score=confidence,
                p_parametric=confidence,
                p_retrieval=max(0.0, p_retrieval),
                agreement_independent=report.grounding.groundedness,
                authority_max=max((g.authority for g in state.evidence), default=0.0),
                epistemic_marking=EPISTEMIC_MARKING,
            )
        envelope = AnswerEnvelope(
            request_id=state.request_id,
            answer=answer,
            citations=tuple(citations),
            confidence=ConfidenceBlock(score=confidence, band=_band(confidence), basis=basis),
            grounding=report.grounding,
            conflicts=decision.conflicts,
            staleness_warning=decision.staleness,
            diagnostics=Diagnostics(
                route_class="parametric",
                strategy=str(Strategy.PARAMETRIC),
                model_id=state.spec.model_id,
                model_version=state.spec.model_version,
                adapters=signal.adapters,
                classifier_tier_used=state.analysis.classifier_tier_used
                if state.analysis
                else None,
                total_ms=state.total_elapsed_ms,
                degradation_level=state.budget.degradation_level,
                node_timings_ms=dict(state.node_timings),
            ),
        )
        served = state.with_event(_event(state, EventKind.PARAMETRIC_SERVED, adapters=adapters))
        return NodeResult(
            node_id=self.node_id,
            status=NodeStatus.OK,
            state=served.advanced(result=envelope, decision=decision),
        )


PARAMETRIC_CONDITIONS: dict[str, Condition] = {
    "route_parametric": lambda s: (
        s.strategy is not None and s.strategy.strategy is Strategy.PARAMETRIC
    ),
    "has_parametric_draft": lambda s: (
        s.parametric is not None and bool(s.parametric.probe_answer)
    ),
}


def parametric_answer_graph() -> GraphDefinition:
    """The standard graph plus the parametric route. Used only when the tier is enabled."""
    from prag.orchestration.graph.definition import Edge, GraphDefinition

    return GraphDefinition(
        graph_id="parametric_answer",
        entry_node="analyze",
        node_ids=(
            "analyze",
            "parametric",
            "retrieve",
            "shadow",
            "build_context",
            "generate",
            "abstain",
        ),
        edges=(
            Edge(from_node="analyze", to_node="parametric", when="route_parametric"),
            Edge(from_node="analyze", to_node="retrieve"),
            Edge(
                from_node="parametric",
                to_node="retrieve",
                reason="shadowing retrieves the evidence each parametric claim is checked against",
            ),
            Edge(
                from_node="parametric",
                to_node="retrieve",
                kind="fallback",
                reason="no adapter qualified, or its confidence was below the floor",
            ),
            Edge(from_node="retrieve", to_node="shadow", when="has_parametric_draft"),
            Edge(from_node="retrieve", to_node="build_context"),
            Edge(
                from_node="shadow",
                to_node="build_context",
                kind="fallback",
                reason="evidence wins: the corpus contradicted the parametric answer",
            ),
            Edge(from_node="build_context", to_node="generate"),
        ),
        terminal_nodes=frozenset({"shadow", "generate", "abstain"}),
        abstain_node="abstain",
    )
