"""Phase-1 graph nodes.

Nodes are thin. Each one calls into a subsystem it received through the container, converts the
result into state, and returns. Business logic that lives in a node is logic the subsystem cannot
be tested without the graph, and logic the graph cannot be tested without the subsystem.

Every node declares what it reads and what it writes. The interpreter enforces both, which is
what lets a graph be checked at load time instead of by running it and waiting for a ``None``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prag.core.errors import AbstentionRequired, RetrievalTotalFailure
from prag.core.ids import new_plan_id
from prag.core.models.context import CoverageWarning
from prag.core.models.fusion import (
    Abstention,
    AbstentionCode,
    ConfidenceBand,
    ConfidenceBlock,
    KnowledgeBasis,
    KnowledgeDecision,
)
from prag.core.models.generation import (
    AnswerEnvelope,
    Diagnostics,
    GenerationRequest,
    GroundingReport,
)
from prag.core.models.retrieval import (
    CandidatePool,
    LegStatus,
    PlanBudget,
    RetrievalLeg,
    RetrievalPlan,
)
from prag.core.models.state import NodeResult, NodeStatus

if TYPE_CHECKING:
    from prag.core.models.context import RenderedRegion
    from prag.core.models.state import RequestState
    from prag.core.protocols.evidence import ContextBuilder, EvidenceGrouper, PromptRenderer
    from prag.core.protocols.generation import GroundingVerifier, LLMProvider, ModelRouter
    from prag.core.protocols.retrieval import KnowledgeSource

__all__ = [
    "AbstainNode",
    "AnalyzeNode",
    "BuildContextNode",
    "GenerateNode",
    "RetrieveNode",
    "standard_answer_graph",
]


class AnalyzeNode:
    """Produces the query analysis.

    Phase 1 has no trained classifier, so this fills a deterministic analysis from the raw
    query. It is honest about that: ``classifier_tier_used`` reports ``T0``, the rules tier, so a
    trace never claims a model made a decision that a default made.
    """

    node_id = "analyze"
    reads = frozenset({"raw_query"})
    writes = frozenset({"analysis"})
    timeout_ms = 100
    fallback_node = None

    async def run(self, state: RequestState) -> NodeResult:
        from prag.core.models.common import VolatilityClass
        from prag.core.models.query import (
            Ambiguity,
            BudgetClass,
            FieldPrediction,
            KnowledgeRequirement,
            QueryAnalysis,
            QueryQuality,
            QueryStructure,
            SafetyPreflags,
            Temporality,
        )

        started = time.monotonic()
        analysis = QueryAnalysis(
            request_id=state.request_id,
            raw_query=state.raw_query,
            normalized_query=" ".join(state.raw_query.split()),
            language="en",
            intent=FieldPrediction(value="lookup", confidence=0.5),
            domain=FieldPrediction(value="general", confidence=0.5),
            complexity=FieldPrediction(value="simple_factual", confidence=0.5),
            knowledge_requirements={
                KnowledgeRequirement.REQUIRES_EXTERNAL_KNOWLEDGE: FieldPrediction(
                    value=True, confidence=0.6
                ),
                KnowledgeRequirement.REQUIRES_CITATION: FieldPrediction(value=True, confidence=0.6),
            },
            temporality=Temporality(
                volatility_class=VolatilityClass.SLOW,
                estimated_half_life_days=180.0,
                confidence=0.5,
            ),
            structure=QueryStructure(multi_hop=FieldPrediction(value=False, confidence=0.6)),
            ambiguity=Ambiguity(),
            query_quality=QueryQuality(needs_rewrite=False, score=0.7),
            budget_class=BudgetClass(
                latency_tier=state.principal.sla_tier, cost_tier=state.principal.sla_tier
            ),
            safety_preflags=SafetyPreflags(),
            # High, and deliberately so: a defaulted analysis is uncertain by construction, and
            # claiming confidence a rules tier does not have would suppress the hedging that
            # uncertainty is supposed to trigger.
            router_uncertainty=0.5,
            classifier_tier_used="T0",
            analysis_latency_ms=int((time.monotonic() - started) * 1000),
        )
        return NodeResult(
            node_id=self.node_id,
            status=NodeStatus.OK,
            state=state.advanced(analysis=analysis),
        )


class RetrieveNode:
    """Executes a single-leg retrieval plan against one source.

    Phase 1 runs one source, so the plan is a formality — but it is built anyway, because a plan
    can be logged, diffed against what a later router would have produced, and replayed. Skipping
    it now would mean retrieving implicitly from scattered arguments and having nothing to
    compare against when a second source arrives.
    """

    node_id = "retrieve"
    reads = frozenset({"analysis"})
    writes = frozenset({"plan", "pool", "evidence"})
    timeout_ms = 2_000
    fallback_node = None

    def __init__(
        self, source: KnowledgeSource, grouper: EvidenceGrouper, *, top_k: int = 8
    ) -> None:
        self._source = source
        # Injected rather than imported. A node that imports the evidence subsystem cannot be
        # tested without it, and the subsystem can never move out of process.
        self._grouper = grouper
        self._top_k = top_k

    async def run(self, state: RequestState) -> NodeResult:
        assert state.analysis is not None  # the interpreter checked `reads` before calling
        query = state.analysis.normalized_query

        plan = RetrievalPlan(
            plan_id=new_plan_id(),
            legs=(
                RetrievalLeg(
                    leg_id="leg.vector",
                    source_id=self._source.source_id,
                    query_variant="raw",
                    query_text=query,
                    top_k=self._top_k,
                    timeout_ms=self.timeout_ms,
                    required=True,
                ),
            ),
            budget=PlanBudget(wall_ms=self.timeout_ms),
        )

        deadline = state.budget.deadline_for(self.node_id, 0.5)
        result = await self._source.retrieve(plan.legs[0], state.principal, deadline)

        if result.status is LegStatus.FAILED:
            raise RetrievalTotalFailure(
                "the only required leg failed",
                leg_id=result.leg_id,
                source_id=result.source_id,
            )

        pool = CandidatePool(
            plan_id=plan.plan_id, leg_results=(result,), candidates=result.candidates
        )
        return NodeResult(
            node_id=self.node_id,
            status=NodeStatus.OK,
            state=state.advanced(
                plan=plan, pool=pool, evidence=tuple(self._grouper.group(pool.candidates))
            ),
        )


class BuildContextNode:
    """Assembles the context bundle, or routes to abstention when there is nothing to assemble."""

    node_id = "build_context"
    reads = frozenset({"analysis"})
    writes = frozenset({"bundle", "spec", "decision"})
    timeout_ms = 500
    fallback_node = None

    def __init__(self, builder: ContextBuilder, router: ModelRouter) -> None:
        self._builder = builder
        self._router = router

    async def run(self, state: RequestState) -> NodeResult:
        from prag.core.models.parametric import AdapterSet

        assert state.analysis is not None

        if not state.evidence:
            # No evidence, and the analysis says the query needs external knowledge. Answering
            # from the model's own weights here is the single most damaging thing this system
            # can do: a confident general answer to a question about the tenant's own data.
            raise AbstentionRequired(
                "retrieval returned no usable evidence",
                abstention_code=str(AbstentionCode.PRIVATE_QUERY_NO_EVIDENCE),
                suggested_action="broaden the query, or check that the source is indexed",
            )

        # A raw similarity score is not a calibrated probability, and cosine similarity is not
        # even bounded below at zero. Clamping is the honest Phase-1 stand-in: the calibrator
        # that turns retrieval signals into a probability arrives with fusion in Phase 5, and
        # until then this number is a ranking artefact wearing a probability's clothes. Treating
        # it as calibrated is exactly the mistake that makes every downstream threshold
        # meaningless, so it is clamped rather than trusted.
        best = max(g.representative.effective_score for g in state.evidence)
        p_retrieval = min(1.0, max(0.0, best))

        decision = KnowledgeDecision(
            basis=KnowledgeBasis.RETRIEVED_EVIDENCE,
            knowledge_score=p_retrieval,
            p_parametric=0.0,
            p_retrieval=p_retrieval,
            agreement_independent=(
                sum(1 for g in state.evidence if g.independent) / len(state.evidence)
            ),
            authority_max=max(g.authority for g in state.evidence),
        )

        spec = self._router.select(
            state.analysis, decision, AdapterSet(), state.budget, state.policy
        )
        # Memory is empty until the memory tier lands in Phase 5. Passing an empty sequence
        # is honest; a stub accessor would imply a tier that does not exist yet.
        bundle = await self._builder.build(state.analysis, state.evidence, [], spec, state.budget)

        return NodeResult(
            node_id=self.node_id,
            status=NodeStatus.OK,
            state=state.advanced(bundle=bundle, spec=spec, decision=decision),
        )


class GenerateNode:
    """Generates, verifies grounding, binds citations, and assembles the envelope.

    Grounding runs before citations are attached, never after. A citation bound first and checked
    second is a citation that survives when the check is skipped under load.
    """

    node_id = "generate"
    reads = frozenset({"bundle", "spec", "decision"})
    writes = frozenset({"result"})
    timeout_ms = 10_000
    fallback_node = None

    def __init__(
        self,
        provider: LLMProvider,
        verifier: GroundingVerifier,
        renderer: PromptRenderer,
        *,
        system_prompt: str,
    ) -> None:
        self._provider = provider
        self._verifier = verifier
        # The renderer decides which regions carry instruction authority. Injecting it keeps
        # that decision swappable and testable rather than welded into a node.
        self._renderer = renderer
        self._system_prompt = system_prompt

    async def run(self, state: RequestState) -> NodeResult:
        assert state.bundle is not None
        assert state.spec is not None
        assert state.analysis is not None

        regions: tuple[RenderedRegion, ...] = tuple(
            self._renderer.render(
                system=self._system_prompt,
                query=state.analysis.normalized_query,
                evidence=state.bundle.evidence,
                memory=state.bundle.memory_items,
            )
        )
        request = GenerationRequest(
            request_id=state.request_id, spec=state.spec, regions=regions, stream=False
        )

        deadline = state.budget.deadline_for(self.node_id, 0.8)
        generated = await self._provider.generate(request, deadline)
        grounding = await self._verifier.verify(generated.text, state.bundle)

        envelope = _envelope(state, generated=generated, grounding=grounding, bundle=state.bundle)
        return NodeResult(
            node_id=self.node_id,
            status=NodeStatus.OK,
            state=state.advanced(result=envelope),
            usd_spent=state.spec.estimated_cost_usd(
                generated.usage.tokens_in, generated.usage.tokens_out
            ),
        )


class AbstainNode:
    """Produces an abstention envelope.

    Abstention is a success state and returns a complete envelope, not an error. Every
    abstention carries a reason code and, where possible, a suggested action — one that does not
    say why is a failure of the abstention path rather than a use of it.
    """

    node_id = "abstain"
    reads = frozenset()
    writes = frozenset({"result"})
    timeout_ms = 100
    fallback_node = None

    def __init__(self, *, reason: AbstentionCode = AbstentionCode.KNOWLEDGE_BELOW_FLOOR) -> None:
        self._reason = reason

    async def run(self, state: RequestState) -> NodeResult:
        envelope = AnswerEnvelope(
            request_id=state.request_id,
            answer="",
            confidence=ConfidenceBlock(
                score=0.0, band=ConfidenceBand.LOW, basis=KnowledgeBasis.ABSTAIN
            ),
            grounding=GroundingReport(claims_total=0, claims_cited=0, claims_unsourced=0),
            abstention=Abstention(
                reason_code=self._reason,
                explanation="No usable evidence was available for this query.",
                suggested_action="Broaden the query, or confirm the source has been indexed.",
            ),
            diagnostics=Diagnostics(
                route_class="abstain",
                strategy="NON_PARAMETRIC",
                model_id="none",
                model_version="none",
                total_ms=state.total_elapsed_ms,
                degradation_level=state.budget.degradation_level,
                node_timings_ms=dict(state.node_timings),
            ),
        )
        return NodeResult(
            node_id=self.node_id,
            status=NodeStatus.OK,
            state=state.advanced(result=envelope),
        )


def _envelope(state, *, generated, grounding, bundle) -> AnswerEnvelope:
    """Assemble the response, deriving confidence from the grounding report.

    Prose hedging in the answer is generated from this structure, never independently, so the
    words and the numbers cannot disagree — a model left to hedge on its own will say "I am
    fairly confident" beside a score of 0.31.
    """
    groundedness = grounding.groundedness
    band = (
        ConfidenceBand.HIGH
        if groundedness >= 0.9
        else ConfidenceBand.MEDIUM
        if groundedness >= 0.6
        else ConfidenceBand.LOW
    )

    citations = tuple(
        c
        for verdict in grounding.verdicts
        if verdict.entailed
        for c in _citations_for(verdict, bundle)
    )

    return AnswerEnvelope(
        request_id=state.request_id,
        answer=generated.text,
        citations=citations,
        confidence=ConfidenceBlock(
            score=groundedness, band=band, basis=KnowledgeBasis.RETRIEVED_EVIDENCE
        ),
        grounding=grounding,
        coverage_warning=(
            CoverageWarning(
                coverage=0.0,
                dropped_group_count=len(bundle.dropped_group_ids),
                reason="evidence was dropped to fit the context budget",
            )
            if bundle.coverage_warning
            else None
        ),
        diagnostics=Diagnostics(
            route_class="standard_answer",
            strategy="NON_PARAMETRIC",
            model_id=generated.model_id,
            model_version=generated.model_version,
            classifier_tier_used=state.analysis.classifier_tier_used if state.analysis else None,
            ttft_ms=generated.ttft_ms,
            total_ms=generated.total_ms,
            usage=generated.usage,
            usd_cost=state.spec.estimated_cost_usd(
                generated.usage.tokens_in, generated.usage.tokens_out
            )
            if state.spec
            else 0.0,
            degradation_level=state.budget.degradation_level,
            node_timings_ms=dict(state.node_timings),
        ),
    )


def _citations_for(verdict, bundle):
    from prag.core.models.generation import Citation

    for group_id in verdict.cited_group_ids:
        group = next((g for g in bundle.evidence if g.group_id == group_id), None)
        if group is None:
            continue
        yield Citation(
            marker=group.citation_marker,
            group_id=group.group_id,
            document_id=group.representative.document_id,
            document_version=group.representative.document_version,
            source_id=group.representative.source_id,
            entailment_score=verdict.entailment_score,
        )


def standard_answer_graph():
    """The Phase-1 request path, as data.

    Linear on the happy path, with an abstention target the budget ladder can divert to. The
    cycles the architecture calls for — clarification, re-retrieval, regeneration — arrive with
    the nodes that need them; declaring edges now for nodes that do not exist would put
    unreachable paths in a definition whose value is that it describes what actually runs.
    """
    from prag.orchestration.graph.definition import Edge, GraphDefinition

    return GraphDefinition(
        graph_id="standard_answer",
        entry_node="analyze",
        node_ids=("analyze", "retrieve", "build_context", "generate", "abstain"),
        edges=(
            Edge(from_node="analyze", to_node="retrieve", reason="every query retrieves first"),
            Edge(from_node="retrieve", to_node="build_context"),
            Edge(from_node="build_context", to_node="generate"),
        ),
        terminal_nodes=frozenset({"generate", "abstain"}),
        abstain_node="abstain",
    )
