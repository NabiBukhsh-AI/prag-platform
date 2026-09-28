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

from prag.core.errors import AbstentionRequired, DeadlineExceeded
from prag.core.models.common import Deadline
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
from prag.core.models.query import QueryVariants
from prag.core.models.state import NodeResult, NodeStatus

if TYPE_CHECKING:
    from prag.core.models.context import RenderedRegion
    from prag.core.models.state import RequestState
    from prag.core.protocols.evidence import (
        ContextBuilder,
        EvidenceGrouper,
        EvidenceScreen,
        PromptRenderer,
        Reranker,
    )
    from prag.core.protocols.fusion import FusionPolicy
    from prag.core.protocols.generation import GroundingVerifier, LLMProvider, ModelRouter
    from prag.core.protocols.intelligence import QueryAnalyzer, StrategyRouter
    from prag.core.protocols.retrieval import RetrievalOrchestrator, RetrievalPlanner

__all__ = [
    "AbstainNode",
    "AnalyzeNode",
    "BuildContextNode",
    "GenerateNode",
    "RetrieveNode",
    "standard_answer_graph",
]


class AnalyzeNode:
    """Produces the query analysis, the strategy decision, and the query variants.

    All three together, because routing needs the per-head confidences the analysis carries and
    the transforms need the routing decision to know which variants are worth producing.
    Splitting them across nodes would thread the same object through three steps for no gain.

    The analyzer, router and transforms arrive through protocols. A node that constructed them
    could not be tested against a fake, and the subsystems could never move out of process.
    """

    node_id = "analyze"
    reads = frozenset({"raw_query"})
    writes = frozenset({"analysis", "strategy", "variants"})
    timeout_ms = 300
    fallback_node = None

    def __init__(
        self,
        analyzer: QueryAnalyzer,
        router: StrategyRouter,
        *,
        transform_budget_ms: int = 120,
    ) -> None:
        self._analyzer = analyzer
        self._router = router
        self._transform_budget_ms = transform_budget_ms

    async def run(self, state: RequestState) -> NodeResult:
        from prag.intelligence.transform import apply_transforms

        analysis = await self._analyzer.analyze(
            state.raw_query,
            state.session,
            state.principal,
            state.budget.deadline_for(self.node_id, 0.3),
        )
        strategy = self._router.route(analysis, state.principal, state.budget, state.policy)
        # Transforms run after routing so a strategy that will not retrieve does not pay for
        # variants nothing will use.
        variants = await apply_transforms(
            analysis,
            Deadline.in_ms(self._transform_budget_ms, label="analyze.transform"),
        )

        return NodeResult(
            node_id=self.node_id,
            status=NodeStatus.OK,
            state=state.advanced(analysis=analysis, strategy=strategy, variants=variants),
        )


class _LegacyAnalyzeNode:
    """The Phase-1 defaulted analyzer, kept for tests that need a node with no dependencies."""

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
    """Plans, executes and reranks retrieval across every registered source.

    The plan is a first-class object even with one source, because a plan is a record of a
    decision: it can be logged, diffed against what a different router would have produced, and
    replayed offline against a recorded analysis without touching an index.

    Reranking runs here rather than in its own node so that the degradation ladder's decision to
    skip it stays adjacent to the retrieval it reorders. A separate node would have to re-derive
    the budget state to make the same call.
    """

    node_id = "retrieve"
    reads = frozenset({"analysis"})
    writes = frozenset({"plan", "pool", "evidence"})
    timeout_ms = 2_000
    fallback_node = None

    def __init__(
        self,
        planner: RetrievalPlanner,
        orchestrator: RetrievalOrchestrator,
        grouper: EvidenceGrouper,
        *,
        reranker: Reranker | None = None,
        screen: EvidenceScreen | None = None,
        wall_ms: int = 260,
        rerank_output_k: int = 8,
    ) -> None:
        # Every one of these arrives through a protocol. A node that imported the retrieval or
        # evidence packages could not be tested without them, and neither subsystem could move
        # out of process — which is the property the whole modular monolith rests on.
        self._planner = planner
        self._orchestrator = orchestrator
        self._grouper = grouper
        self._reranker = reranker
        self._screen = screen
        self._wall_ms = wall_ms
        self._rerank_output_k = rerank_output_k

    async def run(self, state: RequestState) -> NodeResult:
        assert state.analysis is not None  # the interpreter checked `reads` before calling
        variants = state.variants or QueryVariants(raw=state.analysis.normalized_query)

        plan = self._planner.plan(
            state.analysis,
            variants,
            wall_ms=min(self._wall_ms, max(1, state.budget.wall_ms_remaining)),
        )
        pool = await self._orchestrator.execute(plan, state.principal, state.budget)
        candidates = await self._maybe_rerank(state, plan, pool.candidates)
        groups = tuple(self._grouper.group(candidates))

        next_state = state.advanced(plan=plan, pool=pool, evidence=groups)
        if self._screen is not None:
            # After grouping, so a drop removes the whole group: a near-duplicate of an
            # injected or foreign chunk is no safer than the chunk. An isolation violation
            # raises straight through; the engine re-raises it rather than falling back.
            screened = await self._screen.screen(state.principal, groups)
            next_state = next_state.advanced(evidence=screened.kept)
            for verdict in screened.verdicts:
                next_state = next_state.with_verdict(verdict)

        return NodeResult(node_id=self.node_id, status=NodeStatus.OK, state=next_state)

    async def _maybe_rerank(self, state, plan, candidates):
        """Rerank if the ladder allows it, and fall back to the fused order if not.

        The gate lives here rather than in the evidence package because it is an orchestration
        decision: it reads the degradation level, which is request state. Reranking is the
        ladder's cheapest rung, and skipping it is normal operation rather than a failure — the
        fused ordering it falls back to was already a valid ordering.
        """
        from prag.core.budget import DegradationLevel, DegradationPlan

        ladder = DegradationPlan(level=DegradationLevel(state.budget.degradation_level))
        if self._reranker is None or not ladder.rerank_allowed or not candidates:
            return tuple(candidates[: self._rerank_output_k])

        deadline = state.budget.deadline_for(f"{self.node_id}.rerank", 0.2)
        if deadline.expired:
            return tuple(candidates[: self._rerank_output_k])

        try:
            reranked = await self._reranker.rerank(
                state.analysis.normalized_query,
                list(candidates[: plan.rerank.input_k]),
                self._rerank_output_k,
                deadline,
            )
        except (DeadlineExceeded, Exception):
            # A broken or slow reranker degrades the ordering; it must not fail the request.
            return tuple(candidates[: self._rerank_output_k])

        return tuple(reranked)


class BuildContextNode:
    """Assembles the context bundle, or routes to abstention when there is nothing to assemble."""

    node_id = "build_context"
    reads = frozenset({"analysis"})
    writes = frozenset({"bundle", "spec", "decision"})
    timeout_ms = 500
    fallback_node = None

    def __init__(
        self,
        builder: ContextBuilder,
        router: ModelRouter,
        *,
        fusion: FusionPolicy | None = None,
    ) -> None:
        self._builder = builder
        self._router = router
        # Optional so the node still works in graphs assembled before fusion existed; the
        # platform always injects one.
        self._fusion = fusion

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

        if self._fusion is not None:
            # The policy table decides over what actually made it into context. Its two
            # abstention rules are gates: no score overrides them.
            decision = await self._fusion.decide(
                state.analysis, state.parametric, bundle, None, state.policy
            )
            if decision.abstention is not None:
                raise AbstentionRequired(
                    decision.abstention.explanation,
                    abstention_code=str(decision.abstention.reason_code),
                    suggested_action=decision.abstention.suggested_action,
                )
            if decision.basis is KnowledgeBasis.PARAMETRIC:
                # Retrieval could not support an answer and this is the grounded path: the
                # parametric route, not this one, is where the weights may answer.
                raise AbstentionRequired(
                    "retrieved evidence is too weak to ground an answer",
                    abstention_code=str(AbstentionCode.KNOWLEDGE_BELOW_FLOOR),
                    suggested_action="rephrase the question, or check that the source is indexed",
                )
            if decision.conflicts:
                # Conflicting evidence is the case for the reasoning profile, if one exists.
                spec = self._router.select(
                    state.analysis, decision, AdapterSet(), state.budget, state.policy
                )

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

    decision = state.decision
    conflicts = decision.conflicts if decision is not None else ()
    staleness = decision.staleness if decision is not None else None

    return AnswerEnvelope(
        request_id=state.request_id,
        answer=generated.text + _structured_notes(conflicts, staleness),
        citations=citations,
        confidence=ConfidenceBlock(
            score=groundedness, band=band, basis=KnowledgeBasis.RETRIEVED_EVIDENCE
        ),
        grounding=grounding,
        conflicts=conflicts,
        staleness_warning=staleness,
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


def _structured_notes(conflicts, staleness) -> str:
    """Prose generated from the structure, so the words and the numbers cannot disagree.

    Only for what the reader must see: conflicts between sources, and stale evidence. A
    parametric claim that lost to the evidence is logged, not narrated — the answer already
    states what the evidence says.
    """
    from prag.core.models.fusion import ConflictKind, ConflictResolution

    notes = []
    for conflict in conflicts:
        if conflict.kind is not ConflictKind.SOURCE_VS_SOURCE or len(conflict.positions) < 2:
            continue
        first, second = conflict.positions[0], conflict.positions[1]
        if conflict.resolution is ConflictResolution.SURFACED:
            notes.append(
                f'Sources disagree here: {first.origin_id} says "{first.excerpt}", while '
                f'{second.origin_id} says "{second.excerpt}".'
            )
        else:
            notes.append(
                f"A lower-authority source ({second.origin_id}) disagrees: "
                f'"{second.excerpt}".'
            )
    if staleness is not None:
        notes.append(
            f"Note: the supporting evidence is {staleness.staleness_ratio:.1f} times older than "
            f"this kind of information typically stays accurate "
            f"({staleness.half_life_days:.0f}-day half-life)."
        )
    return "".join(f"\n\n{note}" for note in notes)


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
