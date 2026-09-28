"""Planning, parallel execution, rank fusion, lexical retrieval, reranking, and cache keys."""

from __future__ import annotations

import asyncio
import time

import pytest

from prag.caching import (
    analysis_key,
    embedding_key,
    exact_answer_key,
    is_cacheable,
    retrieval_key,
    ttl_for_volatility,
)
from prag.core.errors import RetrievalTotalFailure, SourceUnavailable
from prag.core.models.common import Deadline, SourceCapabilities
from prag.core.models.identity import Budget, Principal, TenantPolicy, UtilityWeights
from prag.core.models.query import QueryVariants, SubQuery
from prag.core.models.retrieval import (
    FusionConfig,
    FusionMethod,
    LegResult,
    LegStatus,
    PartialResultsPolicy,
    RetrievalLeg,
)
from prag.evidence import LexicalOverlapReranker, NoOpReranker, rerank_with_budget
from prag.retrieval import (
    CircuitBreaker,
    InMemoryLexicalIndex,
    LexicalKnowledgeSource,
    ParallelRetrievalOrchestrator,
    SourceSpec,
    build_plan,
    reciprocal_rank_fusion,
)
from tests.fakes.sources import make_candidate
from tests.unit.test_graph_engine import an_analysis

PRINCIPAL = Principal(tenant_id="tenant-a", user_id="u1", acl_hashes=("acl-eng",))
POLICY = TenantPolicy(
    tenant_id="tenant-a",
    config_version="c1",
    utility_weights=UtilityWeights(quality=0.6, latency=0.2, cost=0.2),
)


def a_budget(wall_ms: int = 10_000) -> Budget:
    return Budget(
        wall_ms_total=wall_ms,
        wall_ms_remaining=wall_ms,
        usd_total=0.05,
        max_tokens_in=8_000,
        max_tokens_out=1_024,
    )


def dense_caps(max_top_k: int = 100) -> SourceCapabilities:
    return SourceCapabilities(
        supports_filters=True, supports_vectors=True, supports_text=False, max_top_k=max_top_k
    )


def lexical_caps(max_top_k: int = 100) -> SourceCapabilities:
    return SourceCapabilities(
        supports_filters=True, supports_vectors=False, supports_text=True, max_top_k=max_top_k
    )


class FakeSource:
    """A source with controllable latency, failure, and results."""

    def __init__(
        self,
        source_id: str,
        *,
        candidates: tuple = (),
        delay_s: float = 0.0,
        fail: bool = False,
        capabilities: SourceCapabilities | None = None,
    ) -> None:
        self.source_id = source_id
        self.capabilities = capabilities or dense_caps()
        self._candidates = candidates
        self._delay_s = delay_s
        self._fail = fail
        self.calls = 0

    async def retrieve(self, leg, principal, deadline):
        self.calls += 1
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if self._fail:
            raise SourceUnavailable("configured to fail", source_id=self.source_id)
        return LegResult(
            leg_id=leg.leg_id,
            source_id=self.source_id,
            status=LegStatus.OK,
            candidates=self._candidates,
            latency_ms=1,
        )

    async def health(self):
        from prag.core.models.common import HealthState, HealthStatus

        return HealthStatus(state=HealthState.HEALTHY, checked_at_ms=0)


class TestPlanner:
    def test_a_lexical_source_gets_the_expanded_variant(self) -> None:
        """BM25 matches terms, so an alias is the difference between finding and missing."""
        plan = build_plan(
            an_analysis(),
            QueryVariants(raw="sev-1 target", expanded="sev-1 target sev1 severity 1"),
            (SourceSpec("lexical.primary", lexical_caps()),),
        )
        assert plan.legs[0].query_variant == "expanded"
        assert "sev1" in plan.legs[0].query_text

    def test_a_dense_source_gets_the_rewritten_variant(self) -> None:
        """It matches meaning, so the aliases are noise."""
        plan = build_plan(
            an_analysis(),
            QueryVariants(raw="please tell me the target", rewritten="the target"),
            (SourceSpec("vector.primary", dense_caps()),),
        )
        assert plan.legs[0].query_variant == "rewritten"
        assert plan.legs[0].query_text == "the target"

    def test_required_flags_survive_into_the_plan(self) -> None:
        """A failed optional leg is a warning; a failed required leg is a plan failure."""
        plan = build_plan(
            an_analysis(),
            QueryVariants(raw="q"),
            (
                SourceSpec("vector.primary", dense_caps(), required=True),
                SourceSpec("lexical.primary", lexical_caps(), required=False),
            ),
        )
        assert plan.required_leg_ids == {"leg.vector.primary"}

    def test_top_k_is_capped_by_source_capability(self) -> None:
        plan = build_plan(
            an_analysis(),
            QueryVariants(raw="q"),
            (SourceSpec("vector.primary", dense_caps(max_top_k=5)),),
            top_k=50,
        )
        assert plan.legs[0].top_k == 5

    def test_sub_queries_add_parallel_legs(self) -> None:
        plan = build_plan(
            an_analysis(),
            QueryVariants(
                raw="compare a and b",
                sub_queries=(
                    SubQuery(sub_query_id="sq1", text="a"),
                    SubQuery(sub_query_id="sq2", text="b"),
                ),
            ),
            (SourceSpec("vector.primary", dense_caps()),),
        )
        assert len(plan.legs) == 3

    def test_a_sub_query_leg_is_never_required(self) -> None:
        """A wrong decomposition should cost one hop's recall, not the whole request."""
        plan = build_plan(
            an_analysis(),
            QueryVariants(raw="q", sub_queries=(SubQuery(sub_query_id="sq1", text="a"),)),
            (SourceSpec("vector.primary", dense_caps(), required=True),),
        )
        sub_legs = [leg for leg in plan.legs if leg.sub_query_id]
        assert sub_legs
        assert all(not leg.required for leg in sub_legs)

    def test_every_leg_gets_the_full_wall_budget(self) -> None:
        """Legs run concurrently, so splitting the budget would make more sources less patient."""
        plan = build_plan(
            an_analysis(),
            QueryVariants(raw="q"),
            (
                SourceSpec("vector.primary", dense_caps()),
                SourceSpec("lexical.primary", lexical_caps()),
            ),
            wall_ms=260,
        )
        assert all(leg.timeout_ms == 260 for leg in plan.legs)


class TestRankFusion:
    def test_agreement_across_sources_outranks_a_single_top_hit(self) -> None:
        """Independent agreement is evidence, and rank fusion is the cheapest way to spend it."""
        shared = make_candidate("shared", candidate_id="shared")
        dense_only = make_candidate("dense only", candidate_id="dense-only")
        lexical_only = make_candidate("lexical only", candidate_id="lex-only")

        fused = reciprocal_rank_fusion([[dense_only, shared], [lexical_only, shared]])
        assert fused[0].chunk_id == shared.chunk_id

    def test_it_needs_no_score_calibration(self) -> None:
        """A BM25 score and a cosine similarity do not share a scale.

        RRF compares positions, which every ranker produces on the same scale by construction.
        """
        huge = make_candidate("bm25 style", candidate_id="huge", score=97.4)
        small = make_candidate("cosine style", candidate_id="small", score=0.61)

        fused = reciprocal_rank_fusion([[small], [huge]])
        assert {c.candidate_id for c in fused} == {"small", "huge"}
        assert fused[0].fused_score == pytest.approx(fused[1].fused_score)

    def test_it_is_deterministic(self) -> None:
        lists = [
            [make_candidate("a", candidate_id="a"), make_candidate("b", candidate_id="b")],
            [make_candidate("b", candidate_id="b"), make_candidate("c", candidate_id="c")],
        ]
        assert [c.candidate_id for c in reciprocal_rank_fusion(lists)] == [
            c.candidate_id for c in reciprocal_rank_fusion(lists)
        ]

    def test_per_leg_ranks_are_preserved(self) -> None:
        """The trace has to say which source found a candidate, and where."""
        left = make_candidate("x", candidate_id="x").model_copy(
            update={"rank_by_leg": {"leg.dense": 0}}
        )
        right = make_candidate("x", candidate_id="x").model_copy(
            update={"rank_by_leg": {"leg.lexical": 3}}
        )
        fused = reciprocal_rank_fusion([[left], [right]])
        assert set(fused[0].rank_by_leg) >= {"leg.dense"}

    def test_a_failed_leg_does_not_contribute(self) -> None:
        """An unreachable leg must not look like one that searched and found nothing."""
        from prag.retrieval import fuse_legs

        good = LegResult(
            leg_id="l1",
            source_id="vector.primary",
            status=LegStatus.OK,
            candidates=(make_candidate("hit", candidate_id="hit"),),
            latency_ms=1,
        )
        failed = LegResult(
            leg_id="l2", source_id="lexical.primary", status=LegStatus.FAILED, latency_ms=0
        )

        fused = fuse_legs((good, failed), FusionConfig(method=FusionMethod.RRF))
        assert len(fused) == 1


class TestParallelExecution:
    def _plan(self, *specs: SourceSpec, wall_ms: int = 500, policy=None):
        plan = build_plan(an_analysis(), QueryVariants(raw="q"), specs, wall_ms=wall_ms)
        return (
            plan if policy is None else plan.model_copy(update={"partial_results_policy": policy})
        )

    async def test_legs_run_concurrently(self) -> None:
        """Sequential awaits would make the stage cost the sum of the legs."""
        slow_a = FakeSource("a", delay_s=0.10, candidates=(make_candidate("a", candidate_id="a"),))
        slow_b = FakeSource("b", delay_s=0.10, candidates=(make_candidate("b", candidate_id="b"),))

        plan = self._plan(
            SourceSpec("a", dense_caps(), required=True), SourceSpec("b", dense_caps())
        )
        orchestrator = ParallelRetrievalOrchestrator({"a": slow_a, "b": slow_b})

        started = time.monotonic()
        pool = await orchestrator.execute(plan, PRINCIPAL, a_budget())
        elapsed = time.monotonic() - started

        assert len(pool.candidates) == 2
        assert elapsed < 0.19, "two 100 ms legs run in parallel, not in series"

    async def test_a_slow_leg_does_not_delay_a_fast_one_past_the_budget(self) -> None:
        """An answer from two of three sources beats waiting out the third."""
        fast = FakeSource("fast", candidates=(make_candidate("fast", candidate_id="f"),))
        stuck = FakeSource("stuck", delay_s=5.0)

        plan = self._plan(
            SourceSpec("fast", dense_caps(), required=True),
            SourceSpec("stuck", dense_caps()),
            wall_ms=120,
        )
        pool = await ParallelRetrievalOrchestrator({"fast": fast, "stuck": stuck}).execute(
            plan, PRINCIPAL, a_budget()
        )

        assert pool.degraded
        assert any(r.status is LegStatus.TIMED_OUT for r in pool.leg_results)
        assert pool.candidates, "the fast leg's results are still used"

    async def test_an_optional_leg_failing_is_a_coverage_warning(self) -> None:
        good = FakeSource("good", candidates=(make_candidate("g", candidate_id="g"),))
        bad = FakeSource("bad", fail=True)

        plan = self._plan(
            SourceSpec("good", dense_caps(), required=True), SourceSpec("bad", dense_caps())
        )
        pool = await ParallelRetrievalOrchestrator({"good": good, "bad": bad}).execute(
            plan, PRINCIPAL, a_budget()
        )

        assert pool.degraded
        assert pool.candidates

    async def test_every_required_leg_failing_is_a_total_failure(self) -> None:
        """The caller then decides: answer with explicit marking, or abstain."""
        plan = self._plan(SourceSpec("bad", dense_caps(), required=True))
        orchestrator = ParallelRetrievalOrchestrator({"bad": FakeSource("bad", fail=True)})

        with pytest.raises(RetrievalTotalFailure):
            await orchestrator.execute(plan, PRINCIPAL, a_budget())

    async def test_all_or_nothing_refuses_a_partial_result(self) -> None:
        good = FakeSource("good", candidates=(make_candidate("g", candidate_id="g"),))
        plan = self._plan(
            SourceSpec("good", dense_caps(), required=True),
            SourceSpec("bad", dense_caps()),
            policy=PartialResultsPolicy.ALL_OR_NOTHING,
        )
        orchestrator = ParallelRetrievalOrchestrator(
            {"good": good, "bad": FakeSource("bad", fail=True)}
        )
        with pytest.raises(RetrievalTotalFailure):
            await orchestrator.execute(plan, PRINCIPAL, a_budget())

    async def test_the_request_budget_caps_the_plan_budget(self) -> None:
        """A plan asking for 260 ms cannot have it when 80 ms remain."""
        stuck = FakeSource("stuck", delay_s=2.0)
        plan = self._plan(SourceSpec("stuck", dense_caps()), wall_ms=5_000)

        started = time.monotonic()
        pool = await ParallelRetrievalOrchestrator({"stuck": stuck}).execute(
            plan, PRINCIPAL, a_budget(wall_ms=100)
        )
        assert time.monotonic() - started < 1.0
        assert pool.leg_results[0].status is LegStatus.TIMED_OUT

    async def test_an_unregistered_source_fails_its_leg_not_the_process(self) -> None:
        plan = self._plan(SourceSpec("ghost", dense_caps()))
        pool = await ParallelRetrievalOrchestrator({}).execute(plan, PRINCIPAL, a_budget())
        assert pool.leg_results[0].status is LegStatus.FAILED


class TestCircuitBreaker:
    def test_it_opens_after_the_threshold(self) -> None:
        """Without one, an unhealthy source turns every request into a timeout."""
        breaker = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            breaker.record_failure("flaky")
        assert breaker.is_open("flaky")
        assert breaker.state("flaky") == "open"

    def test_success_resets_it(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2)
        breaker.record_failure("s")
        breaker.record_success("s")
        breaker.record_failure("s")
        assert not breaker.is_open("s")

    def test_it_half_opens_after_the_cooldown(self) -> None:
        """Refusing traffic forever after a transient outage is its own outage."""
        breaker = CircuitBreaker(failure_threshold=1, cooldown_s=0.0)
        breaker.record_failure("s")
        assert not breaker.is_open("s"), "the cooldown has elapsed, so one probe goes through"

    async def test_an_open_breaker_skips_the_leg_without_attempting_it(self) -> None:
        """Skipped is distinct from failed: nothing was tried, so health is not implicated."""
        source = FakeSource("flaky", fail=True)
        breaker = CircuitBreaker(failure_threshold=1, cooldown_s=60.0)
        breaker.record_failure("flaky")

        plan = build_plan(
            an_analysis(), QueryVariants(raw="q"), (SourceSpec("flaky", dense_caps()),)
        )
        pool = await ParallelRetrievalOrchestrator({"flaky": source}, breaker=breaker).execute(
            plan, PRINCIPAL, a_budget()
        )

        assert pool.leg_results[0].status is LegStatus.SKIPPED_BREAKER_OPEN
        assert source.calls == 0, "an open breaker must not call the source"


class TestLexicalRetrieval:
    def _index(self) -> InMemoryLexicalIndex:
        index = InMemoryLexicalIndex()
        index.index(
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "text": "sev-1 incidents page the on-call lead within 15 minutes",
                    "heading_path": ["Incident Response", "Escalation"],
                    "tenant_id": "tenant-a",
                    "acl_hash": "public",
                    "authority": 0.9,
                },
                {
                    "chunk_id": "c2",
                    "document_id": "d1",
                    "text": "records are retained for 30 days then archived to cold storage",
                    "heading_path": ["Incident Response", "Retention"],
                    "tenant_id": "tenant-a",
                    "acl_hash": "public",
                    "authority": 0.9,
                },
                {
                    "chunk_id": "c3",
                    "document_id": "d2",
                    "text": "finance quarterly figures are confidential",
                    "heading_path": [],
                    "tenant_id": "tenant-a",
                    "acl_hash": "acl-finance",
                    "authority": 0.5,
                },
            ]
        )
        return index

    def test_it_matches_an_exact_identifier(self) -> None:
        """The case dense retrieval reliably fails: it returns a semantic neighbour instead."""
        hits = self._index().search("sev-1", 5, None)
        assert hits[0][0] == "c1"

    def test_hyphenated_identifiers_survive_tokenization(self) -> None:
        """Splitting them would destroy exactly what this source exists to match."""
        index = self._index()
        assert index.search("sev-1", 5, None)

    def test_the_heading_path_is_indexed_with_the_body(self) -> None:
        """Two sections can both say "30 days"; only the heading tells them apart."""
        assert self._index().search("retention", 5, None)[0][0] == "c2"

    def test_filters_apply_before_scoring(self) -> None:
        hits = self._index().search("confidential figures", 5, {"acl_hash": {"$in": ["public"]}})
        assert all(doc_id != "c3" for doc_id, _, _ in hits)

    def test_reindexing_replaces_rather_than_duplicates(self) -> None:
        index = self._index()
        before = index.size
        index.index([{"chunk_id": "c1", "text": "replaced text", "tenant_id": "tenant-a"}])
        assert index.size == before

    async def test_the_source_enforces_acls(self) -> None:
        source = LexicalKnowledgeSource(index=self._index())
        leg = RetrievalLeg(
            leg_id="l1",
            source_id="lexical.primary",
            query_variant="raw",
            query_text="confidential figures",
            top_k=5,
            timeout_ms=200,
        )
        result = await source.retrieve(leg, PRINCIPAL, Deadline.in_ms(500))
        assert all(c.metadata.acl_hash != "acl-finance" for c in result.candidates)

    async def test_an_empty_index_is_degraded(self) -> None:
        health = await LexicalKnowledgeSource(index=InMemoryLexicalIndex()).health()
        assert health.state.value == "degraded"

    async def test_it_declares_text_not_vectors(self) -> None:
        """Honest capabilities are what let the planner send each source what it needs."""
        source = LexicalKnowledgeSource()
        assert source.capabilities.supports_text
        assert not source.capabilities.supports_vectors


class TestReranking:
    def _candidates(self) -> list:
        return [
            make_candidate("unrelated content about storage", candidate_id="a", score=0.9),
            make_candidate("the escalation window is fifteen minutes", candidate_id="b", score=0.5),
            make_candidate("more unrelated filler text here", candidate_id="c", score=0.4),
        ]

    async def test_it_promotes_the_matching_candidate(self) -> None:
        reranked = await LexicalOverlapReranker(position_weight=0.1).rerank(
            "escalation window", self._candidates(), 3, Deadline.in_ms(500)
        )
        assert reranked[0].candidate_id == "b"

    async def test_scores_are_attached(self) -> None:
        reranked = await LexicalOverlapReranker().rerank(
            "escalation", self._candidates(), 3, Deadline.in_ms(500)
        )
        assert all(c.rerank_score is not None for c in reranked)

    async def test_the_no_op_tier_preserves_order(self) -> None:
        """ "No reranking" is a configured choice, not a missing dependency."""
        candidates = self._candidates()
        reranked = await NoOpReranker().rerank("q", candidates, 2, Deadline.in_ms(500))
        assert [c.candidate_id for c in reranked] == ["a", "b"]

    async def test_a_degraded_budget_skips_it_and_says_so(self) -> None:
        """The ladder's cheapest rung, and the skip must be visible in the trace."""
        outcome = await rerank_with_budget(
            LexicalOverlapReranker(),
            "q",
            self._candidates(),
            input_k=10,
            output_k=2,
            deadline=Deadline.in_ms(500),
            allowed=False,
        )
        assert not outcome.ran
        assert outcome.skipped_reason == "degraded_budget"
        assert len(outcome.candidates) == 2, "the fused ordering still stands"

    async def test_an_expired_deadline_skips_it(self) -> None:
        outcome = await rerank_with_budget(
            LexicalOverlapReranker(),
            "q",
            self._candidates(),
            input_k=10,
            output_k=2,
            deadline=Deadline.in_ms(0),
        )
        assert outcome.skipped_reason == "deadline_exceeded"

    async def test_a_broken_reranker_degrades_rather_than_failing(self) -> None:
        """The fused order was already valid, which is why this stage is skippable."""

        class Broken:
            model_id = "broken"

            async def rerank(self, *args, **kwargs):
                raise RuntimeError("boom")

        outcome = await rerank_with_budget(
            Broken(),
            "q",
            self._candidates(),
            input_k=10,
            output_k=3,
            deadline=Deadline.in_ms(500),
        )
        assert outcome.skipped_reason == "reranker_error"
        assert len(outcome.candidates) == 3

    def test_an_invalid_position_weight_is_refused(self) -> None:
        with pytest.raises(ValueError, match="position_weight"):
            LexicalOverlapReranker(position_weight=1.5)


class TestCacheKeys:
    def test_the_tenant_separates_entries(self) -> None:
        """Without it an entry crosses a permission boundary, and the answer looks fine."""
        mine = exact_answer_key("q", PRINCIPAL, POLICY)
        theirs = exact_answer_key("q", Principal(tenant_id="tenant-b", user_id="u"), POLICY)
        assert mine.render() != theirs.render()

    def test_the_acl_set_separates_entries(self) -> None:
        wider = Principal(tenant_id="tenant-a", user_id="u1", acl_hashes=("acl-eng", "acl-hr"))
        assert (
            exact_answer_key("q", PRINCIPAL, POLICY).render()
            != exact_answer_key("q", wider, POLICY).render()
        )

    def test_acl_order_does_not_matter(self) -> None:
        """Otherwise the cache is correct and nearly useless."""
        one = Principal(tenant_id="t", user_id="u", acl_hashes=("a", "b"))
        two = Principal(tenant_id="t", user_id="u", acl_hashes=("b", "a"))
        assert (
            exact_answer_key("q", one, POLICY).render()
            == exact_answer_key("q", two, POLICY).render()
        )

    def test_the_config_version_invalidates_automatically(self) -> None:
        """A tuning change served from old entries would appear not to work."""
        other = POLICY.model_copy(update={"config_version": "c2"})
        assert (
            exact_answer_key("q", PRINCIPAL, POLICY).render()
            != exact_answer_key("q", PRINCIPAL, other).render()
        )

    def test_strict_mode_separates_entries(self) -> None:
        """Strict mode changes what the system will say, so entries must not be shared."""
        strict = POLICY.model_copy(update={"strict_mode": True})
        assert (
            exact_answer_key("q", PRINCIPAL, POLICY).render()
            != exact_answer_key("q", PRINCIPAL, strict).render()
        )

    def test_top_k_separates_retrieval_entries(self) -> None:
        """A cached top-8 serving a top-24 request silently narrows recall."""
        eight = retrieval_key("q", PRINCIPAL, POLICY, embedding_version="v1", top_k=8)
        twenty = retrieval_key("q", PRINCIPAL, POLICY, embedding_version="v1", top_k=24)
        assert eight.render() != twenty.render()

    def test_the_embedding_version_separates_retrieval_entries(self) -> None:
        """A key outliving a reindex returns results for vectors that no longer exist."""
        v1 = retrieval_key("q", PRINCIPAL, POLICY, embedding_version="v1", top_k=8)
        v2 = retrieval_key("q", PRINCIPAL, POLICY, embedding_version="v2", top_k=8)
        assert v1.render() != v2.render()

    def test_embeddings_are_shared_across_tenants(self) -> None:
        """A pure function of text and model, carrying no permission-bearing content."""
        key = embedding_key("some text", model_id="m", model_version="v1")
        assert key.tenant_id == "*"
        assert key.acl_discriminator == ()

    def test_analysis_keys_are_tenant_agnostic_but_config_scoped(self) -> None:
        key = analysis_key("q", POLICY)
        assert key.tenant_id == "*"
        assert key.config_version == "c1"

    def test_whitespace_does_not_change_a_key(self) -> None:
        assert (
            exact_answer_key("the  query", PRINCIPAL, POLICY).render()
            == exact_answer_key("the query", PRINCIPAL, POLICY).render()
        )

    def test_realtime_content_is_never_cached(self) -> None:
        """Not a short TTL but none: a value valid for seconds is wrong for most of a minute."""
        assert ttl_for_volatility("realtime") == 0

    def test_an_unknown_volatility_gets_the_conservative_ttl(self) -> None:
        """Guessing long on unknown staleness is how a cache serves last week's answer."""
        assert ttl_for_volatility("unheard-of") == ttl_for_volatility("fast")


class TestCacheability:
    def _envelope(self, **overrides):
        from prag.core.models.fusion import ConfidenceBand, ConfidenceBlock, KnowledgeBasis
        from prag.core.models.generation import AnswerEnvelope, Diagnostics, GroundingReport

        base = {
            "request_id": "r1",
            "answer": "an answer",
            "confidence": ConfidenceBlock(
                score=0.9, band=ConfidenceBand.HIGH, basis=KnowledgeBasis.RETRIEVED_EVIDENCE
            ),
            "grounding": GroundingReport(claims_total=2, claims_cited=2, claims_unsourced=0),
            "diagnostics": Diagnostics(
                route_class="standard_answer",
                strategy="NON_PARAMETRIC",
                model_id="m",
                model_version="1",
                total_ms=100,
            ),
        }
        return AnswerEnvelope(**{**base, **overrides})

    def test_a_clean_answer_is_cacheable(self) -> None:
        cacheable, reason = is_cacheable(self._envelope())
        assert cacheable
        assert reason is None

    def test_an_abstention_is_not(self) -> None:
        from prag.core.models.fusion import Abstention, AbstentionCode

        cacheable, reason = is_cacheable(
            self._envelope(
                abstention=Abstention(
                    reason_code=AbstentionCode.KNOWLEDGE_BELOW_FLOOR, explanation="x"
                )
            )
        )
        assert not cacheable
        assert reason == "abstained"

    def test_a_coverage_warning_is_not(self) -> None:
        from prag.core.models.context import CoverageWarning

        cacheable, reason = is_cacheable(
            self._envelope(coverage_warning=CoverageWarning(coverage=0.4))
        )
        assert not cacheable
        assert reason == "coverage_warning"

    def test_an_unsourced_claim_is_not(self) -> None:
        """Caching it republishes it indefinitely, without the report that flagged it."""
        from prag.core.models.generation import GroundingReport

        cacheable, reason = is_cacheable(
            self._envelope(
                grounding=GroundingReport(claims_total=2, claims_cited=1, claims_unsourced=1)
            )
        )
        assert not cacheable
        assert reason == "unsourced_claims"

    def test_the_reason_makes_the_decision_observable(self) -> None:
        """A hit rate falling for retrieval reasons and for key reasons look identical."""
        from prag.core.models.fusion import Abstention, AbstentionCode

        _, reason = is_cacheable(
            self._envelope(
                abstention=Abstention(
                    reason_code=AbstentionCode.RETRIEVAL_TOTAL_FAILURE, explanation="x"
                )
            )
        )
        assert reason
