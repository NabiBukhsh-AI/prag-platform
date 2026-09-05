"""Claim extraction, entailment, citation binding, streaming, and model routing."""

from __future__ import annotations

import pytest

from prag.core.errors import ConfigurationError
from prag.core.models.context import ContextBundle, ContextRegion, RegionName
from prag.core.models.fusion import KnowledgeBasis, KnowledgeDecision
from prag.core.models.generation import GenerationChunk
from prag.core.models.identity import Budget
from prag.core.models.parametric import AdapterSet
from prag.generation import (
    DEFAULT_PROFILES,
    FallbackChain,
    GenerationProfile,
    HeuristicGroundingVerifier,
    ModelOption,
    PolicyModelRouter,
    SentenceBuffer,
    buffered_stream,
    cited_markers,
    entailment_score,
    extract_claims,
    profile_for,
)
from tests.unit.test_context import a_group
from tests.unit.test_graph_engine import an_analysis


def a_bundle(*groups) -> ContextBundle:
    return ContextBundle(
        bundle_id="ctx-1",
        regions=(
            ContextRegion(
                name=RegionName.EVIDENCE, allocated_tokens=1000, used_tokens=100, trimmable=True
            ),
        ),
        evidence=tuple(groups),
        rendered_prompt_hash="h" * 32,
    )


class TestClaimExtraction:
    def test_splits_into_sentences(self) -> None:
        claims = extract_claims("The window is fifteen minutes. Records last thirty days.")
        assert len(claims) == 2

    def test_a_trailing_marker_stays_with_its_sentence(self) -> None:
        """The convention is "sentence. [E1]", so splitting strands the marker.

        Left uncorrected, every claim is checked against the evidence for the claim before it,
        and the verifier then strips citations that were correct all along.
        """
        claims = [
            c
            for c in extract_claims(
                "The window is fifteen minutes. [E1] Records last thirty days. [E2]"
            )
            if c.is_substantive
        ]
        assert [c.claimed_markers for c in claims] == [("E1",), ("E2",)]
        assert "fifteen minutes" in claims[0].text
        assert "thirty days" in claims[1].text

    def test_several_markers_on_one_sentence(self) -> None:
        claims = [
            c for c in extract_claims("Both sources agree on this. [E1] [E3]") if c.is_substantive
        ]
        assert claims[0].claimed_markers == ("E1", "E3")

    def test_a_leading_marker_with_no_previous_claim_is_kept(self) -> None:
        claims = [
            c for c in extract_claims("[E1] The window is fifteen minutes.") if c.is_substantive
        ]
        assert claims[0].claimed_markers == ("E1",)

    def test_questions_are_not_claims(self) -> None:
        """A question asserts nothing, so scoring it drags faithfulness down for no reason."""
        claims = extract_claims("Is the window fifteen minutes long?")
        assert not claims[0].is_substantive

    def test_refusals_are_not_claims(self) -> None:
        claims = extract_claims("I do not have evidence in context to answer that.")
        assert not claims[0].is_substantive

    def test_markers_are_deduplicated_in_order(self) -> None:
        assert cited_markers("a [E2] b [E1] c [E2]") == ("E2", "E1")


class TestEntailment:
    def test_supported_claim_scores_high(self) -> None:
        assert entailment_score(
            "the window is fifteen minutes",
            "the escalation window is fifteen minutes for a sev-1 incident",
        ) == pytest.approx(1.0)

    def test_unsupported_claim_scores_zero(self) -> None:
        assert entailment_score("records last ninety days", "the window is fifteen minutes") == 0.0

    def test_it_is_directional(self) -> None:
        """It measures how much of the *claim* the evidence accounts for.

        A symmetric measure would penalise a short precise quote for being short, and reward a
        long passage for happening to contain a few of the claim's words.
        """
        claim = "records are retained"
        short = entailment_score(claim, "records are retained")
        padded = entailment_score(claim, "records are retained " + "unrelated filler text " * 20)
        assert short == pytest.approx(padded)

    def test_empty_inputs_score_zero(self) -> None:
        assert entailment_score("", "anything") == 0.0
        assert entailment_score("anything", "") == 0.0


class TestCitationBinding:
    async def test_a_supported_claim_gets_its_citation(self) -> None:
        verifier = HeuristicGroundingVerifier(entailment_threshold=0.5)
        bundle = a_bundle(
            a_group("the escalation window is fifteen minutes", group_id="g1", marker="E1")
        )
        report = await verifier.verify("The escalation window is fifteen minutes. [E1]", bundle)

        assert report.claims_cited == 1
        assert report.claims_unsourced == 0
        assert report.groundedness == pytest.approx(1.0)

    async def test_an_unsupported_claim_is_marked_unsourced(self) -> None:
        """No citation rather than a plausible one.

        A citation on an unsupported claim converts an unverified statement into an apparently
        verified one, which is worse than leaving it uncited.
        """
        verifier = HeuristicGroundingVerifier(entailment_threshold=0.6)
        bundle = a_bundle(a_group("the escalation window is fifteen minutes", marker="E1"))
        report = await verifier.verify(
            "Customer records are deleted after ninety days entirely. [E1]", bundle
        )

        assert report.claims_cited == 0
        assert report.claims_unsourced == 1

    async def test_a_marker_absent_from_this_request_is_discarded(self) -> None:
        """A hallucinated citation is a hallucination even when the claim is true.

        Scoring it would give a fabricated reference a chance to survive.
        """
        verifier = HeuristicGroundingVerifier(
            entailment_threshold=0.5, allow_unclaimed_support=False
        )
        bundle = a_bundle(a_group("the escalation window is fifteen minutes", marker="E1"))
        report = await verifier.verify("The escalation window is fifteen minutes. [E9]", bundle)

        assert report.claims_cited == 0, "E9 is not in this request's context"

    async def test_support_may_come_from_uncited_evidence(self) -> None:
        """Failing to name the marker is a formatting miss, not a grounding failure."""
        verifier = HeuristicGroundingVerifier(entailment_threshold=0.5)
        bundle = a_bundle(a_group("the escalation window is fifteen minutes", marker="E1"))
        report = await verifier.verify("The escalation window is fifteen minutes.", bundle)

        assert report.claims_cited == 1

    async def test_verdicts_carry_the_score_that_justified_them(self) -> None:
        verifier = HeuristicGroundingVerifier(entailment_threshold=0.5)
        bundle = a_bundle(a_group("the escalation window is fifteen minutes", marker="E1"))
        report = await verifier.verify("The escalation window is fifteen minutes. [E1]", bundle)

        assert report.verdicts[0].entailed
        assert report.verdicts[0].entailment_score >= 0.5

    async def test_an_answer_with_no_claims_is_fully_grounded(self) -> None:
        """A refusal has nothing to be unfaithful about."""
        verifier = HeuristicGroundingVerifier()
        report = await verifier.verify("I cannot answer that.", a_bundle())
        assert report.groundedness == 1.0


class TestSentenceBuffer:
    def test_holds_until_a_boundary(self) -> None:
        """Nothing reaches the client before the guardrails have seen it."""
        buffer = SentenceBuffer()
        assert buffer.feed("The window is") == []
        assert buffer.feed(" fifteen minutes. ") == ["The window is fifteen minutes. "]

    def test_releases_several_at_once(self) -> None:
        buffer = SentenceBuffer()
        assert len(buffer.feed("One. Two. Three. ")) == 3

    def test_a_long_unpunctuated_run_still_releases(self) -> None:
        """A stalled stream is worse for the reader than a coarse boundary."""
        buffer = SentenceBuffer(max_chars=40)
        released = buffer.feed("word " * 20)
        assert released, "a long run must not stall the stream indefinitely"

    def test_it_releases_on_a_word_boundary(self) -> None:
        """A split token is visible to the reader in a way an early sentence break is not."""
        buffer = SentenceBuffer(max_chars=30)
        released = buffer.feed("alpha beta gamma delta epsilon zeta eta")
        assert all(not chunk.endswith(("alph", "bet", "gamm")) for chunk in released)

    def test_flush_returns_the_tail(self) -> None:
        buffer = SentenceBuffer()
        buffer.feed("no terminal punctuation")
        assert buffer.flush() == "no terminal punctuation"
        assert buffer.pending == ""


class TestBufferedStream:
    async def _chunks(self, *texts: str):
        for index, text in enumerate(texts):
            yield GenerationChunk(text=text, index=index, ttft_ms=7 if index == 0 else None)
        yield GenerationChunk(text="", index=len(texts), finish_reason="stop")

    async def test_text_is_released_and_terminated(self) -> None:
        events = [e async for e in buffered_stream(self._chunks("One. ", "Two. "))]
        text = "".join(e.text for e in events)

        assert "One." in text
        assert "Two." in text
        assert events[-1].finish_reason == "stop"

    async def test_ttft_lands_on_the_first_released_event(self) -> None:
        """Reporting the provider's first token would flatter it by the buffering delay."""
        events = [e async for e in buffered_stream(self._chunks("One. ", "Two. "))]
        with_ttft = [e for e in events if e.ttft_ms is not None]

        assert len(with_ttft) == 1
        assert with_ttft[0].text

    async def test_a_guard_can_withhold_a_sentence(self) -> None:
        async def block_everything(_: str) -> str | None:
            return None

        events = [
            e async for e in buffered_stream(self._chunks("Secret. "), guard=block_everything)
        ]
        assert "".join(e.text for e in events) == ""

    async def test_a_guard_can_rewrite_a_sentence(self) -> None:
        """Redaction is the useful response to detected PII, not refusal."""

        async def redact(sentence: str) -> str:
            return sentence.replace("secret", "[redacted]")

        events = [e async for e in buffered_stream(self._chunks("A secret. "), guard=redact)]
        assert "[redacted]" in "".join(e.text for e in events)


class TestProfiles:
    def test_grounded_extraction_is_deterministic(self) -> None:
        """Sampling on an extraction task buys nothing and costs faithfulness."""
        profile = profile_for("grounded_extraction")
        assert profile.temperature == 0.0
        assert profile.is_deterministic

    def test_the_creative_profile_is_the_only_sampled_one(self) -> None:
        sampled = [p.name for p in DEFAULT_PROFILES.values() if not p.is_deterministic]
        assert "grounded_extraction" not in sampled
        assert "creative" in sampled

    def test_an_unknown_profile_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown generation profile"):
            profile_for("nonexistent")

    @pytest.mark.parametrize("temperature", [-0.1, 2.5])
    def test_temperature_is_bounded(self, temperature: float) -> None:
        with pytest.raises(ValueError, match="temperature"):
            GenerationProfile(name="x", temperature=temperature)


class TestModelRouting:
    def _router(self, *, reasoning: str | None = None) -> PolicyModelRouter:
        options = {
            "mid.instruct": ModelOption(
                model_id="mid.instruct",
                model_version="1",
                provider_id="p",
                context_window=8_192,
                cost_per_1k_in=0.001,
                cost_per_1k_out=0.003,
            ),
            "large.reasoning": ModelOption(
                model_id="large.reasoning",
                model_version="1",
                provider_id="p",
                context_window=128_000,
                cost_per_1k_in=0.01,
                cost_per_1k_out=0.03,
                tier=2,
            ),
        }
        return PolicyModelRouter(options, default_model="mid.instruct", reasoning_model=reasoning)

    def _decision(self, **overrides) -> KnowledgeDecision:
        base = {
            "basis": KnowledgeBasis.RETRIEVED_EVIDENCE,
            "knowledge_score": 0.8,
            "p_parametric": 0.0,
            "p_retrieval": 0.8,
            "agreement_independent": 1.0,
            "authority_max": 0.9,
        }
        return KnowledgeDecision(**{**base, **overrides})

    def _budget(self, degradation: int = 0) -> Budget:
        return Budget(
            wall_ms_total=5_000,
            wall_ms_remaining=5_000,
            usd_total=0.1,
            max_tokens_in=8_000,
            max_tokens_out=1_024,
            degradation_level=degradation,
        )

    def test_simple_factual_uses_the_cheap_model(self, policy) -> None:
        """Extraction from provided evidence does not need a frontier model."""
        spec = self._router(reasoning="large.reasoning").select(
            an_analysis(), self._decision(), AdapterSet(), self._budget(), policy
        )
        assert spec.model_id == "mid.instruct"
        assert spec.profile == "grounded_extraction"

    def test_degradation_blocks_escalation(self, policy) -> None:
        """A degraded request must not be able to spend its way out of degradation."""
        analysis = an_analysis().model_copy(
            update={"complexity": an_analysis().complexity.model_copy(update={"value": "complex"})}
        )
        router = self._router(reasoning="large.reasoning")

        undegraded = router.select(
            analysis, self._decision(), AdapterSet(), self._budget(0), policy
        )
        degraded = router.select(analysis, self._decision(), AdapterSet(), self._budget(4), policy)

        assert undegraded.model_id == "large.reasoning"
        assert degraded.model_id == "mid.instruct"

    def test_an_unregistered_default_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="default model"):
            PolicyModelRouter({}, default_model="missing")


class TestFallbackChain:
    def test_advances_through_the_chain(self) -> None:
        chain = FallbackChain(profile="grounded_extraction", providers=("a", "b", "c"))
        assert chain.next_after("a") == "b"
        assert chain.next_after("b") == "c"

    def test_the_end_of_the_chain_is_none(self) -> None:
        chain = FallbackChain(profile="p", providers=("a", "b"))
        assert chain.next_after("b") is None

    def test_an_unknown_provider_starts_at_the_beginning(self) -> None:
        chain = FallbackChain(profile="p", providers=("a", "b"))
        assert chain.next_after("unknown") == "a"
