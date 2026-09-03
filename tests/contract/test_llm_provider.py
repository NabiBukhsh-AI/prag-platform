"""Conformance suite for ``LLMProvider``."""

from __future__ import annotations

from typing import Any

import pytest

from prag.core.errors import ProviderUnavailable
from prag.core.models.common import Deadline
from prag.core.models.context import RegionName, RenderedRegion
from prag.core.models.generation import FinishReason, GenerationRequest, ModelSpec
from prag.core.protocols import LLMProvider

pytestmark = pytest.mark.contract


@pytest.fixture
def spec() -> ModelSpec:
    return ModelSpec(
        model_id="mid.instruct",
        model_version="1",
        provider_id="fake.recorded",
        profile="grounded_extraction",
        context_window=8192,
        cost_per_1k_in=0.001,
        cost_per_1k_out=0.003,
    )


@pytest.fixture
def gen_request(spec: ModelSpec, request_id: str) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        spec=spec,
        regions=(
            RenderedRegion(
                name=RegionName.SYSTEM,
                content="Answer only from the evidence.",
                grants_instruction_authority=True,
            ),
            RenderedRegion(
                name=RegionName.EVIDENCE,
                content="[E1] Sev-1 escalates to the on-call lead within 15 minutes.",
                grants_instruction_authority=False,
            ),
            RenderedRegion(name=RegionName.QUERY, content="What is the sev-1 escalation path?"),
        ),
    )


def test_satisfies_protocol(llm_provider_factory: Any) -> None:
    assert isinstance(llm_provider_factory(), LLMProvider)


def test_declares_support(llm_provider_factory: Any, spec: ModelSpec) -> None:
    provider = llm_provider_factory()
    assert provider.provider_id
    assert isinstance(provider.supports(spec), bool)


async def test_generate_returns_usage_and_identity(
    llm_provider_factory: Any, gen_request: GenerationRequest, deadline: Deadline
) -> None:
    """The result must say which model produced it, and what it cost.

    Cost attribution uses the same numbers the router used to choose, which is the only way
    cost-aware routing can be checked against reality rather than assumed.
    """
    provider = llm_provider_factory()
    result = await provider.generate(gen_request, deadline)

    assert result.text
    assert result.finish_reason is FinishReason.STOP
    assert result.usage.tokens_in > 0
    assert result.model_id == gen_request.spec.model_id
    assert result.provider_id == provider.provider_id


async def test_replays_recorded_response_by_prompt(
    llm_provider_factory: Any, gen_request: GenerationRequest, deadline: Deadline
) -> None:
    """Determinism: the same prompt yields the same text.

    Every test outside the evaluation suite depends on this. A provider that varies its output
    turns an assertion about platform behaviour into an assertion about a model's mood.
    """
    provider = llm_provider_factory()
    first = await provider.generate(gen_request, deadline)
    second = await provider.generate(gen_request, deadline)
    assert first.text == second.text


async def test_first_stream_chunk_carries_ttft(
    llm_provider_factory: Any, gen_request: GenerationRequest, deadline: Deadline
) -> None:
    """Time to first token cannot be reconstructed from a total.

    It is the latency figure that describes what the user actually experiences, so a provider
    that omits it has silently removed the number the whole latency budget is written against.
    """
    provider = llm_provider_factory()
    chunks = [chunk async for chunk in provider.stream(gen_request, deadline)]

    assert chunks, "stream produced nothing"
    assert chunks[0].ttft_ms is not None, "first chunk must carry ttft_ms"
    assert chunks[0].ttft_ms >= 0
    assert all(c.ttft_ms is None for c in chunks[1:]), "only the first chunk carries ttft_ms"


async def test_stream_terminates_with_a_finish_reason(
    llm_provider_factory: Any, gen_request: GenerationRequest, deadline: Deadline
) -> None:
    provider = llm_provider_factory()
    chunks = [chunk async for chunk in provider.stream(gen_request, deadline)]

    assert chunks[-1].finish_reason is not None, "stream must end with a finish reason"
    assert chunks[-1].usage is not None, "final chunk must report usage"


async def test_stream_and_generate_agree(
    llm_provider_factory: Any, gen_request: GenerationRequest, deadline: Deadline
) -> None:
    """Streaming and non-streaming must produce the same answer for the same prompt.

    If they diverge, a bug reproducible only in streaming mode is indistinguishable from a bug
    in the platform's stream handling.
    """
    provider = llm_provider_factory()
    whole = await provider.generate(gen_request, deadline)
    streamed = "".join([c.text async for c in provider.stream(gen_request, deadline)])
    assert streamed == whole.text


async def test_failure_raises_a_typed_error(
    llm_provider_factory: Any, gen_request: GenerationRequest, deadline: Deadline
) -> None:
    provider = llm_provider_factory(fail=True)
    with pytest.raises(ProviderUnavailable) as excinfo:
        await provider.generate(gen_request, deadline)
    assert excinfo.value.reason_code == "provider_unavailable"
    assert excinfo.value.retryable is True, "retry is permitted before the first token"


async def test_expired_deadline_is_refused(
    llm_provider_factory: Any, gen_request: GenerationRequest, expired_deadline: Deadline
) -> None:
    """Work must not start against an expired deadline.

    Beginning a generation that cannot finish spends the money and produces nothing.
    """
    from prag.core.errors import DeadlineExceeded

    provider = llm_provider_factory()
    with pytest.raises(DeadlineExceeded):
        await provider.generate(gen_request, expired_deadline)


async def test_health_is_reported(llm_provider_factory: Any) -> None:
    healthy = await llm_provider_factory().health()
    failing = await llm_provider_factory(fail=True).health()
    assert healthy.usable
    assert not failing.usable
