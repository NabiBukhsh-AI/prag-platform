"""Sentence-buffered streaming.

Streaming and output validation look incompatible: you cannot validate text you have already
sent. The resolution is to buffer to a sentence boundary, run the fast output guardrails over
the completed sentence, and release it only then.

That costs one sentence of latency — roughly 60 to 120 ms — and it is the only arrangement that
makes streaming and safety compatible. Token-by-token release means the guardrails run after the
client has already read the output, which is not a guardrail; withholding the whole answer means
not streaming at all.

Heavier checks — grounding verification and citation binding — run on the complete answer after
the last token. When one fails, the client receives a **correction event on the same stream**.
The protocol says so explicitly rather than pretending a stream is immutable, because pretending
would mean either shipping an unsupported claim or silently dropping text the reader already saw.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prag.core.models.generation import GenerationChunk

__all__ = [
    "CorrectionEvent",
    "CorrectionKind",
    "SentenceBuffer",
    "StreamEvent",
    "buffered_stream",
]

#: A sentence boundary: terminal punctuation followed by whitespace. Deliberately simple, and
#: the failure mode is benign — a missed boundary buffers slightly longer, it never releases
#: unchecked text.
_BOUNDARY = re.compile(r"(?<=[.!?])(\s+)")

#: Released without waiting for a boundary once the buffer reaches this size. A model producing
#: a long unpunctuated run — a list, a code block, a table — would otherwise stall the stream
#: indefinitely, and a stalled stream is worse for the reader than a slightly coarse boundary.
_MAX_BUFFER_CHARS = 400


class CorrectionKind(StrEnum):
    """Why a correction was issued after text had already been sent."""

    #: A claim was not entailed by its evidence and has been withdrawn.
    CLAIM_RETRACTED = "claim_retracted"
    #: A citation resolved to no evidence group in this request. A hallucinated reference is a
    #: hallucination even when the claim it decorates happens to be true.
    CITATION_STRIPPED = "citation_stripped"
    #: The answer was regenerated once with a tightened prompt naming the unsupported claim.
    REGENERATED = "regenerated"


@dataclass(frozen=True, slots=True)
class CorrectionEvent:
    """A retraction or amendment issued after the fact."""

    kind: CorrectionKind
    detail: str
    #: What is being corrected, so a client can locate it rather than re-rendering everything.
    claim: str | None = None
    span: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One item on the wire: released text, or a correction."""

    text: str = ""
    ttft_ms: int | None = None
    finish_reason: str | None = None
    correction: CorrectionEvent | None = None

    @property
    def is_correction(self) -> bool:
        return self.correction is not None


class SentenceBuffer:
    """Accumulates tokens and releases completed sentences.

    Kept separate from the streaming loop so it can be tested without an event loop or a
    provider — boundary handling is where the fiddly cases live, and they deserve direct tests.
    """

    def __init__(self, *, max_chars: int = _MAX_BUFFER_CHARS) -> None:
        self._buffer = ""
        self._max_chars = max_chars

    def feed(self, text: str) -> list[str]:
        """Add text and return whatever sentences are now complete."""
        self._buffer += text
        released: list[str] = []

        while True:
            match = _BOUNDARY.search(self._buffer)
            if match is None:
                break
            end = match.end()
            released.append(self._buffer[:end])
            self._buffer = self._buffer[end:]

        if len(self._buffer) >= self._max_chars:
            # Release on the last word boundary rather than mid-word: a split token is visible
            # to the reader in a way a slightly early sentence break is not.
            cut = self._buffer.rfind(" ")
            if cut > 0:
                released.append(self._buffer[: cut + 1])
                self._buffer = self._buffer[cut + 1 :]

        return released

    def flush(self) -> str:
        """Release whatever is left. Called once, after the provider's final chunk."""
        remaining, self._buffer = self._buffer, ""
        return remaining

    @property
    def pending(self) -> str:
        return self._buffer


#: Applied to each completed sentence before release. Returns the text to emit — possibly
#: modified, as with a redaction — or ``None`` to withhold it entirely.
SentenceGuard = Callable[[str], Awaitable[str | None]]


async def buffered_stream(
    chunks: AsyncIterator[GenerationChunk],
    *,
    guard: SentenceGuard | None = None,
    max_chars: int = _MAX_BUFFER_CHARS,
) -> AsyncIterator[StreamEvent]:
    """Wrap a provider stream so nothing reaches the client before it has been checked.

    ``ttft_ms`` is carried from the provider's first chunk onto the first *released* event, not
    onto the first token. Time to first token is what the reader experiences, and under buffering
    that is the moment a sentence appears — reporting the provider's internal first token would
    flatter the number by exactly the buffering delay this design deliberately accepts.
    """
    buffer = SentenceBuffer(max_chars=max_chars)
    provider_ttft: int | None = None
    emitted_any = False
    finish_reason: str | None = None

    async for chunk in chunks:
        if chunk.ttft_ms is not None and provider_ttft is None:
            provider_ttft = chunk.ttft_ms
        if chunk.finish_reason is not None:
            finish_reason = str(chunk.finish_reason)

        for sentence in buffer.feed(chunk.text):
            released = sentence if guard is None else await guard(sentence)
            if released is None:
                continue
            yield StreamEvent(
                text=released,
                ttft_ms=None if emitted_any else provider_ttft,
            )
            emitted_any = True

    tail = buffer.flush()
    if tail:
        released = tail if guard is None else await guard(tail)
        if released is not None:
            yield StreamEvent(text=released, ttft_ms=None if emitted_any else provider_ttft)
            emitted_any = True

    yield StreamEvent(finish_reason=finish_reason or "stop")
