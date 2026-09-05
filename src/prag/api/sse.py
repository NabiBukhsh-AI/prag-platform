"""The server-sent event protocol.

Four event types, and the third is the one that makes the protocol honest.

``token`` carries released text. ``progress`` reports which node is running, so a long request
shows something other than a spinner. ``correction`` retracts or amends text the client has
already rendered. ``done`` carries the final envelope.

**Corrections are part of the protocol rather than an embarrassment.** Grounding verification
runs on the complete answer, after the last token, and a claim can fail it. The alternatives are
both worse: ship the unsupported claim, or silently drop text the reader already saw. Naming the
correction lets a client render it — strike the sentence, show why — and lets a client that
ignores it at least not be misled about what happened.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from prag.core.models.generation import AnswerEnvelope
    from prag.generation.streaming import StreamEvent

__all__ = ["SseEvent", "format_sse", "stream_envelope"]


class SseEvent(StrEnum):
    TOKEN = "token"
    PROGRESS = "progress"
    #: A retraction or amendment to text already sent.
    CORRECTION = "correction"
    DONE = "done"
    ERROR = "error"


def format_sse(event: SseEvent, data: dict[str, Any]) -> str:
    """Render one SSE frame.

    Compact JSON separators, because a token event is the most frequent frame on the wire and
    the whitespace is pure overhead at that rate.
    """
    payload = json.dumps(data, separators=(",", ":"), default=str)
    return f"event: {event}\ndata: {payload}\n\n"


async def stream_envelope(
    events: AsyncIterator[StreamEvent],
    *,
    envelope_factory: Any,
) -> AsyncIterator[str]:
    """Render a generation stream as SSE frames, ending with the envelope.

    The envelope is always sent, including after an abstention or a correction. A client that
    only reads ``done`` gets a complete, accurate answer with its confidence, citations and
    warnings; a client that renders tokens live gets the same thing plus whatever was corrected
    along the way.
    """
    ttft_sent = False

    async for event in events:
        if event.correction is not None:
            yield format_sse(
                SseEvent.CORRECTION,
                {
                    "kind": str(event.correction.kind),
                    "detail": event.correction.detail,
                    "claim": event.correction.claim,
                    "span": list(event.correction.span) if event.correction.span else None,
                },
            )
            continue

        if event.text:
            frame: dict[str, Any] = {"text": event.text}
            if event.ttft_ms is not None and not ttft_sent:
                frame["ttft_ms"] = event.ttft_ms
                ttft_sent = True
            yield format_sse(SseEvent.TOKEN, frame)

    envelope: AnswerEnvelope = await envelope_factory()
    yield format_sse(SseEvent.DONE, envelope.model_dump(mode="json"))


def progress_frame(node_id: str, *, elapsed_ms: int) -> str:
    """A progress event naming the node currently running.

    Node ids rather than prose. A client showing "retrieving" needs a stable token to map, and
    a human-readable phrase would be a second thing to keep in sync with the graph.
    """
    return format_sse(SseEvent.PROGRESS, {"node": node_id, "elapsed_ms": elapsed_ms})


def error_frame(reason_code: str, detail: str, *, retryable: bool) -> str:
    """A terminal error frame.

    Sent instead of ``done`` when the request failed. It carries the same stable reason code the
    non-streaming path returns, so a client handles one vocabulary rather than two.
    """
    return format_sse(
        SseEvent.ERROR,
        {"reason_code": reason_code, "detail": detail, "retryable": retryable},
    )
