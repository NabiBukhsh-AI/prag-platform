"""Memory items and their namespace boundaries.

The platform holds four memories: working memory (the request state itself), session memory,
long-term user memory, and the parametric store. Only the middle two are ``MemoryStore``
implementations; working memory is a Python object and the parametric store is retrieved as
weights, not as text.

Namespaces are a correctness boundary. A citation may never resolve across them — an answer
cannot cite something the user said three turns ago as though it were a retrieved document, and
the citation validator rejects it if the context builder ever lets one through. Without that
separation, a conversational assertion becomes indistinguishable from a sourced fact, which is
the most quietly damaging kind of contamination the system can have.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from prag.core.models.common import MemoryNamespace, Provenance

__all__ = ["MemoryItem", "MemorySelector", "SessionSummary"]


class MemoryItem(BaseModel):
    """One remembered fact.

    ``provenance`` gates what may persist. Long-term memory must reject
    ``Provenance.MODEL_GENERATED``: letting the model's own output persist as a user fact is how
    a system starts confidently believing things nobody ever told it, and the belief survives
    every subsequent session.
    """

    model_config = ConfigDict(frozen=True)

    item_id: str
    namespace: MemoryNamespace
    text: str
    provenance: Provenance
    created_at_ms: int
    #: How much this item matters, before decay. Combined with age to decide what survives a
    #: bounded store and what gets evicted.
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    last_accessed_ms: int | None = None
    #: Citation marker within the memory namespace, kept separate from evidence markers so the
    #: two can never be confused for one another.
    citation_marker: str | None = None

    @property
    def persistable(self) -> bool:
        """Whether this item may be written to long-term memory.

        Checked at the store boundary rather than trusted from the caller, because a write
        gating rule enforced only by convention is a rule that holds until the first hurried
        change to a caller.
        """
        if self.namespace is not MemoryNamespace.LONG_TERM:
            return True
        return self.provenance in (
            Provenance.USER_ASSERTED,
            Provenance.CONFIRMED_STRUCTURED,
        )


class SessionSummary(BaseModel):
    """A rolling summary of a conversation.

    Summarising after a turn threshold rather than carrying full history keeps the memory region
    bounded. Entities and decisions are kept verbatim alongside the prose, because those are
    exactly what coreference resolution needs to be exact about and exactly what a summary is
    most likely to blur.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    summary: str
    turns_summarized: int = Field(ge=0)
    verbatim_entities: tuple[str, ...] = ()
    verbatim_decisions: tuple[str, ...] = ()
    #: Hash of the summary, used in the rewrite cache key. A rewrite computed against an older
    #: summary is not valid for a newer one.
    summary_hash: str
    updated_at_ms: int


class MemorySelector(BaseModel):
    """What to forget.

    Supports the right-to-erasure workflow, which has to be able to name what it is erasing
    without a full scan. An empty selector matches nothing rather than everything: a forget call
    that deletes a principal's entire memory because a field was left unset is not a failure
    mode worth leaving open.
    """

    model_config = ConfigDict(frozen=True)

    namespace: MemoryNamespace | None = None
    session_id: str | None = None
    item_ids: tuple[str, ...] = ()
    older_than_ms: int | None = None
    provenance: Provenance | None = None

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.namespace is not None,
                self.session_id is not None,
                self.item_ids,
                self.older_than_ms is not None,
                self.provenance is not None,
            )
        )
