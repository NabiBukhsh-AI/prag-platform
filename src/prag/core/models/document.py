"""The canonical document representation, and the chunks derived from it.

Extraction produces wildly different shapes — a PDF gives positioned text runs, a Markdown file
gives a heading tree, a spreadsheet gives rows. Everything downstream is written against this one
shape instead, so adding an extractor never touches chunking, and chunking never learns what a
PDF is.

**Blocks, not a string.** A document flattened to text has thrown away exactly the structure the
chunker needs: where a heading ends, which paragraphs belong under it, whether a table row is a
row or a sentence. Reconstructing that from prose is guesswork; preserving it is free.

**Parent-child is the default shape.** The child chunk is what gets embedded and retrieved, for
precision. The parent is what actually goes in the context window, for sufficiency. That split
resolves the perennial chunk-size tension without tuning one global number, and it is the single
highest-leverage retrieval quality decision available.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from prag.core.models.common import VolatilityClass

__all__ = [
    "Block",
    "BlockKind",
    "Chunk",
    "ChunkingStrategyName",
    "Document",
    "DocumentType",
]


class BlockKind(StrEnum):
    """What a block is, structurally.

    The distinctions that survive here are the ones that change chunking behaviour. A heading
    bounds a section; a table row must not be split mid-row; a code block must not be split
    mid-function. Anything finer belongs to the extractor and dies there.
    """

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_ROW = "table_row"
    TABLE_HEADER = "table_header"
    CODE = "code"
    QUOTE = "quote"
    #: A numbered clause in legal, policy, or regulatory text. Never split within one.
    CLAUSE = "clause"
    #: One speaker's turn in a transcript or chat log.
    SPEAKER_TURN = "speaker_turn"


class DocumentType(StrEnum):
    """The document's genre, which selects a chunking strategy.

    Determined during normalization by a deterministic decision function, never by a model call.
    A model call here would put a nondeterministic step inside a pipeline whose whole value is
    that re-running it is free.
    """

    STRUCTURED_PROSE = "structured_prose"
    UNSTRUCTURED_PROSE = "unstructured_prose"
    LEGAL_OR_POLICY = "legal_or_policy"
    TRANSCRIPT = "transcript"
    TABULAR = "tabular"
    CODE = "code"
    #: Extraction produced something the pipeline could not classify, or produced it badly.
    #: Routed to the most conservative chunker rather than guessed at.
    DEGRADED = "degraded"


class ChunkingStrategyName(StrEnum):
    STRUCTURE_AWARE = "structure_aware"
    CLAUSE_BOUNDED = "clause_bounded"
    SEMANTIC = "semantic"
    SPEAKER_TURN = "speaker_turn"
    ROW_GROUP = "row_group"
    SYNTAX_AWARE = "syntax_aware"
    RECURSIVE_CHARACTER = "recursive_character"


class Block(BaseModel):
    """One structural unit of a document.

    ``heading_path`` is the ancestor heading trail, and it is carried on every block rather than
    reconstructed later. A chunk lifted out of a long document with no idea which section it came
    from is much harder to rank and much easier to misread — "the limit is 30 days" means
    different things under *Retention* and under *Appeals*.
    """

    model_config = ConfigDict(frozen=True)

    block_id: str
    kind: BlockKind
    text: str
    #: Heading depth, for ``HEADING`` blocks. ``None`` for everything else.
    level: int | None = Field(default=None, ge=1, le=6)
    heading_path: tuple[str, ...] = ()
    #: Position in the document, so ordering survives any regrouping.
    order: int = Field(ge=0)

    @property
    def is_heading(self) -> bool:
        return self.kind is BlockKind.HEADING

    @property
    def is_splittable(self) -> bool:
        """Whether a chunker may cut through the middle of this block.

        Code, table rows, and legal clauses are atomic. Half a function does not compile, half a
        row loses its header alignment, and half a clause reverses its meaning as often as not.
        """
        return self.kind not in (
            BlockKind.CODE,
            BlockKind.TABLE_ROW,
            BlockKind.TABLE_HEADER,
            BlockKind.CLAUSE,
        )


class Document(BaseModel):
    """A normalized document, ready for chunking.

    ``content_hash`` is what makes the pipeline idempotent. Every ingestion stage is keyed on
    ``(source_id, document_id, content_hash)``, so an unchanged document short-circuits the whole
    pipeline and re-running costs nothing.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    document_id: str
    tenant_id: str
    source_id: str
    #: A new version on every content change. The old version's chunks are tombstoned rather
    #: than deleted, so in-flight requests holding references do not break mid-answer.
    version: str
    content_hash: str

    doc_type: DocumentType
    blocks: tuple[Block, ...] = ()
    title: str | None = None
    language: str = "en"
    volatility_class: VolatilityClass = VolatilityClass.SLOW
    #: Opaque ACL hash, written into every derived chunk's payload so retrieval can filter on a
    #: cheap equality check rather than joining a permissions table on the request path.
    acl_hash: str = "public"
    authority: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, str] = Field(default_factory=dict)

    created_at_ms: int
    updated_at_ms: int | None = None

    @property
    def text(self) -> str:
        """The whole document as text, for hashing and for whole-document fallbacks."""
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def has_headings(self) -> bool:
        return any(b.is_heading for b in self.blocks)

    @property
    def heading_count(self) -> int:
        return sum(1 for b in self.blocks if b.is_heading)

    def blocks_of(self, *kinds: BlockKind) -> tuple[Block, ...]:
        return tuple(b for b in self.blocks if b.kind in kinds)


class Chunk(BaseModel):
    """One retrievable unit, plus the parent that gives it context.

    ``text`` is embedded and matched. ``parent_text`` is what reaches the model. Keeping both on
    one object rather than joining them later means a retrieval result is self-sufficient — no
    second fetch on the request path to discover what a match was actually about.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    document_version: str
    tenant_id: str
    source_id: str

    text: str
    #: The enclosing section or window. ``None`` when the chunk is already its own parent,
    #: which happens for short documents and for the parent chunks themselves.
    parent_text: str | None = None
    parent_id: str | None = None

    heading_path: tuple[str, ...] = ()
    #: Position among the document's chunks, so neighbours can be recovered for windowing.
    order: int = Field(ge=0)
    strategy: ChunkingStrategyName
    #: Estimated, not measured. The real count depends on the model's tokenizer, and holding a
    #: tokenizer dependency in the domain layer to be exactly right about a budget that is
    #: itself approximate is a bad trade. Named so nobody mistakes it for exact.
    token_estimate: int = Field(ge=0)

    acl_hash: str = "public"
    authority: float = Field(default=0.5, ge=0.0, le=1.0)
    volatility_class: VolatilityClass = VolatilityClass.SLOW
    #: Ultimate ancestor in the derivation graph, carried so the independence correction in
    #: fusion never has to walk the lineage table on the request path.
    lineage_root: str | None = None
    embedding_version: str | None = None

    @property
    def context_text(self) -> str:
        """What goes in the prompt: the parent where one exists."""
        return self.parent_text if self.parent_text is not None else self.text

    @property
    def prefixed_text(self) -> str:
        """The chunk with its heading trail prepended, as embedded.

        The heading path is embedded *with* the chunk rather than stored beside it, because it
        disambiguates otherwise-identical text. Two sections can both say "this applies for 30
        days"; only the heading tells them apart, and a retriever that never sees it cannot.
        """
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n\n" + self.text
