"""Heading-bounded chunking with parent-child output.

The default strategy for anything with a heading hierarchy, which is most of what an enterprise
corpus actually contains: wikis, runbooks, policy manuals, product documentation.

The shape it produces is the point. Each section becomes a *parent*, and the section's paragraphs
become *children* sized for embedding. Retrieval matches the child, because a 300-token paragraph
about one thing embeds far more precisely than a 2000-token section about six. Context receives
the parent, because the paragraph that matched usually does not contain enough to answer on its
own.

That resolves the chunk-size tension without tuning a single global number: precision comes from
the child, sufficiency from the parent, and neither has to compromise for the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prag.core.ids import derive_id
from prag.core.models.document import (
    Block,
    BlockKind,
    Chunk,
    ChunkingStrategyName,
    Document,
)
from prag.ingestion.chunking.tokens import estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["StructureAwareChunker"]


class StructureAwareChunker:
    """Splits at heading boundaries, then packs each section's blocks into child chunks."""

    name = ChunkingStrategyName.STRUCTURE_AWARE

    def __init__(
        self,
        *,
        target_child_tokens: int = 400,
        max_child_tokens: int = 800,
        #: Sections below this merge into the next one. A heading followed by two sentences is
        #: a label, not a section, and indexing it alone produces a chunk that matches queries
        #: it cannot answer.
        min_section_tokens: int = 120,
        max_parent_tokens: int = 2000,
    ) -> None:
        if target_child_tokens > max_child_tokens:
            raise ValueError("target_child_tokens cannot exceed max_child_tokens")
        self._target = target_child_tokens
        self._max_child = max_child_tokens
        self._min_section = min_section_tokens
        self._max_parent = max_parent_tokens

    def applies_to(self, document: Document) -> bool:
        """Suitable when the document has a real heading hierarchy.

        One heading is a title, not a hierarchy, so it does not qualify: splitting on it yields
        a single section and all the cost of structure-aware chunking with none of the benefit.
        """
        return document.heading_count >= 2

    def chunk(self, document: Document) -> Sequence[Chunk]:
        sections = self._sections(document)
        chunks: list[Chunk] = []
        order = 0

        for section_blocks in sections:
            parent_text = "\n\n".join(b.text for b in section_blocks)
            if not parent_text.strip():
                continue

            heading_path = self._heading_path(section_blocks)
            parent_id = derive_id(
                "parent", document.document_id, document.version, str(order), heading_path
            )
            # A parent longer than the cap would defeat the point of retrieving a child: the
            # context window fills with one section and crowds out every other source.
            capped_parent = self._truncate(parent_text, self._max_parent)

            for child_text in self._pack_children(section_blocks):
                chunks.append(
                    self._build(
                        document,
                        text=child_text,
                        parent_text=capped_parent,
                        parent_id=parent_id,
                        heading_path=heading_path,
                        order=order,
                    )
                )
                order += 1

        return tuple(chunks)

    def _sections(self, document: Document) -> list[list[Block]]:
        """Group blocks into heading-bounded sections, merging ones too small to stand alone."""
        sections: list[list[Block]] = []
        current: list[Block] = []

        for block in document.blocks:
            if block.is_heading and current:
                sections.append(current)
                current = [block]
            else:
                current.append(block)
        if current:
            sections.append(current)

        return self._merge_small(sections)

    def _merge_small(self, sections: list[list[Block]]) -> list[list[Block]]:
        """Fold undersized sections forward into the next one.

        Forward rather than backward so a heading stays attached to the content it introduces.
        Merging backward would file "Exceptions" under the previous section, which is precisely
        the association that makes a retrieved chunk misleading.
        """
        merged: list[list[Block]] = []
        pending: list[Block] = []

        for section in sections:
            candidate = pending + section
            size = estimate_tokens("\n\n".join(b.text for b in candidate))
            if size < self._min_section:
                pending = candidate
                continue
            merged.append(candidate)
            pending = []

        if pending:
            # Nothing left to merge into: attach to the last section rather than dropping it.
            if merged:
                merged[-1].extend(pending)
            else:
                merged.append(pending)
        return merged

    def _pack_children(self, blocks: Sequence[Block]) -> list[str]:
        """Greedily pack blocks into child chunks near the target size.

        Block boundaries are respected wherever possible: a paragraph split in half embeds worse
        than either half whole, because the vector ends up describing a fragment of an argument.
        Only a single block that exceeds the hard cap is split internally.
        """
        children: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0
        # A heading labels what follows; it is never a retrievable unit on its own. Flushing a
        # buffer holding only headings emits a chunk that validation then rejects — or worse,
        # one that slips through and matches queries it cannot possibly answer.
        buffer_has_content = False

        for block in blocks:
            block_tokens = estimate_tokens(block.text)

            if block_tokens > self._max_child:
                pieces = self._split_oversized(block)
                if buffer and buffer_has_content:
                    children.append("\n\n".join(buffer))
                elif buffer and pieces:
                    # Carry a pending heading onto the first piece rather than stranding it.
                    pieces[0] = "\n\n".join([*buffer, pieces[0]])
                buffer, buffer_tokens, buffer_has_content = [], 0, False
                children.extend(pieces)
                continue

            if buffer_has_content and buffer_tokens + block_tokens > self._target:
                children.append("\n\n".join(buffer))
                buffer, buffer_tokens, buffer_has_content = [], 0, False

            buffer.append(block.text)
            buffer_tokens += block_tokens
            buffer_has_content = buffer_has_content or not block.is_heading

        if buffer_has_content:
            children.append("\n\n".join(buffer))
        elif buffer and children:
            # A trailing heading with nothing under it: attach it to the previous chunk rather
            # than emitting it alone or dropping it.
            children[-1] = children[-1] + "\n\n" + "\n\n".join(buffer)
        return [c for c in children if c.strip()]

    def _split_oversized(self, block: Block) -> list[str]:
        """Split one block that is larger than the hard cap.

        Unsplittable blocks are emitted whole and over-cap on purpose. A truncated code block or
        table row is worse than a large one: it looks complete while having lost the part that
        made it correct.
        """
        if not block.is_splittable:
            return [block.text]

        pieces: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for sentence in _split_sentences(block.text):
            sentence_tokens = estimate_tokens(sentence)
            if buffer and buffer_tokens + sentence_tokens > self._target:
                pieces.append(" ".join(buffer))
                buffer, buffer_tokens = [], 0
            buffer.append(sentence)
            buffer_tokens += sentence_tokens

        if buffer:
            pieces.append(" ".join(buffer))
        return pieces or [block.text]

    @staticmethod
    def _heading_path(blocks: Sequence[Block]) -> tuple[str, ...]:
        for block in blocks:
            if block.is_heading:
                return (*block.heading_path, block.text)
        return blocks[0].heading_path if blocks else ()

    @staticmethod
    def _truncate(text: str, max_tokens: int) -> str:
        if estimate_tokens(text) <= max_tokens:
            return text
        # Characters rather than tokens, since the estimate is approximate either way. The
        # marker matters more than the exact cut: a reader must be able to tell it was cut.
        limit = max_tokens * 4
        return text[:limit].rsplit(" ", 1)[0] + " […]"

    def _build(
        self,
        document: Document,
        *,
        text: str,
        parent_text: str | None,
        parent_id: str | None,
        heading_path: tuple[str, ...],
        order: int,
    ) -> Chunk:
        return Chunk(
            # Derived rather than random, so re-running ingestion on unchanged content produces
            # identical ids and the index is not churned for nothing.
            chunk_id=derive_id("chunk", document.document_id, document.version, str(order)),
            document_id=document.document_id,
            document_version=document.version,
            tenant_id=document.tenant_id,
            source_id=document.source_id,
            text=text,
            parent_text=parent_text if parent_text != text else None,
            parent_id=parent_id,
            heading_path=heading_path,
            order=order,
            strategy=self.name,
            token_estimate=estimate_tokens(text),
            acl_hash=document.acl_hash,
            authority=document.authority,
            volatility_class=document.volatility_class,
            lineage_root=document.document_id,
        )


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ish boundaries.

    Deliberately naive. A proper sentence splitter is a model or a large rule set, and this runs
    only on the rare block that already exceeds the hard cap — where an imperfect boundary costs
    far less than the dependency would.
    """
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


BlockKindsNeedingOwnChunk = frozenset({BlockKind.TABLE_HEADER, BlockKind.CODE})
