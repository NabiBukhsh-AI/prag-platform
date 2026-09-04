"""Recursive character chunking: the conservative fallback.

Used when a document has no structure worth respecting, or when extraction degraded badly enough
that its apparent structure cannot be trusted. It assumes nothing about the content, which is
exactly why it is the right choice in both cases — a structure-aware chunker splitting on
headings that extraction hallucinated produces confidently wrong boundaries, and confidently
wrong is worse than plainly approximate.

Overlap exists here and not in the structure-aware chunker for a reason. When boundaries are
arbitrary, a fact near a boundary would otherwise be split across two chunks and retrievable from
neither. Where boundaries are meaningful, overlap just duplicates content and inflates the
apparent independence of two chunks that are largely the same text.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

from prag.core.ids import derive_id
from prag.core.models.document import Chunk, ChunkingStrategyName, Document
from prag.ingestion.chunking.tokens import estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["RecursiveCharacterChunker"]

#: Separators from most to least semantically meaningful. The splitter descends this list only
#: as far as it must, so a paragraph break is always preferred over a sentence break, and a
#: sentence break over cutting mid-word.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


class RecursiveCharacterChunker:
    """Splits on the most meaningful separator that fits the size target."""

    name = ChunkingStrategyName.RECURSIVE_CHARACTER

    def __init__(
        self,
        *,
        target_tokens: int = 512,
        overlap_ratio: float = 0.15,
    ) -> None:
        if not 0.0 <= overlap_ratio < 0.5:
            raise ValueError(
                f"overlap_ratio must be in [0, 0.5), got {overlap_ratio}; "
                "at half the chunk size every fact would appear in three chunks"
            )
        self._target = target_tokens
        self._overlap_tokens = int(target_tokens * overlap_ratio)

    def applies_to(self, document: Document) -> bool:
        """The universal fallback: it applies to anything.

        The selector consults it last, so returning True here never shadows a better strategy.
        """
        return True

    def chunk(self, document: Document) -> Sequence[Chunk]:
        text = document.text
        if not text.strip():
            return ()

        pieces = self._split(text, self._target)
        overlapped = self._add_overlap(pieces)

        return tuple(
            Chunk(
                chunk_id=derive_id("chunk", document.document_id, document.version, str(order)),
                document_id=document.document_id,
                document_version=document.version,
                tenant_id=document.tenant_id,
                source_id=document.source_id,
                text=piece,
                # No natural parent: the boundaries are arbitrary, so a "section" would be a
                # fiction. Windowing neighbours at retrieval time is the honest alternative.
                parent_text=None,
                heading_path=(),
                order=order,
                strategy=self.name,
                token_estimate=estimate_tokens(piece),
                acl_hash=document.acl_hash,
                authority=document.authority,
                volatility_class=document.volatility_class,
                lineage_root=document.document_id,
            )
            for order, piece in enumerate(overlapped)
        )

    def _split(self, text: str, target: int) -> list[str]:
        """Recursively split until every piece fits, descending the separator list."""
        if estimate_tokens(text) <= target:
            return [text] if text.strip() else []

        for separator in _SEPARATORS:
            if separator == "":
                # Last resort: a hard character cut. Reached only for a single unbroken run
                # longer than the target, such as a base64 blob or minified source.
                return self._hard_split(text, target)
            if separator not in text:
                continue

            parts = text.split(separator)
            if len(parts) == 1:
                continue

            merged: list[str] = []
            buffer = ""
            for part in parts:
                candidate = f"{buffer}{separator}{part}" if buffer else part
                if estimate_tokens(candidate) > target and buffer:
                    merged.append(buffer)
                    buffer = part
                else:
                    buffer = candidate
            if buffer:
                merged.append(buffer)

            # A piece can still exceed the target if one part alone is oversized, so recurse
            # into whatever is still too big rather than returning it.
            result: list[str] = []
            for piece in merged:
                if estimate_tokens(piece) > target:
                    result.extend(self._split(piece, target))
                else:
                    result.append(piece)
            return [p for p in result if p.strip()]

        return self._hard_split(text, target)

    @staticmethod
    def _hard_split(text: str, target: int) -> list[str]:
        span = max(1, target * 4)
        return [text[i : i + span] for i in range(0, len(text), span) if text[i : i + span].strip()]

    def _add_overlap(self, pieces: list[str]) -> list[str]:
        """Prepend a tail of the previous piece to each chunk.

        A fact sitting on an arbitrary boundary is otherwise split across two chunks and
        retrievable from neither, which is the specific failure this exists to prevent.
        """
        if self._overlap_tokens <= 0 or len(pieces) < 2:
            return pieces

        overlap_chars = self._overlap_tokens * 4
        out = [pieces[0]]
        for previous, current in pairwise(pieces):
            tail = previous[-overlap_chars:].lstrip()
            out.append(f"{tail} {current}" if tail else current)
        return out
