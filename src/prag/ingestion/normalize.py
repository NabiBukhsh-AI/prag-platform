"""Normalization: raw text into the canonical Document IR.

Markdown, HTML and plain text are handled here because they need no external dependency and
they cover the corpus Phase 1 seeds from. PDF and DOCX arrive as separate ``Extractor``
implementations; nothing in this module or downstream changes when they do, which is the point
of normalizing to one shape.

Document typing is a deterministic decision function over structural signals. It is a *claim*
about the document, and the chunking selector re-checks it against structural reality rather
than trusting it — extraction routinely reports structure that is not there.
"""

from __future__ import annotations

import hashlib
import re
import time
from html.parser import HTMLParser
from typing import ClassVar

from prag.core.ids import derive_id
from prag.core.models.document import Block, BlockKind, Document, DocumentType

__all__ = ["classify_document_type", "normalize_html", "normalize_markdown", "normalize_text"]

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_MD_FENCE = re.compile(r"^\s*```")
_MD_QUOTE = re.compile(r"^\s*>\s?(.*)$")
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_DIVIDER = re.compile(r"^\s*\|[\s:|-]+\|\s*$")

#: A numbered clause: "4.2 Retention", "Section 7.", "Article 12". The marker of legal, policy
#: and regulatory text, where clause boundaries are semantic and must never be split through.
_CLAUSE = re.compile(r"^\s*(?:section|article|clause)?\s*\d+(?:\.\d+)*\.?\s+\S")

#: "Speaker:" or "[00:12:04] Speaker:" — transcripts and chat logs.
_SPEAKER_TURN = re.compile(r"^\s*(?:\[[\d:.\s]+\]\s*)?([A-Z][\w .'-]{1,40}):\s+\S")


def content_hash(text: str) -> str:
    """Stable hash of a document's content.

    Every ingestion stage is keyed on this, so an unchanged document short-circuits the whole
    pipeline. Whitespace is normalized first: a reformatted file with identical content should
    not trigger a reindex and invalidate every cache entry derived from it.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _block(
    kind: BlockKind,
    text: str,
    order: int,
    heading_path: tuple[str, ...],
    level: int | None = None,
) -> Block:
    return Block(
        block_id=derive_id("block", text, str(order)),
        kind=kind,
        text=text.strip(),
        level=level,
        heading_path=heading_path,
        order=order,
    )


def _trim_heading_path(path: list[tuple[int, str]], level: int) -> tuple[str, ...]:
    """Drop headings at or below the incoming level, then return the remaining trail.

    An H2 closes every open H3, so the path under a new H2 is just its ancestors. Without this
    the trail accumulates every heading ever seen and stops describing where a block sits.
    """
    while path and path[-1][0] >= level:
        path.pop()
    return tuple(text for _, text in path)


def normalize_markdown(
    raw: str,
    *,
    document_id: str,
    tenant_id: str,
    source_id: str,
    version: str = "v1",
    title: str | None = None,
    **document_fields: object,
) -> Document:
    """Parse Markdown into blocks, preserving the heading hierarchy."""
    blocks: list[Block] = []
    heading_stack: list[tuple[int, str]] = []
    order = 0

    paragraph: list[str] = []
    in_fence = False
    fenced: list[str] = []

    def flush_paragraph() -> None:
        nonlocal order, paragraph
        if not paragraph:
            return
        text = " ".join(paragraph).strip()
        if text:
            path = tuple(t for _, t in heading_stack)
            kind = BlockKind.CLAUSE if _CLAUSE.match(text) else BlockKind.PARAGRAPH
            blocks.append(_block(kind, text, order, path))
            order += 1
        paragraph = []

    for line in raw.splitlines():
        if _MD_FENCE.match(line):
            if in_fence:
                blocks.append(
                    _block(
                        BlockKind.CODE,
                        "\n".join(fenced),
                        order,
                        tuple(t for _, t in heading_stack),
                    )
                )
                order += 1
                fenced = []
            else:
                flush_paragraph()
            in_fence = not in_fence
            continue

        if in_fence:
            fenced.append(line)
            continue

        heading = _MD_HEADING.match(line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            path = _trim_heading_path(heading_stack, level)
            blocks.append(_block(BlockKind.HEADING, text, order, path, level=level))
            order += 1
            heading_stack.append((level, text))
            continue

        if not line.strip():
            flush_paragraph()
            continue

        path = tuple(t for _, t in heading_stack)

        if _MD_TABLE_DIVIDER.match(line):
            continue
        if _MD_TABLE_ROW.match(line):
            flush_paragraph()
            kind = (
                BlockKind.TABLE_HEADER
                if not blocks or blocks[-1].kind is not BlockKind.TABLE_ROW
                else BlockKind.TABLE_ROW
            )
            blocks.append(_block(kind, line.strip(), order, path))
            order += 1
            continue

        quote = _MD_QUOTE.match(line)
        if quote:
            flush_paragraph()
            blocks.append(_block(BlockKind.QUOTE, quote.group(1), order, path))
            order += 1
            continue

        item = _MD_LIST_ITEM.match(line)
        if item:
            flush_paragraph()
            blocks.append(_block(BlockKind.LIST_ITEM, item.group(1), order, path))
            order += 1
            continue

        paragraph.append(line.strip())

    flush_paragraph()
    if fenced:
        # An unterminated fence. Keep the content rather than discarding it: a truncated file is
        # still worth indexing, and dropping the tail loses exactly the part that got cut off.
        blocks.append(
            _block(BlockKind.CODE, "\n".join(fenced), order, tuple(t for _, t in heading_stack))
        )

    resolved_title = title or next((b.text for b in blocks if b.is_heading and b.level == 1), None)
    return _assemble(
        blocks,
        raw,
        document_id=document_id,
        tenant_id=tenant_id,
        source_id=source_id,
        version=version,
        title=resolved_title,
        **document_fields,
    )


class _HTMLToBlocks(HTMLParser):
    """Collect text under structural tags, discarding presentation.

    Uses the standard library parser rather than a dependency. It is lenient about malformed
    markup, which matters because scraped HTML routinely is, and a strict parser would reject
    documents a lenient one indexes usefully.
    """

    _BLOCK_TAGS: ClassVar[dict[str, tuple[BlockKind, int | None]]] = {
        "h1": (BlockKind.HEADING, 1),
        "h2": (BlockKind.HEADING, 2),
        "h3": (BlockKind.HEADING, 3),
        "h4": (BlockKind.HEADING, 4),
        "h5": (BlockKind.HEADING, 5),
        "h6": (BlockKind.HEADING, 6),
        "p": (BlockKind.PARAGRAPH, None),
        "li": (BlockKind.LIST_ITEM, None),
        "blockquote": (BlockKind.QUOTE, None),
        "pre": (BlockKind.CODE, None),
        "td": (BlockKind.TABLE_ROW, None),
        "th": (BlockKind.TABLE_HEADER, None),
    }
    #: Never indexed. Script and style are not content, and nav/footer boilerplate matches
    #: everything while answering nothing.
    _SKIP_TAGS: ClassVar[frozenset[str]] = frozenset(
        {"script", "style", "nav", "footer", "header", "aside"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.collected: list[tuple[BlockKind, str, int | None]] = []
        self._current: tuple[BlockKind, int | None] | None = None
        self._buffer: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self._BLOCK_TAGS:
            self._flush()
            kind, level = self._BLOCK_TAGS[tag]
            self._current = (kind, level)

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in self._BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._current is None:
            return
        self._buffer.append(data)

    def _flush(self) -> None:
        if self._current is None:
            return
        text = " ".join("".join(self._buffer).split())
        if text:
            kind, level = self._current
            self.collected.append((kind, text, level))
        self._buffer = []
        self._current = None

    def close(self) -> None:
        super().close()
        self._flush()


def normalize_html(
    raw: str,
    *,
    document_id: str,
    tenant_id: str,
    source_id: str,
    version: str = "v1",
    title: str | None = None,
    **document_fields: object,
) -> Document:
    """Parse HTML into blocks, discarding scripts, styles, and navigation boilerplate."""
    parser = _HTMLToBlocks()
    parser.feed(raw)
    parser.close()

    blocks: list[Block] = []
    heading_stack: list[tuple[int, str]] = []

    for order, (kind, text, level) in enumerate(parser.collected):
        if kind is BlockKind.HEADING and level is not None:
            path = _trim_heading_path(heading_stack, level)
            blocks.append(_block(kind, text, order, path, level=level))
            heading_stack.append((level, text))
        else:
            resolved = BlockKind.CLAUSE if _CLAUSE.match(text) else kind
            blocks.append(_block(resolved, text, order, tuple(t for _, t in heading_stack)))

    resolved_title = title or next((b.text for b in blocks if b.is_heading and b.level == 1), None)
    return _assemble(
        blocks,
        raw,
        document_id=document_id,
        tenant_id=tenant_id,
        source_id=source_id,
        version=version,
        title=resolved_title,
        **document_fields,
    )


def normalize_text(
    raw: str,
    *,
    document_id: str,
    tenant_id: str,
    source_id: str,
    version: str = "v1",
    title: str | None = None,
    **document_fields: object,
) -> Document:
    """Split plain text on blank lines, classifying clauses and speaker turns."""
    blocks: list[Block] = []
    for order, part in enumerate(p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()):
        if _SPEAKER_TURN.match(part):
            kind = BlockKind.SPEAKER_TURN
        elif _CLAUSE.match(part):
            kind = BlockKind.CLAUSE
        else:
            kind = BlockKind.PARAGRAPH
        blocks.append(_block(kind, part, order, ()))

    return _assemble(
        blocks,
        raw,
        document_id=document_id,
        tenant_id=tenant_id,
        source_id=source_id,
        version=version,
        title=title,
        **document_fields,
    )


def classify_document_type(blocks: tuple[Block, ...]) -> DocumentType:
    """Infer the document's genre from structural signals alone.

    Deterministic and cheap. This is a claim about the document, not a guarantee: the chunking
    selector re-checks it against structural reality, because extraction reports structure that
    is not there often enough to matter.
    """
    if not blocks:
        return DocumentType.DEGRADED

    total = len(blocks)
    counts: dict[BlockKind, int] = {}
    for block in blocks:
        counts[block.kind] = counts.get(block.kind, 0) + 1

    def share(kind: BlockKind) -> float:
        return counts.get(kind, 0) / total

    # Checked before headings: a policy manual has both, and clause boundaries are the stronger
    # constraint because splitting through one changes what it says.
    if share(BlockKind.CLAUSE) >= 0.30:
        return DocumentType.LEGAL_OR_POLICY
    if share(BlockKind.SPEAKER_TURN) >= 0.40:
        return DocumentType.TRANSCRIPT
    if share(BlockKind.TABLE_ROW) + share(BlockKind.TABLE_HEADER) >= 0.50:
        return DocumentType.TABULAR
    if share(BlockKind.CODE) >= 0.50:
        return DocumentType.CODE
    if sum(1 for b in blocks if b.is_heading) >= 2:
        return DocumentType.STRUCTURED_PROSE
    return DocumentType.UNSTRUCTURED_PROSE


def _assemble(
    blocks: list[Block],
    raw: str,
    *,
    document_id: str,
    tenant_id: str,
    source_id: str,
    version: str,
    title: str | None,
    **document_fields: object,
) -> Document:
    ordered = tuple(blocks)
    return Document(
        document_id=document_id,
        tenant_id=tenant_id,
        source_id=source_id,
        version=version,
        content_hash=content_hash(raw),
        doc_type=classify_document_type(ordered),
        blocks=ordered,
        title=title,
        created_at_ms=int(time.time() * 1000),
        **document_fields,  # type: ignore[arg-type]
    )
