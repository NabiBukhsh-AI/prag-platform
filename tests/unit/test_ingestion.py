"""Normalization, document typing, chunking, and chunk validation."""

from __future__ import annotations

import pytest

from prag.core.models.document import BlockKind, ChunkingStrategyName, Document, DocumentType
from prag.core.protocols import Chunker
from prag.ingestion import (
    RecursiveCharacterChunker,
    StructureAwareChunker,
    content_hash,
    default_registry,
    normalize_html,
    normalize_markdown,
    normalize_text,
    select_strategy,
    validate_chunks,
)
from prag.ingestion.chunking import estimate_tokens

RUNBOOK = """# Incident Response

Intro text that sets out the scope of this runbook and who it applies to across
the organisation, including which teams are on the rota.

## Severity levels

Sev-1 means a total outage affecting all tenants. It pages the on-call lead
immediately and opens a bridge call. The incident commander is whoever is on the
primary rota at the time the page fires, not whoever happens to be awake.

Sev-2 means degraded service for a subset of tenants. It pages during business
hours only, and the response target is four hours rather than fifteen minutes.

## Escalation

For a sev-1 incident the on-call lead must be paged within 15 minutes of
detection. If the page is unacknowledged after 5 minutes, escalation moves to the
engineering manager, and after a further 10 minutes to the director of
engineering.

Escalation is automatic and does not require anyone to make a judgement call at
three in the morning, which is the entire point of having a written policy.
"""


def a_markdown_doc(raw: str = RUNBOOK, **kwargs: object) -> Document:
    defaults: dict[str, object] = {
        "document_id": "doc-1",
        "tenant_id": "tenant-a",
        "source_id": "kb.runbooks",
    }
    return normalize_markdown(raw, **{**defaults, **kwargs})  # type: ignore[arg-type]


class TestContentHash:
    def test_is_stable(self) -> None:
        assert content_hash("hello world") == content_hash("hello world")

    def test_ignores_reformatting(self) -> None:
        """A reformatted file with identical content must not trigger a reindex.

        Reindexing invalidates every cache entry derived from the document, so a whitespace-only
        diff would throw away real work for no change in meaning.
        """
        assert content_hash("a  b\n\nc") == content_hash("a b c")

    def test_detects_real_changes(self) -> None:
        assert content_hash("30 days") != content_hash("60 days")


class TestMarkdownNormalization:
    def test_headings_build_a_hierarchy(self) -> None:
        doc = a_markdown_doc()
        headings = [b for b in doc.blocks if b.is_heading]

        assert [h.text for h in headings] == [
            "Incident Response",
            "Severity levels",
            "Escalation",
        ]
        assert headings[0].level == 1
        assert headings[1].level == 2

    def test_heading_path_tracks_ancestry(self) -> None:
        """A chunk with no idea which section it came from is much easier to misread."""
        doc = a_markdown_doc()
        escalation = next(
            b for b in doc.blocks if b.kind is BlockKind.PARAGRAPH and "15 minutes" in b.text
        )
        assert escalation.heading_path == ("Incident Response", "Escalation")

    def test_sibling_heading_closes_the_previous_one(self) -> None:
        """An H2 closes every open H3, or the trail accumulates every heading ever seen."""
        doc = a_markdown_doc("# A\n\ntext\n\n## B\n\ntext\n\n## C\n\nunder c\n")
        under_c = next(b for b in doc.blocks if b.text == "under c")
        assert under_c.heading_path == ("A", "C"), "B must not still be open"

    def test_code_fences_are_atomic(self) -> None:
        doc = a_markdown_doc("# T\n\n```\ndef f():\n    return 1\n```\n")
        code = next(b for b in doc.blocks if b.kind is BlockKind.CODE)
        assert "def f():" in code.text
        assert not code.is_splittable, "half a function does not compile"

    def test_unterminated_fence_keeps_its_content(self) -> None:
        """A truncated file is still worth indexing, and the tail is what got cut off."""
        doc = a_markdown_doc("# T\n\n```\nstill here\n")
        assert any("still here" in b.text for b in doc.blocks)

    def test_lists_quotes_and_tables_are_recognised(self) -> None:
        doc = a_markdown_doc("# T\n\n- an item\n\n> a quote\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
        kinds = {b.kind for b in doc.blocks}
        assert BlockKind.LIST_ITEM in kinds
        assert BlockKind.QUOTE in kinds
        assert BlockKind.TABLE_HEADER in kinds

    def test_numbered_clauses_are_recognised(self) -> None:
        """Clause boundaries are semantic, and splitting through one changes what it says."""
        doc = a_markdown_doc("# Policy\n\n4.2 Records are retained for thirty days.\n")
        clause = next(b for b in doc.blocks if b.kind is BlockKind.CLAUSE)
        assert "thirty days" in clause.text
        assert not clause.is_splittable

    def test_title_defaults_to_the_h1(self) -> None:
        assert a_markdown_doc().title == "Incident Response"


class TestHtmlNormalization:
    def test_extracts_structure(self) -> None:
        doc = normalize_html(
            "<h1>Title</h1><p>Body text here.</p><h2>Section</h2><p>More text.</p>",
            document_id="d",
            tenant_id="t",
            source_id="s",
        )
        assert doc.title == "Title"
        assert doc.heading_count == 2

    def test_discards_scripts_and_boilerplate(self) -> None:
        """Script is not content, and nav matches everything while answering nothing."""
        doc = normalize_html(
            "<nav><p>Home About</p></nav>"
            "<script>var x = 1;</script>"
            "<h1>Real</h1><p>Real content.</p>"
            "<footer><p>Copyright</p></footer>",
            document_id="d",
            tenant_id="t",
            source_id="s",
        )
        text = doc.text
        assert "Real content." in text
        assert "Home About" not in text
        assert "var x" not in text
        assert "Copyright" not in text

    def test_malformed_markup_still_yields_content(self) -> None:
        """Scraped HTML routinely is malformed; a strict parser would reject usable documents."""
        doc = normalize_html(
            "<h1>Title<p>unclosed paragraph<p>another",
            document_id="d",
            tenant_id="t",
            source_id="s",
        )
        assert "unclosed paragraph" in doc.text


class TestDocumentTyping:
    def test_headings_make_it_structured_prose(self) -> None:
        assert a_markdown_doc().doc_type is DocumentType.STRUCTURED_PROSE

    def test_no_structure_is_unstructured_prose(self) -> None:
        doc = normalize_text(
            "Just a paragraph.\n\nAnd another one.",
            document_id="d",
            tenant_id="t",
            source_id="s",
        )
        assert doc.doc_type is DocumentType.UNSTRUCTURED_PROSE

    def test_numbered_clauses_make_it_legal(self) -> None:
        """Checked before headings: a policy manual has both, and clauses bind harder."""
        doc = normalize_text(
            "1.1 The first clause applies here.\n\n"
            "1.2 The second clause applies there.\n\n"
            "2.1 A third clause covers something else.",
            document_id="d",
            tenant_id="t",
            source_id="s",
        )
        assert doc.doc_type is DocumentType.LEGAL_OR_POLICY

    def test_speaker_turns_make_it_a_transcript(self) -> None:
        doc = normalize_text(
            "Alice: We should ship on Friday.\n\n"
            "Bob: That seems early given the open bugs.\n\n"
            "Alice: Agreed, let us revisit on Wednesday.",
            document_id="d",
            tenant_id="t",
            source_id="s",
        )
        assert doc.doc_type is DocumentType.TRANSCRIPT

    def test_empty_document_is_degraded(self) -> None:
        """Routed to the conservative chunker rather than guessed at."""
        doc = normalize_text("", document_id="d", tenant_id="t", source_id="s")
        assert doc.doc_type is DocumentType.DEGRADED


class TestStrategySelection:
    def test_structured_prose_gets_structure_aware(self) -> None:
        assert select_strategy(a_markdown_doc()) is ChunkingStrategyName.STRUCTURE_AWARE

    def test_type_is_a_claim_that_gets_checked(self) -> None:
        """A document typed structured whose extraction found no headings falls back.

        Splitting on structure that is not there produces confidently wrong boundaries, and
        confidently wrong is worse than plainly approximate.
        """
        doc = a_markdown_doc("# Only one heading\n\nJust body text, no hierarchy.\n")
        assert doc.heading_count == 1
        assert select_strategy(doc) is ChunkingStrategyName.RECURSIVE_CHARACTER

    def test_unavailable_strategy_degrades_rather_than_failing(self) -> None:
        """A deployment without a syntax-aware chunker must still ingest code."""
        doc = a_markdown_doc()
        chosen = select_strategy(
            doc, available=frozenset({ChunkingStrategyName.RECURSIVE_CHARACTER})
        )
        assert chosen is ChunkingStrategyName.RECURSIVE_CHARACTER

    def test_degraded_documents_get_the_conservative_chunker(self) -> None:
        doc = normalize_text("", document_id="d", tenant_id="t", source_id="s")
        assert select_strategy(doc) is ChunkingStrategyName.RECURSIVE_CHARACTER

    def test_registry_resolves_a_chunker_for_a_document(self) -> None:
        assert isinstance(default_registry().for_document(a_markdown_doc()), Chunker)

    def test_unregistered_strategy_raises(self) -> None:
        from prag.core.errors import ConfigurationError
        from prag.ingestion import ChunkerRegistry

        with pytest.raises(ConfigurationError, match="no chunker registered"):
            ChunkerRegistry().get(ChunkingStrategyName.STRUCTURE_AWARE)


class TestStructureAwareChunking:
    def test_produces_parent_child_pairs(self) -> None:
        """The highest-leverage retrieval decision: precision from the child, sufficiency
        from the parent."""
        doc = a_markdown_doc()
        chunks = StructureAwareChunker(min_section_tokens=20, target_child_tokens=40).chunk(doc)

        assert len(chunks) >= 3
        with_parents = [c for c in chunks if c.parent_text]
        assert with_parents, "sections larger than one child must yield a parent"
        for chunk in with_parents:
            assert chunk.text in chunk.parent_text
            assert chunk.context_text == chunk.parent_text

    def test_child_is_what_gets_embedded_parent_is_what_reaches_the_model(self) -> None:
        doc = a_markdown_doc()
        chunk = next(
            c
            for c in StructureAwareChunker(min_section_tokens=20, target_child_tokens=40).chunk(doc)
            if c.parent_text
        )
        assert len(chunk.text) < len(chunk.parent_text)
        assert chunk.context_text == chunk.parent_text

    def test_heading_path_is_carried_and_embedded(self) -> None:
        """Two sections can both say "30 days"; only the heading tells them apart."""
        doc = a_markdown_doc()
        chunks = StructureAwareChunker(min_section_tokens=20, target_child_tokens=40).chunk(doc)
        escalation = next(c for c in chunks if "Escalation" in c.heading_path)

        assert escalation.heading_path[0] == "Incident Response"
        assert escalation.prefixed_text.startswith("Incident Response > Escalation")

    def test_tiny_sections_merge_forward(self) -> None:
        """A heading followed by two sentences is a label, not a section.

        Merging forward keeps the heading attached to the content it introduces; merging
        backward would file "Exceptions" under the previous section.
        """
        doc = a_markdown_doc("# A\n\n## Tiny\n\nOne line.\n\n## Also tiny\n\nAnother line.\n")
        chunks = StructureAwareChunker(min_section_tokens=200).chunk(doc)
        assert len(chunks) == 1

    def test_unsplittable_blocks_survive_whole(self) -> None:
        """A truncated code block looks complete while having lost what made it correct."""
        long_code = "\n".join(f"    line_{i} = compute({i})" for i in range(200))
        doc = a_markdown_doc(f"# A\n\ntext\n\n## Code\n\n```\n{long_code}\n```\n")
        chunks = StructureAwareChunker(
            min_section_tokens=10, target_child_tokens=100, max_child_tokens=100
        ).chunk(doc)

        code_chunks = [c for c in chunks if "line_0 = " in c.text]
        assert code_chunks, "the code block should appear"
        assert "line_199" in code_chunks[0].text, "it must not have been cut"

    def test_ids_are_deterministic(self) -> None:
        """Re-running ingestion on unchanged content must not churn the index."""
        chunker = StructureAwareChunker(min_section_tokens=20, target_child_tokens=40)
        first = chunker.chunk(a_markdown_doc())
        second = chunker.chunk(a_markdown_doc())
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_document_metadata_reaches_every_chunk(self) -> None:
        """Fusion needs authority and the ACL hash without a second fetch per candidate."""
        doc = a_markdown_doc(acl_hash="acl-eng", authority=0.93)
        for chunk in StructureAwareChunker(min_section_tokens=20, target_child_tokens=40).chunk(
            doc
        ):
            assert chunk.acl_hash == "acl-eng"
            assert chunk.authority == pytest.approx(0.93)
            assert chunk.tenant_id == "tenant-a"
            assert chunk.lineage_root == "doc-1"

    def test_applies_only_to_a_real_hierarchy(self) -> None:
        """One heading is a title, not a hierarchy."""
        chunker = StructureAwareChunker()
        assert chunker.applies_to(a_markdown_doc())
        assert not chunker.applies_to(a_markdown_doc("# Only\n\nbody\n"))

    def test_rejects_inconsistent_size_config(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            StructureAwareChunker(target_child_tokens=900, max_child_tokens=800)


class TestRecursiveChunking:
    def test_splits_on_the_most_meaningful_separator(self) -> None:
        text = "\n\n".join(f"Paragraph {i} with enough words to matter here." for i in range(40))
        doc = normalize_text(text, document_id="d", tenant_id="t", source_id="s")
        chunks = RecursiveCharacterChunker(target_tokens=60, overlap_ratio=0.0).chunk(doc)

        assert len(chunks) > 1
        assert all(c.token_estimate <= 120 for c in chunks)

    def test_overlap_carries_context_across_boundaries(self) -> None:
        """A fact on an arbitrary boundary would otherwise be retrievable from neither side."""
        text = "\n\n".join(f"Paragraph {i} with several words in it." for i in range(30))
        doc = normalize_text(text, document_id="d", tenant_id="t", source_id="s")

        without = RecursiveCharacterChunker(target_tokens=60, overlap_ratio=0.0).chunk(doc)
        with_overlap = RecursiveCharacterChunker(target_tokens=60, overlap_ratio=0.2).chunk(doc)

        assert sum(c.token_estimate for c in with_overlap) > sum(c.token_estimate for c in without)

    def test_no_fictional_parent(self) -> None:
        """Boundaries here are arbitrary, so a "section" would be a fiction."""
        doc = normalize_text("word " * 2000, document_id="d", tenant_id="t", source_id="s")
        chunks = RecursiveCharacterChunker(target_tokens=100).chunk(doc)
        assert all(c.parent_text is None for c in chunks)

    def test_handles_an_unbroken_run(self) -> None:
        """A base64 blob has no separator to prefer, so a hard cut is the only option."""
        doc = normalize_text("x" * 20_000, document_id="d", tenant_id="t", source_id="s")
        chunks = RecursiveCharacterChunker(target_tokens=100).chunk(doc)
        assert len(chunks) > 1

    def test_empty_document_yields_nothing(self) -> None:
        doc = normalize_text("   \n\n  ", document_id="d", tenant_id="t", source_id="s")
        assert RecursiveCharacterChunker().chunk(doc) == ()

    def test_rejects_excessive_overlap(self) -> None:
        with pytest.raises(ValueError, match="overlap_ratio"):
            RecursiveCharacterChunker(overlap_ratio=0.6)

    def test_applies_to_anything(self) -> None:
        assert RecursiveCharacterChunker().applies_to(a_markdown_doc())


class TestChunkValidation:
    def test_keeps_substantive_chunks(self) -> None:
        chunks = StructureAwareChunker(min_section_tokens=20, target_child_tokens=40).chunk(
            a_markdown_doc()
        )
        report = validate_chunks(chunks)
        assert len(report.kept) >= 3
        assert report.rejection_rate == 0.0

    @pytest.mark.parametrize(
        "junk", ["Page 4 of 12", "Figure 7", "-----------", "• • •", "42", "   "]
    )
    def test_rejects_page_furniture(self, junk: str) -> None:
        """It carries no information but still occupies an index slot and still matches."""
        doc = normalize_text(junk, document_id="d", tenant_id="t", source_id="s")
        chunks = RecursiveCharacterChunker().chunk(doc)
        assert len(validate_chunks(chunks).kept) == 0

    def test_high_rejection_rate_points_at_extraction(self) -> None:
        """A third of a source's chunks failing is an extraction bug, not a content problem.

        Chasing it as a chunking problem wastes a day.
        """
        doc = normalize_text(
            "\n\n".join(["Page 1", "Page 2", "Page 3", "Real content that carries meaning."]),
            document_id="d",
            tenant_id="t",
            source_id="s",
        )
        report = validate_chunks(RecursiveCharacterChunker(target_tokens=10).chunk(doc))
        assert report.extraction_suspect()

    def test_clean_source_is_not_suspect(self) -> None:
        chunks = StructureAwareChunker(min_section_tokens=20, target_child_tokens=40).chunk(
            a_markdown_doc()
        )
        report = validate_chunks(chunks)
        assert not report.extraction_suspect()


class TestTokenEstimation:
    def test_empty_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_grows_with_length(self) -> None:
        assert estimate_tokens("word " * 100) > estimate_tokens("word " * 10)

    def test_dense_text_estimates_higher_per_character(self) -> None:
        """Symbols and CJK tokenize close to one token per character."""
        prose = "the quick brown fox jumps over it"
        dense = "{}[]<>{}[]<>{}[]<>{}[]<>{}[]<>{}[]"
        assert estimate_tokens(dense) > estimate_tokens(prose[: len(dense)])


class TestHeadingsAreNeverAlone:
    def test_a_heading_never_becomes_its_own_chunk(self) -> None:
        """A heading is a label for what follows, not a retrievable unit.

        Emitted alone it either fails validation or, worse, slips through and matches queries
        it cannot possibly answer.
        """
        chunker = StructureAwareChunker(min_section_tokens=10, target_child_tokens=15)
        chunks = chunker.chunk(a_markdown_doc())

        headings = {"Incident Response", "Severity levels", "Escalation"}
        assert not [c for c in chunks if c.text.strip() in headings]

    def test_no_chunk_is_below_the_validation_floor(self) -> None:
        """Chunking and validation must agree, or the pipeline discards its own output."""
        chunker = StructureAwareChunker(min_section_tokens=10, target_child_tokens=15)
        report = validate_chunks(chunker.chunk(a_markdown_doc()))
        assert report.rejection_rate == 0.0
