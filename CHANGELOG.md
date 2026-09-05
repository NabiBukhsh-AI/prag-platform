# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Note that the platform versions four things independently of this package version: the
configuration contract, the response envelope schema, adapter versions, and embedding index
versions. Their compatibility rules are specified separately.

## [Unreleased]

### Added
- `core`: 24 protocols, 112 domain models, the typed error hierarchy, the budget controller
  with its six-rung degradation ladder, monotonic deadlines, and the DI container.
- Conformance suites for `KnowledgeSource`, `LLMProvider`, `EmbeddingProvider`, `MemoryStore`
  and `CacheTier`, parameterized over every registered implementation, plus in-memory fakes
  that actually enforce their contracts rather than mocking them.
- `storage`: knowledge registry models, four repository protocols, in-memory implementations,
  and a conformance suite per repository.
- `orchestration.graph`: a serializable graph definition with load-time structural validation,
  and an interpreter that enforces the budget between nodes, derives each node's deadline from
  what the request has left, and checks every node's reads/writes declarations.
- `config`: the full Pydantic Settings contract with documented defaults, cross-section
  validation, four-layer loading, and an explicit tenant override allow-list.
- `ingestion`: the canonical Document IR, Markdown/HTML/text normalization, deterministic
  document typing and strategy selection, structure-aware and recursive chunkers with
  parent-child output, and chunk validation that reports a per-source rejection rate.
- `storage.vectorstore`: a vendor-neutral filter dialect and an exact-search in-memory
  `VectorStore`, with its own conformance suite.
- `retrieval`: a vector `KnowledgeSource` doing parent-child retrieval with index-side ACL
  filtering, plus `ingestion.indexing` to close the ingest-to-retrieve loop.
- `evidence.dedup`: near-duplicate grouping that links rather than drops, and marks
  cross-group independence from shared lineage.
- `context`: region allocation with a hard evidence cap, value-density packing with redundancy
  and coverage corrections, four ordering modes, and a renderer where the evidence region
  carries no instruction authority.
- `generation`: profiles, policy model routing, a local extractive provider, sentence-buffered
  streaming with corrections, and grounding that binds citations only after entailment.
- `orchestration.nodes` and the `standard_answer` graph, with an abstention path.
- `api`: composition root, identity resolution, one-place error mapping, SSE protocol, and a
  FastAPI app exposing `/health`, `/v1/answer` and `/v1/ingest`.
- `observability`: span attribute schema with enforced redaction.
- Local stack: `docker-compose.yml`, `Dockerfile`, and a seed script that proves a query answers
  end to end.
- Repository scaffolding: packaging metadata, Apache-2.0 license, Makefile targets for the
  development, test, and evaluation loops.
- `importlinter.ini` encoding the dependency contracts as enforced CI checks: `core` imports
  nothing internal, `storage` holds no business logic, `orchestration` sees protocols and never
  implementations, nothing imports `api`, and the parametric and retrieval tiers stay mutually
  independent so the parametric tier remains disableable.
