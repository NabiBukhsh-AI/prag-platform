# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Note that the platform versions four things independently of this package version: the
configuration contract, the response envelope schema, adapter versions, and embedding index
versions. Their compatibility rules are specified separately.

## [Unreleased]

### Changed
- The grounded path and the parametric shadow step both decide through the fusion policy. An
  off-topic retrieval now abstains, and a parametric answer the corpus independently supports is
  reported as hybrid.
- The strategy router checks adapter coverage per tenant and domain, and admits private-data
  queries to the parametric route only with a tenant-scoped adapter.
- The parametric/retrieval independence contract now covers both packages.
- A request that ends by exception now carries its state on the exception (set by the engine),
  and `Platform.answer` publishes, counts and traces every outcome through one finish step. A
  request that abstained after the retrieval screen dropped groups used to lose those security
  events; it no longer does.
- The HTTP layer no longer records its own span; the platform's `prag.request` trace replaces it.
- The in-memory span buffer is bounded.
- Guardrail verdicts that are security-relevant are published as events by
  `RequestState.with_verdict`, so no call site can record a block and forget the alert.
- The graph engine re-raises `IsolationViolation` instead of treating it as a node failure; a
  fallback must never answer from what is left of a request that crossed a tenant boundary.
- `Platform.answer` wraps the graph in the input and output chains; the HTTP endpoints, the seed
  script and the evaluation runner all use it. `Platform.request_state` replaces the HTTP
  module's private state builder, so nothing outside HTTP needs a web framework.
- The seed documents moved to `eval/seed/corpus/` and are shared by the seed script, the golden
  set and the adversarial suite.

### Fixed
- Cluster coherence is clamped to 1.0; float error on a one-member cluster produced a value the
  economics model rejects.
- The local extractive provider padded answers with sentences sharing a single word with the
  query. It now keeps only sentences within a relative margin of the best match; the evaluation
  gate caught this as citation precision 0.87 against a 0.95 floor.
- `httpx` is declared in the `dev` extra; FastAPI's test client needs it.

### Added
- `fusion.policy`: the §11.4 decision table (`TablePolicy`), independence-corrected agreement,
  source-versus-source conflict detection with surfacing or authority resolution, parametric
  conflict detection, and staleness warnings with strict-mode abstention.
- `fusion.calibration`: an isotonic calibrator and expected calibration error.
- The envelope carries conflicts and staleness warnings, with explanatory prose generated from
  that structure.
- `parametric`: the eligibility gate, the adapter registry, checksum-verified LRU residency,
  the tenant-scoped centroid selector, and document revocation.
- `parametric.pipeline`: clustering, augmentation and its quality filter, training, probes, the
  promotion gate, and shadow-to-active promotion.
- `parametric.local` and `parametric.serving`: the local stand-in trainer and multi-adapter
  provider, supporting single-best, weighted-merge and sequential-probe composition.
- `fusion`: per-sentence stance detection, entailment-based provenance shadowing, and the
  per-adapter conflict monitor that queues retraining and demotes adapters.
- `orchestration.parametric`: the parametric and shadow nodes and the `parametric_answer` graph,
  used only when `parametric.enabled` is set.
- `scripts/compare_parametric.py`: parametric versus non-parametric on the same corpus subset.
- `core`: `ProvenanceShadower`, `ShadowReport`, `EventKind.PARAMETRIC_SERVED`,
  `GenerationRequest.tenant_id`, `GenerationResult.mean_logprob`, and
  `AdapterRecord.blob_sha256`.
- `observability.events`: an in-memory event bus. Every request publishes its accumulated events
  once it has finished, whatever the outcome; a failing consumer never fails the publisher.
- `observability.metrics`: declared metrics with enforced label sets, rendered in Prometheus
  exposition format at `/metrics`.
- `observability.tracing.request_spans`: one trace per request with the §16.1 span names, built
  from the final state, and OTLP export with real timings and parenting (the `otel` extra).
- `deploy/observability/`: Prometheus alert rules (isolation, canary and ACL-mismatch pages;
  abstention, citation validity, degradation, latency, injection and cost tickets) and a Grafana
  dashboard, tested against the declared metrics, labels and emitted reason codes.
- `evaluation.judge`: an LLM faithfulness judge that records its provenance on every score and
  grades the answer in a region with no instruction authority, plus `calibrate` against human
  labels.
- `guardrails`: input and output chains built from configured names (unknown names fail at
  startup, unimplemented ones are reported as deferred on `/health`); injection and
  instruction-override pattern families, PII detection with policy-gated redaction, payload
  limits, citation validation, and leakage detection for secrets, exfiltration-shaped URLs and
  foreign canaries. The chain fails closed when a guardrail raises.
- `guardrails.screen`: the retrieval-phase `EvidenceScreen` — canary sighting fails the request,
  an ACL recheck that re-verifies tenant and ACL per member, source quarantine, and
  document-borne injection detection over child and parent text.
- `guardrails.tools`: a provenance-gated tool executor that refuses calls attributed to the
  evidence region, to no region, or to a region the request never rendered.
- `evaluation`: one implementation of every metric (retrieval, coverage@k, groundedness,
  citation precision, completeness, abstention, adversarial outcome), scorecards, a regression
  gate with floors and baseline tolerance, deterministic online sampling, and judge calibration
  tracking; uncalibrated judge metrics gate nothing.
- `eval/seed/`: a synthetic corpus with golden and adversarial sets and committed baselines, and
  `scripts/run_eval.py`, whose exit code is the CI gate.
- `.github/workflows/ci.yml`: lint, dependency contracts, tests, the adversarial suite and the
  evaluation gate, all blocking.
- `core`: `EvidenceScreen` protocol, `ScreenResult` and `ToolCall` models,
  `ChunkMetadata.tenant_id`, and `EvalSample.retrieved_document_ids`.
- `intelligence`: T0 rules classifier, cascade query analyzer, two-stage strategy router (hard
  constraints, then expected utility over an empirical quality table), and four gated,
  budgeted query transforms.
- `retrieval`: planner and `RetrievalPlanner` protocol, parallel orchestrator with per-leg
  deadlines, partial-results policy and per-source circuit breaker, a BM25 lexical source, and
  reciprocal rank fusion.
- `evidence.rerankers`: skippable rerank tiers; the ladder's first rung.
- `caching.keys`: key construction carrying tenant, ACL set, config and embedding versions, and
  the never-cache rules enforced in one place.
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
