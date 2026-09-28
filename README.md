# prag-platform

A **Parametric RAG** platform. It answers queries by composing three classes of knowledge and
deciding, per query, which to use:

| Knowledge class | Where it lives | Retrieved artifact |
|---|---|---|
| Weight-resident | Base model pretraining | — |
| Adapter-retrieved | Adapter registry + object store | A low-rank parameter delta, merged at inference |
| Text-retrieved | Vector / lexical / graph / SQL indexes | A span of text, placed in the context window |

The routing decision between those three is the central architectural concern, not an
afterthought. Everything else in this repository exists to make that decision measurable,
traceable, and reversible.

> **Parametric RAG** means retrieval whose retrieved artifact is *a set of model parameters*
> rather than a span of text. The corpus is pre-encoded offline into low-rank adapters; at
> inference the retriever selects which deltas are relevant and merges them. This is not
> "RAG plus fine-tuning" — the parametric component is itself retrieved, per query, from an
> index, and refreshed on the same cadence as a document index.

## Shape of the system

A **modular monolith** in Python: one deployable codebase, strict internal package boundaries,
no cross-package imports except through the protocols declared in `core/`. Three process types
(sync API/orchestrator, async workers, evaluation workers) over PostgreSQL, Qdrant, OpenSearch,
Redis, and a vLLM tier with multi-LoRA serving.

The request path is a **deterministic typed state graph**, not an agent. Agents are confined to
three bounded roles (multi-hop research, SQL with self-repair, adversarial verification) where
the step count genuinely cannot be known in advance. Everything else is a fixed graph with
explicit edges, timeouts, and fallbacks, because a deterministic graph is testable, traceable,
and cost-predictable in a way an agent loop is not.

Five decisions carry the design:

1. **Parametric knowledge is a tiered store, not a monolith.** Tier 1 adapters cover *form*
   (vocabulary, schema, tone); Tier 2 covers *fact* at the topic-cluster level. Per-document
   adapters do not scale and are capped to a small hot set.
2. **Parametric knowledge is uncitable, unrevocable, and unfilterable per request.** An
   explicit eligibility gate blocks anything tenant-scoped, ACL-sensitive, or under a
   revocation SLA shorter than the retrain cadence.
3. **Fusion is a policy, not a scalar.** A decision table over calibrated confidence, source
   authority, half-life-relative freshness, and provenance-deduplicated agreement.
4. **Routing is a small trained classifier, not an LLM call.** Roughly 15 to 20 ms. A 300 ms
   LLM call cannot sit on the critical path of a 650 ms TTFT budget.
5. **Retrieved text is data, never instruction.** Evidence sits in an isolated context region
   and can never authorize a tool call.

## Status

**Phases 1 to 4 are complete** — up to the items that need production traffic or GPUs. The
build order was followed strictly: `core/` first and in full, then the contract suites, then
storage with in-memory fakes, then the graph engine, then vertical slices one at a time.

**866 tests passing**, ruff clean, all 7 dependency contracts enforced by `import-linter`, and the
evaluation gate passing on the seed golden and adversarial sets. CI runs all of it on every push
and pull request, and every step blocks.

```bash
docker compose up -d && make seed   # a working system, no cloud dependency
make eval                           # scorecards plus the blocking regression gate
```

### Foundations — complete

| Component | What it holds |
|---|---|
| `core/` | 35 protocols, 130 domain models, typed errors, the budget ladder, monotonic deadlines, DI container |
| `tests/contract/` | Conformance suites for 7 protocols, parameterized over every registered implementation |
| `storage/` | Knowledge registry, 4 repositories, in-memory vector store with a vendor-neutral filter dialect |
| `orchestration/graph` | Serializable graph definitions with load-time validation, plus the interpreter |
| `config/` | Full settings contract, cross-section validation, four-layer loading, tenant allow-list |

### Phase 1 — complete

| Slice | What it delivers |
|---|---|
| Ingestion | Document IR, Markdown/HTML/text normalization, deterministic chunking with parent-child output, indexing |
| Retrieval | Vector `KnowledgeSource` with index-side ACL filtering, vendor-neutral filter dialect |
| Evidence | Near-duplicate linking and independence marking from shared lineage |
| Context | Region budgeting, value-density packing, four ordering modes, isolated evidence region |
| Generation | Model routing, generation profiles, local extractive provider, sentence-buffered streaming |
| Grounding | Claim extraction, entailment, citations bound only after verification |
| Orchestration | `standard_answer` graph with an abstention path, five nodes, protocol-injected dependencies |
| API | `/health`, `/v1/answer` (JSON and SSE), `/v1/ingest`, identity resolution, one-place error mapping |
| Observability | Span attribute schema with enforced redaction |
| Local stack | `docker-compose.yml`, `Dockerfile`, seed script that proves a query answers end to end |

### Phase 2 — complete

| Slice | What it delivers |
|---|---|
| Query intelligence | T0 rules classifier, cascade analyzer with honest per-field confidence and tier reporting |
| Strategy routing | Hard constraints first (live data, exact quotation, private data), then expected-utility scoring over an empirical quality table with a Bayesian prior, exploration, and hedging |
| Query transforms | Rewrite, coreference, alias expansion, rule-based decomposition into a sub-query DAG — each gated and budgeted |
| Retrieval orchestration | Plans as data; parallel legs with per-leg deadlines, partial-results policy, per-source circuit breaker |
| Hybrid retrieval | BM25 lexical source alongside dense, fused with reciprocal rank fusion (no score calibration needed) |
| Reranking | Tiered and skippable; the degradation ladder's cheapest rung, with the skip reason recorded |
| Caching | Key construction carrying tenant, ACL set, config and embedding versions; never-cache rules enforced in one place |

T1 (trained encoder) and T2 (LLM fallback) classifiers are deliberately not built: T1 needs labels
that Phase 2 traffic is meant to produce, and building it first would mean training on synthetic
data and calling the result empirical. The analyzer counts escalations, so the case for building
them is measurable.

### Phase 3 — complete (except what needs production data)

| Slice | What it delivers |
|---|---|
| Guardrail chain | Input and output chains configured by name: injection and instruction-override pattern families, PII detection with policy-gated redaction, payload limits, citation validation, leakage (secrets, exfiltration-shaped URLs, foreign canaries). Fails closed; unknown names fail at startup |
| Retrieval screen | Per-group checks at retrieval time: canary sighting fails the request, ACL recheck that does not trust the index filter, source quarantine, document-borne injection |
| Tool gate | Provenance-gated executor: a tool call attributed to the evidence region, or to nothing, is refused before it runs |
| Security events | Every security-relevant verdict becomes an event in one place; critical ones are isolation alerts |
| Metrics | Recall/precision/hit rate/MRR/NDCG@k, coverage@k, groundedness, citation precision, completeness, abstention correctness, adversarial outcome — one implementation, used by every trigger |
| Regression gate | Floors from the evaluation config plus tolerance against a committed baseline; judge-scored metrics gate nothing until the judge is calibrated |
| Datasets | Synthetic seed corpus with golden and adversarial sets (`eval/seed/`); real sets live in the private files repository |
| Online sampling | Deterministic per-request sampling that publishes an evaluation event off the request path |
| LLM judge | Faithfulness judge that records its model and version on every score, grades the answer in a region with no instruction authority, and excludes unparseable output rather than scoring it zero; `calibrate` measures its agreement with human labels |
| Event bus | Every request publishes its events once it has finished, however it ended — including one that abstained after the screen dropped groups, whose security events used to be lost |
| Tracing | One trace per request using the §16.1 span names, built from the final state; exported over OTLP with real timings and parenting when `observability.otel_endpoint` is set (`otel` extra) |
| Metrics and alerts | Declared metrics exposed at `/metrics` for Prometheus; alert rules and a Grafana dashboard in `deploy/observability/`, tested so every metric, label and reason code they use actually exists |
| CI | `.github/workflows/ci.yml`: lint, contracts, tests, adversarial suite, evaluation gate |

The gate earned its place on its first run: it caught the local provider padding answers with
off-topic sentences (citation precision 0.87 against a 0.95 floor), which is now fixed at the
cause.

Waiting on production data rather than code: the T1 classifier (needs labels from real traffic),
the judge-versus-human calibration report (needs human labels; the calibration it would report is
built and gates the judge), the tenant content-policy classifier, and poisoning heuristics (need
per-source embedding distributions). The last two are reported as deferred on `/health` rather
than assumed covered. Evidence-utilization, adapter and calibration-drift alerts arrive with the
phases that produce those signals; `alerts.yml` lists them.

### Phase 4 — complete (local stand-in for training and serving)

| Slice | What it delivers |
|---|---|
| Eligibility gate | A pure function reporting every hard blocker — knowledge class, ACL narrower than the tenant, revocation SLA shorter than the retrain cadence, exact quotation, short half-life, per-claim provenance without shadowing, contested, PII — and weighing economics (volume, savings threshold, cluster coherence) only once none apply |
| Adapter registry | The single writer for immutable rows, weight blobs, residency and the routing snapshot; promotion deprecates the previous version; erasing a document revokes and evicts every adapter trained on it |
| Residency | LRU cache that re-reads registry status on every load and verifies blob checksums on cold loads, so a stale selection cannot load a revoked or corrupted adapter |
| Selection | Tenant scope is a hard filter before scoring, re-asserted per record; base-model and embedding-version mismatches are excluded; below the coverage floor it returns nothing |
| Offline pipeline | Clustering, augmentation with a non-optional entailment/dedup/diversity filter, training, held-out recall, general-regression and interference probes, the promotion gate, shadow, then active — and tenant knowledge can never train a global adapter |
| Parametric route | `parametric_answer` graph: the adapter answers with no evidence, retrieval runs as provenance shadowing, entailed claims are cited and the rest marked unsourced, and the answer states it came from learned knowledge |
| Evidence wins | A claim the corpus contradicts sends the request down the grounded path and publishes a conflict event naming the adapter |
| Serving-to-training loop | A per-adapter conflict-rate monitor queues stale adapters for retraining and demotes critical ones, whose traffic falls back to retrieval on the next request |
| Isolation | Cross-tenant canaries at selection, training and serving: the weight store refuses a foreign or unattributed adapter at load |
| Comparison | `scripts/compare_parametric.py` runs the golden set with and without the tier and reports route, cost, latency, groundedness and completeness |

**Training and serving use a local stand-in** (`prag.parametric.local`) that memorises filtered
QA pairs. It exercises every gate and route without a GPU and says nothing about how well LoRA
learns; PEFT training and vLLM multi-LoRA serving replace it behind the same shapes. On the seed
set, 2 of 9 queries are served parametrically at about 3% of the grounded path's cost with equal
groundedness, and the rest fall back — mostly because the local hashing embedder's coverage scores
are low, which is why the tests and comparison lower the coverage floor from 0.62 to 0.35.

### Phase 5 — in progress

| Slice | State |
|---|---|
| Fusion decision policy | Done. The §11.4 table as code, one test per row; the two abstention rules are hard gates evaluated before any score |
| Confidence calibration | Done. Isotonic calibrator with expected calibration error on held-out data; an uncalibrated calibrator reports the worst error rather than a flattering one |
| Independence correction | Done. Sources sharing a lineage root count once, at their strongest authority |
| Conflict surfacing | Done. Source-versus-source conflicts are surfaced at comparable authority and resolved toward the stronger source otherwise; both appear in the envelope as structure, and the prose is generated from it |
| Staleness | Done. Evidence older than the query's half-life carries a warning, or abstains in strict mode |
| Memory tiers | Next |
| Graph and SQL sources, multi-hop | After memory |

Fusion also changed two behaviours for the better: an off-topic retrieval now abstains instead of
answering with its nearest sentence, and a parametric answer the corpus independently supports is
reported as hybrid rather than parametric.

### Later phases

| Phase | Scope | State |
|---|---|---|
| 6 | Bounded agents | not started |
| 7 | Scale and multi-tenancy hardening | not started |

**The parametric tier arrives in Phase 4, not Phase 1.** You cannot know what is worth
parameterizing until a non-parametric baseline has been measured, and without evaluation and
observability in place there is no way to tell whether an adapter helped. The system is
designed to be fully useful with `parametric.enabled: false`, and it ships that way.

## Layout

```
src/prag/
  core/            protocols, domain models, errors, budget, DI — depends on nothing internal
  api/             FastAPI routers, middleware, SSE streaming
  orchestration/   graph engine, graph definitions, nodes, policies, replay
  intelligence/    query understanding and strategy routing (one module, deliberately)
  parametric/      adapter registry, selection, eligibility gate, offline training
  retrieval/       plan construction and execution over KnowledgeSource implementations
  evidence/        dedup, fusion, reranking, selection (one pipeline, deliberately)
  context/         budget, pack, order, compress, validate, render
  fusion/          confidence, authority, freshness, agreement, decision policy
  generation/      model routing, providers, streaming, grounding, tools
  agents/          the three bounded agents, each with caps and a deterministic fallback
  memory/          four memory tiers with citation-namespace separation
  ingestion/       connectors, extraction, chunking, embedding, indexing, lineage
  knowledge/       registry, authority administration, staleness
  guardrails/      one chain, all phases
  evaluation/      one implementation of every metric, three triggers
  caching/         tiers, key construction, invalidation
  observability/   tracing, metrics, logging, event bus
  storage/         repositories, no business logic
  config/          the configuration contract and its layering
  workers/         ingestion, training, evaluation, maintenance
eval/seed/         synthetic corpus, golden and adversarial sets, committed baselines
deploy/            Prometheus alert rules and the Grafana dashboard
```

Dependency direction is enforced by `importlinter.ini`, not by convention. `core` imports
nothing internal, `storage` holds no business logic, `orchestration` sees protocols and never
implementations, and nothing imports `api`.

## Getting started

```bash
make dev      # venv + editable install with dev extras
make check    # ruff, import contracts, mypy --strict, unit + contract tests
make up       # full local stack
make seed     # seeded corpus
make test     # everything that needs no cloud credentials
```

The design goal is that `docker compose up` plus a seed script yields a working system on a
laptop with a small model, so the whole thing can be implemented and tested end to end with no
cloud dependencies.

## Conventions worth knowing before you write code

- **No vendor SDK type crosses a module boundary.** A Qdrant response becomes a `Candidate`
  inside the adapter, not outside it. If a caller can tell which vector store is in use, the
  abstraction has failed.
- **Every external call takes a `Deadline` and honors it.** No unbounded awaits, no default
  timeouts inherited from a client library.
- **Dependencies arrive through the DI container.** No module constructs another module's
  concrete implementation. No global singletons, no service locator.
- **Every failure path is typed.** `core/errors.py` owns the hierarchy; the API layer maps
  errors to responses in exactly one place.
- **Configuration before code.** New behavior gets a config field with a default first, then
  the code that reads it. No magic numbers in modules.
- **Guardrails are never optional at the code level.** The chain always runs; individual
  guardrails are toggled by config.
- **Nothing that failed validation is cached.** Enforced in key construction, not left to
  callers.

Deliberately absent: a generic "AI engine" facade, an agent that decides which retriever to
call, an LLM call on the pre-generation critical path, a `utils/` package, and classes named
`Manager`, `Handler`, or `Processor`.

## How retrieval is shaped

**Parent-child is the default, everywhere.** The child chunk is embedded and matched, because a
300-token paragraph about one thing embeds far more precisely than a 2000-token section about
six. The parent is what reaches the model, because the paragraph that matched usually cannot
answer on its own. Neither side compromises for the other, and no global chunk size needs tuning.

**Chunking strategy is a deterministic decision function**, never a model call. Ingestion is
idempotent and keyed on a content hash, so re-running is free — a property a nondeterministic
step would destroy. Document type is treated as a *claim* the selector re-checks against
structural reality, because extraction reports headings that are not there often enough to
matter.

**ACL filtering happens at the index, not after it.** Filtering downstream means fetching rows
the caller may not see, and fetched data can leak through a log line, a cache entry, or a timing
difference even when it never reaches the response.

**Running out of time is not a source failing.** A deadline breach returns a partial result;
wrapping it as an unavailable source would open the circuit breaker on a healthy backend and
turn one slow query into an outage.

## How context is assembled

**Regions, not concatenation.** The window is partitioned into named regions with explicit
allocations. That makes overflow deterministic, and it is the structural basis for injection
defense: the evidence region carries no instruction authority, and a string built by
concatenation cannot express "this part is data".

**The evidence cap is a quality control, not a cost control.** Past roughly 8k tokens, marginal
retrieved chunks lower answer quality by diluting attention — so evidence is capped even when
the window has room to spare. Raising the cap to fill the window makes answers worse.

**Packing optimises value per token, with two corrections.** A redundancy penalty stops the
greedy pass filling the budget with near-identical chunks about the most salient aspect, which
are exactly the chunks that score highest. A coverage bonus rewards evidence touching an aspect
nothing selected has covered yet — the difference between answering the loudest part of a
question very well and answering the question.

**Ordering is not cosmetic.** Attention over a long context is not uniform, so edge-weighted
ordering puts the strongest evidence at both ends and the weakest in the middle. Chronological
and source-grouped modes exist because they are correct for timeline and source-comparison
queries respectively.

**Near-duplicates are linked, not dropped.** Three documents quoting one press release are one
piece of evidence. The link is what stops the agreement signal from counting a single fact
several times, which is how a system becomes most confident exactly where it is most wrong.

**Silent truncation is prohibited.** If evidence was dropped or a query aspect went uncovered,
the bundle carries a coverage warning and the response has to say so.

## How answers are grounded

**Citations are bound only after entailment is checked.** A plausible citation on a claim its
source does not support is worse than no citation: an uncited claim reads as the model's
assertion and a reader discounts it, while a cited one reads as verified. Post-hoc attachment
converts an unsupported statement into an apparently checked one, most convincingly where the
model was least reliable.

**An invalid citation is a hallucination even when the claim is true.** A marker resolving to no
evidence group in this request means the model cited something it was never given, and a system
that tolerates that cannot tell a lucky guess from a grounded answer.

**Abstention is a success state, returned as a 200 with an envelope.** A 4xx would make the
abstention rate indistinguishable from client error in every dashboard that groups by status —
and abstention rate is a metric with both an upper and a lower alert. Every abstention carries a
reason code and, where possible, a suggested next action.

**Streaming buffers to a sentence boundary.** Releasing token by token runs the output guardrails
after the client has already read the text, which is not a guardrail. Corrections are part of the
SSE protocol rather than an embarrassment: grounding runs on the complete answer, and a claim can
fail it after the reader has seen it.

## Documentation

The full architecture specification — 34 sections covering interfaces, data models, latency and
cost budgets, failure modes, security, and the phased roadmap — lives in the private companion
repository, `prag-platform_files`, along with per-tenant configuration and evaluation datasets.
Section references throughout this codebase (`7.3`, `22.1`, and so on) point into it.

## License

Apache-2.0.
