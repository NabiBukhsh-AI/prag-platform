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

Early. Following the build order in the specification: `core/` first and in full, then the
contract suites, then storage with in-memory fakes, then the graph engine, then Phase 1
vertical slices one node at a time.

| Phase | Scope | State |
|---|---|---|
| 1 | Grounded RAG core: protocols, models, graph engine, pgvector, ingestion, citations | in progress |
| 2 | Query intelligence, retrieval orchestration, caching, budget ladder | not started |
| 3 | Evaluation, guardrails, observability | not started |
| 4 | Parametric tier | not started |
| 5 | Fusion, memory, structured knowledge | not started |
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

## Documentation

The full architecture specification — 34 sections covering interfaces, data models, latency and
cost budgets, failure modes, security, and the phased roadmap — lives in the private companion
repository, `prag-platform_files`, along with per-tenant configuration and evaluation datasets.
Section references throughout this codebase (`7.3`, `22.1`, and so on) point into it.

## License

Apache-2.0.
