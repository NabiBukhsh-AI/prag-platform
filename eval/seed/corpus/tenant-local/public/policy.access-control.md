# Access Control Policy

## Roles

The platform defines five roles: viewer, contributor, curator, tenant admin, and platform admin.
A curator manages sources and authority scores. Adapter promotion and authority overrides require
curator or higher and are audit-logged.

## Tenant isolation

Isolation is enforced at five independent layers, and any single layer failing must not produce a
breach. Row-level security scopes every query by tenant. The vector index filters on payload.
Candidate processing rechecks ACLs without trusting the index. Adapter selection filters by
tenant scope before scoring. Every cache key carries the tenant id.

## Secrets

Secrets live in a dedicated manager and are rotated on schedule. They are never stored in
environment variables in production and never committed to a repository.

## Authority scoring

Every registered source carries an authority score between zero and one, and a set of
domain-scoped overrides. A source that is the system of record for legal matters may be
worthless on engineering ones, and collapsing that into a single number forces one of those two
judgements to be wrong everywhere it is applied.

Authority scores are set by a curator and carry the curator's identity and a timestamp.
Authority is the strongest signal in conflict resolution, so a score with no accountable owner
is a way for one person's opinion to silently outrank a system of record.

## Staleness

Sources move through a staleness state machine: fresh, aging, stale, expired, reindexing.
Entering the stale state enqueues a reindex automatically. Entering expired makes the source
unusable in strict mode, and usable elsewhere only with an explicit age warning attached to the
answer.

Staleness is measured against each source's expected half-life rather than against a fixed
window. A two-year-old constitutional provision is fresh; a two-day-old exchange rate is not.

## Right to erasure

Erasure is a first-class workflow rather than a manual procedure. It removes the document from
object storage, tombstones and purges it from every index, purges cache entries by document id,
identifies every adapter trained on it, revokes those adapters, and queues the affected clusters
for retraining. The whole chain is recorded in the audit log.

Revocation degrades to the non-parametric path. It never degrades to serving stale weights.
