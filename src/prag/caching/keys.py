"""Cache key construction and the rules that keep caching from breaking correctness.

Caching is where a working system quietly becomes a wrong one. Every rule here exists because
its absence produces a cache that looks healthy and serves the wrong thing:

**The tenant and the ACL set are part of every key.** Without them an entry crosses a permission
boundary, and the failure is invisible — the answer is well-formed, it is simply someone else's.

**The config version is part of every key.** Behaviour is a function of configuration, so an
entry produced under one config is not a valid answer under another. A tuning change that did not
invalidate caches would be served from entries produced under the old behaviour, and the change
would appear not to work.

**The embedding version is part of every key.** A key that outlives a reindex returns results for
vectors that no longer exist.

**Nothing that failed validation is cacheable.** Enforced here rather than left to callers,
because a caching rule that depends on every call site remembering it gets broken once and then
stays broken invisibly — republishing an unsupported answer indefinitely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from prag.core.ids import short_hash
from prag.core.models.events import CacheKey

if TYPE_CHECKING:
    from prag.core.models.generation import AnswerEnvelope
    from prag.core.models.identity import Principal, TenantPolicy

__all__ = [
    "NEVER_CACHE_REASONS",
    "analysis_key",
    "embedding_key",
    "exact_answer_key",
    "is_cacheable",
    "retrieval_key",
    "ttl_for_volatility",
]

#: Response classes that must never enter a cache, whatever the tier's TTL says. Each is a case
#: where the answer was qualified, and a cache strips the qualification by construction: a
#: warning is attached to one request's circumstances, and the next request does not share them.
NEVER_CACHE_REASONS: Final[frozenset[str]] = frozenset(
    {"abstained", "coverage_warning", "staleness_warning", "validation_failed", "unsourced_claims"}
)

#: TTL by volatility class, in seconds. Realtime is zero — not a short TTL but *no* caching,
#: because a value valid for seconds cached for even a minute is wrong for most of that minute
#: and confidently so.
_TTL_BY_VOLATILITY: Final[dict[str, int]] = {
    "static": 86_400,
    "slow": 14_400,
    "fast": 900,
    "realtime": 0,
}


def ttl_for_volatility(volatility: str) -> int:
    """TTL for a volatility class, defaulting to the most conservative non-zero value.

    An unknown class gets the ``fast`` TTL rather than the ``static`` one. Guessing long on
    something whose staleness is unknown is how a cache serves last week's answer.
    """
    return _TTL_BY_VOLATILITY.get(volatility, _TTL_BY_VOLATILITY["fast"])


def _base(
    tier: str,
    principal: Principal,
    policy: TenantPolicy,
    *,
    embedding_version: str | None = None,
    subject: str = "",
    qualifiers: tuple[tuple[str, str], ...] = (),
) -> CacheKey:
    return CacheKey(
        tier=tier,
        tenant_id=principal.tenant_id,
        # Sorted, so two principals with identical permissions in a different order share an
        # entry. Unsorted, the cache would be correct and nearly useless.
        acl_discriminator=tuple(sorted(principal.acl_hashes)),
        config_version=policy.config_version,
        embedding_version=embedding_version,
        subject_hash=subject,
        qualifiers=qualifiers,
    )


def exact_answer_key(
    query: str,
    principal: Principal,
    policy: TenantPolicy,
    *,
    embedding_version: str | None = None,
) -> CacheKey:
    """Key for a whole answer.

    Strict mode is a qualifier because it changes what the system will say: a request that
    abstains under strict mode and answers without it must not share an entry. So is the SLA
    tier, which selects the rerank tier and therefore the ordering the answer was built from.
    """
    return _base(
        "exact",
        principal,
        policy,
        embedding_version=embedding_version,
        subject=short_hash(" ".join(query.lower().split())),
        qualifiers=(
            ("strict", str(policy.strict_mode).lower()),
            ("sla", str(principal.sla_tier)),
        ),
    )


def retrieval_key(
    query: str,
    principal: Principal,
    policy: TenantPolicy,
    *,
    embedding_version: str,
    top_k: int,
) -> CacheKey:
    """Key for a retrieval result.

    ``top_k`` is a qualifier: a cached top-8 cannot serve a request asking for top-24, and
    serving it would silently narrow recall in a way nothing downstream could detect.
    """
    return _base(
        "retrieval",
        principal,
        policy,
        embedding_version=embedding_version,
        subject=short_hash(" ".join(query.lower().split())),
        qualifiers=(("top_k", str(top_k)),),
    )


def embedding_key(text: str, *, model_id: str, model_version: str) -> CacheKey:
    """Key for one embedding.

    No tenant and no ACL: an embedding is a pure function of text and model, carries no
    permission-bearing content, and is shared across tenants deliberately. That sharing is most
    of the tier's value — a common query embedded once serves everyone.
    """
    return CacheKey(
        tier="embedding",
        tenant_id="*",
        acl_discriminator=(),
        config_version=model_version,
        embedding_version=model_version,
        subject_hash=short_hash(text),
        qualifiers=(("model", model_id),),
    )


def analysis_key(query: str, policy: TenantPolicy) -> CacheKey:
    """Key for a query analysis.

    Tenant-agnostic in content but keyed by config version, because the classifier thresholds
    that produced it are configuration. It carries no retrieved content, so it is shared.
    """
    return CacheKey(
        tier="analysis",
        tenant_id="*",
        acl_discriminator=(),
        config_version=policy.config_version,
        subject_hash=short_hash(" ".join(query.lower().split())),
    )


def is_cacheable(envelope: AnswerEnvelope) -> tuple[bool, str | None]:
    """Whether a response may be cached, and the reason when it may not.

    Returns the reason rather than a bare boolean so the decision is observable. A cache hit
    rate that dropped because more answers carry warnings is a retrieval problem; one that
    dropped because of a key change is a deploy problem, and the two look identical from the
    hit rate alone.
    """
    if envelope.abstention is not None:
        return False, "abstained"
    if envelope.coverage_warning is not None:
        return False, "coverage_warning"
    if envelope.staleness_warning is not None:
        return False, "staleness_warning"
    if envelope.conflicts:
        # A surfaced conflict is a statement about this request's evidence set. Replaying it for
        # a different request asserts a disagreement that request never encountered.
        return False, "validation_failed"
    if envelope.grounding.claims_unsourced > 0:
        # Caching an answer with an unsupported claim republishes it indefinitely, and the
        # republished copy has lost the grounding report that would have flagged it.
        return False, "unsourced_claims"
    return True, None
