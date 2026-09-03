"""Who is asking, what they are allowed to see, and what the request may spend."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prag.core.models.common import Deadline, SlaTier

__all__ = ["Budget", "Principal", "TenantPolicy", "UtilityWeights"]


class Principal(BaseModel):
    """The resolved caller.

    Every request resolves to one of these before any work happens, and it is threaded through
    retrieval filters, cache keys, adapter selection, and the ACL recheck.

    ``acl_hashes`` is computed once per request rather than per candidate. That is not only an
    optimisation: recomputing it midway through a request would open a window where the
    permissions used to filter the index differ from the ones used to recheck the results, and
    an inconsistency there is indistinguishable from a leak.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    groups: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()

    #: Opaque hashes of the ACL entries this principal satisfies. Compared, never interpreted;
    #: the platform must not be able to reconstruct a tenant's group structure from a cache key.
    acl_hashes: tuple[str, ...] = ()

    sla_tier: SlaTier = SlaTier.STANDARD

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    @property
    def cache_discriminator(self) -> tuple[str, ...]:
        """The part of a cache key that makes an entry unshareable across principals.

        Tenant first, then the sorted ACL set. Sorted because two principals with identical
        permissions in a different order must hit the same cache entry; if they did not, the
        cache would be correct but nearly useless.
        """
        return (self.tenant_id, *sorted(self.acl_hashes))


class UtilityWeights(BaseModel):
    """How a tier trades quality against latency against cost.

    These are the numbers that make routing an optimisation rather than a preference. A
    high-stakes tier weights quality at 0.85 and latency at 0.05, and will therefore accept a
    slow answer; an interactive tier will not.
    """

    model_config = ConfigDict(frozen=True)

    quality: float = Field(ge=0.0, le=1.0)
    latency: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> UtilityWeights:
        total = self.quality + self.latency + self.cost
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"utility weights must sum to 1.0, got {total:.6f}")
        return self


class TenantPolicy(BaseModel):
    """The per-tenant configuration in force for this request.

    Resolved once per request by merging the tenant's overrides over shipped defaults. Only keys
    on the allow-list can be overridden: a tenant may tune how cautious the system is, and may
    not disable a guardrail, weaken isolation, or make failed validations cacheable.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(min_length=1)
    config_version: str

    utility_weights: UtilityWeights
    strict_mode: bool = False
    max_evidence_tokens: int = Field(default=8000, gt=0)
    abstain_below_knowledge_score: float = Field(default=0.42, ge=0.0, le=1.0)
    semantic_cache_similarity_floor: float = Field(default=0.95, ge=0.0, le=1.0)
    parametric_enabled: bool = False

    @property
    def abstains_on_irreconcilable_conflict(self) -> bool:
        """Strict mode turns an unresolvable source conflict into an abstention.

        Outside strict mode both positions are surfaced with their dates and authority, which is
        usually more useful. A tenant in a regulated domain would rather be told nothing than be
        told two things.
        """
        return self.strict_mode


class Budget(BaseModel):
    """What this request has left to spend, in time, money, and tokens.

    Frozen, and spending returns a new instance. The graph threads state immutably per step, so
    a mutable budget would let a node's overspend appear retroactively in an already-recorded
    state and break replay.

    ``degradation_level`` is carried here rather than alongside because every consumer that
    cares about the budget also needs to know how degraded the request already is. A reranker
    deciding whether to run wants both numbers or neither.
    """

    model_config = ConfigDict(frozen=True)

    wall_ms_total: int = Field(gt=0)
    wall_ms_remaining: int = Field(ge=0)
    usd_total: float = Field(ge=0.0)
    usd_spent: float = Field(default=0.0, ge=0.0)
    max_tokens_in: int = Field(gt=0)
    max_tokens_out: int = Field(gt=0)

    #: 0 is undegraded; 6 is abstention. See the degradation ladder in :mod:`prag.core.budget`.
    degradation_level: int = Field(default=0, ge=0, le=6)

    @property
    def usd_remaining(self) -> float:
        return max(0.0, self.usd_total - self.usd_spent)

    @property
    def wall_exhausted(self) -> bool:
        return self.wall_ms_remaining <= 0

    @property
    def cost_exhausted(self) -> bool:
        return self.usd_remaining <= 0.0

    @property
    def exhausted(self) -> bool:
        return self.wall_exhausted or self.cost_exhausted

    def deadline_for(self, node_id: str, share: float) -> Deadline:
        """A deadline for one node, as a share of the time this request has left.

        Derived from what remains rather than from a fixed per-node value. Fixed timeouts are
        the standard way to build a cascading timeout: each node is individually within its
        limit while their sum exceeds what the client will wait for.
        """
        if not 0.0 < share <= 1.0:
            raise ValueError(f"share must be in (0, 1], got {share}")
        return Deadline.in_ms(self.wall_ms_remaining * share, label=node_id)

    def spend(
        self,
        *,
        wall_ms: int = 0,
        usd: float = 0.0,
    ) -> Budget:
        """Record consumption and return the resulting budget.

        Clamped at zero rather than allowed to go negative. An overrun is real and must be
        visible, but it is visible as ``exhausted``; a negative remaining budget would make
        every downstream ``share`` computation produce nonsense.
        """
        if wall_ms < 0 or usd < 0.0:
            raise ValueError("spend amounts must be non-negative")
        return self.model_copy(
            update={
                "wall_ms_remaining": max(0, self.wall_ms_remaining - wall_ms),
                "usd_spent": self.usd_spent + usd,
            }
        )

    def degraded_to(self, level: int) -> Budget:
        """Move to a degradation level.

        Monotonic: a request never un-degrades. Recovering because one fast node clawed back
        time would produce a request whose behaviour oscillates, and an evaluation set that
        cannot explain its own variance.
        """
        if not 0 <= level <= 6:
            raise ValueError(f"degradation level must be 0..6, got {level}")
        return self.model_copy(update={"degradation_level": max(self.degradation_level, level)})
