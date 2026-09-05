"""Generation profiles and model routing.

Profiles live in configuration rather than code, and they are per route class rather than
global. The one setting worth stating outright: **temperature 0 is the default for anything
evidence-grounded.** Nonzero temperature on an extraction task buys nothing and costs
faithfulness — there is no creative upside in reading a policy document accurately.

Routing is a policy over complexity, strategy, conflict state and budget. Escalation is one-way,
capped at one per request, and never permitted after tokens have reached the client: a second
model's answer replacing a partially streamed first one is indistinguishable, from the client's
side, from corruption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from prag.core.budget import DegradationLevel, DegradationPlan
from prag.core.errors import ConfigurationError
from prag.core.models.generation import ModelSpec

if TYPE_CHECKING:
    from prag.core.models.fusion import KnowledgeDecision
    from prag.core.models.identity import Budget, TenantPolicy
    from prag.core.models.parametric import AdapterSet
    from prag.core.models.query import QueryAnalysis

__all__ = ["DEFAULT_PROFILES", "GenerationProfile", "PolicyModelRouter", "profile_for"]


@dataclass(frozen=True, slots=True)
class GenerationProfile:
    """Sampling and output settings for one route class."""

    name: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 800
    repetition_penalty: float = 1.0
    stop: tuple[str, ...] = ()
    structured_output: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

    @property
    def is_deterministic(self) -> bool:
        """Whether this profile produces the same output for the same prompt.

        Recorded-state replay depends on it, and so does any regression assertion about answer
        content. A grounded profile that drifted to a nonzero temperature would make every such
        test flaky in a way that looks like a real regression.
        """
        return self.temperature == 0.0


DEFAULT_PROFILES: dict[str, GenerationProfile] = {
    # Extraction from provided evidence. Temperature 0, because the answer is in the context and
    # sampling can only move away from it.
    "grounded_extraction": GenerationProfile(
        name="grounded_extraction",
        temperature=0.0,
        max_tokens=800,
        stop=("</answer>",),
        structured_output="answer_envelope.v2",
    ),
    # Synthesis and comparison, where some sampling helps the model connect sources.
    "reasoning_synthesis": GenerationProfile(
        name="reasoning_synthesis",
        temperature=0.3,
        max_tokens=2_000,
        structured_output="answer_envelope.v2",
    ),
    # The only profile where sampling is the point. Never used on an evidence-grounded route.
    "creative": GenerationProfile(name="creative", temperature=0.8, max_tokens=1_500),
}


def profile_for(
    name: str, profiles: dict[str, GenerationProfile] | None = None
) -> GenerationProfile:
    resolved = (profiles or DEFAULT_PROFILES).get(name)
    if resolved is None:
        raise ConfigurationError(
            "unknown generation profile",
            profile=name,
            known=sorted(profiles or DEFAULT_PROFILES),
        )
    return resolved


@dataclass(frozen=True, slots=True)
class ModelOption:
    """A model the router may choose, with the facts routing needs about it."""

    model_id: str
    model_version: str
    provider_id: str
    context_window: int
    cost_per_1k_in: float
    cost_per_1k_out: float
    supports_tools: bool = False
    supports_structured_output: bool = True
    #: Rough capability tier. Escalation moves up exactly one step, never more.
    tier: int = 1


class PolicyModelRouter:
    """Selects a model, version, adapter set and profile. Synchronous and pure.

    Pure because a routing decision that cannot be recomputed from its inputs cannot be
    regression tested, and model choice is one of the larger levers on both cost and quality.
    """

    def __init__(
        self,
        options: dict[str, ModelOption],
        *,
        default_model: str,
        reasoning_model: str | None = None,
        profiles: dict[str, GenerationProfile] | None = None,
    ) -> None:
        if default_model not in options:
            raise ConfigurationError(
                "default model is not among the registered options",
                default_model=default_model,
                known=sorted(options),
            )
        self._options = dict(options)
        self._default = default_model
        self._reasoning = reasoning_model
        self._profiles = profiles or DEFAULT_PROFILES

    def select(
        self,
        analysis: QueryAnalysis,
        decision: KnowledgeDecision,
        adapters: AdapterSet,
        budget: Budget,
        policy: TenantPolicy,
    ) -> ModelSpec:
        plan = DegradationPlan(level=DegradationLevel(budget.degradation_level))

        complexity = str(analysis.complexity.value)
        wants_reasoning = complexity != "simple_factual" or bool(decision.conflicts)

        # Escalation is capped by the ladder, not just by the router. At the CHEAPER_MODEL rung
        # the premium option is unavailable regardless of what the query would otherwise justify,
        # which is what stops a degraded request from spending its way out of degradation.
        chosen_id = (
            self._reasoning
            if wants_reasoning and self._reasoning and plan.premium_model_allowed
            else self._default
        )
        option = self._options[chosen_id]

        profile = profile_for(
            "reasoning_synthesis"
            if wants_reasoning and chosen_id != self._default
            else "grounded_extraction",
            self._profiles,
        )

        return ModelSpec(
            model_id=option.model_id,
            model_version=option.model_version,
            provider_id=option.provider_id,
            adapters=adapters.adapters,
            profile=profile.name,
            context_window=option.context_window,
            supports_tools=option.supports_tools,
            supports_structured_output=option.supports_structured_output,
            cost_per_1k_in=option.cost_per_1k_in,
            cost_per_1k_out=option.cost_per_1k_out,
        )

    @property
    def registered(self) -> tuple[str, ...]:
        return tuple(sorted(self._options))


@dataclass
class FallbackChain:
    """An ordered list of providers to try for one profile.

    Advancing the chain is only permitted before the first token has been emitted. Once output
    has reached the client the request must be completed or explicitly failed — a silent switch
    would either duplicate text or replace it, and the client cannot tell which happened.
    """

    profile: str
    providers: tuple[str, ...] = field(default_factory=tuple)

    def next_after(self, provider_id: str) -> str | None:
        try:
            position = self.providers.index(provider_id)
        except ValueError:
            return self.providers[0] if self.providers else None
        following = self.providers[position + 1 :]
        return following[0] if following else None
