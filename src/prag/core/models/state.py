"""The request state threaded through the graph.

Nodes declare what they read and what they write; the interpreter enforces those declarations.
That is what makes the graph statically analysable — a node that reads ``evidence`` cannot be
scheduled before the node that writes it, and the check happens at graph-load time rather than
as a ``None`` dereference in production.

The state is replaced per step rather than mutated. A recorded state can therefore be replayed
against pinned versions and asserted to produce identical decisions, which is the regression
mechanism the whole evaluation strategy rests on. Mutation would make every recorded state a
snapshot of the end rather than of the step.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from prag.core.models.context import ContextBundle, ContextValidation
from prag.core.models.events import DomainEvent
from prag.core.models.fusion import KnowledgeDecision, ParametricSignal
from prag.core.models.generation import AnswerEnvelope, ModelSpec
from prag.core.models.guardrails import GuardrailVerdict
from prag.core.models.identity import Budget, Principal, TenantPolicy
from prag.core.models.query import (
    QueryAnalysis,
    QueryVariants,
    SessionContext,
    StrategyDecision,
)
from prag.core.models.retrieval import CandidatePool, EvidenceGroup, RetrievalPlan

__all__ = ["NodeResult", "NodeStatus", "RequestState"]

from enum import StrEnum


class NodeStatus(StrEnum):
    """How one node's execution ended."""

    OK = "ok"
    #: The node ran its fallback and produced a usable, degraded result.
    FELL_BACK = "fell_back"
    #: The node was skipped, typically by the degradation ladder. Not a failure.
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


class RequestState(BaseModel):
    """Everything known about one in-flight request.

    Optional fields are ``None`` until the node that produces them has run. That is not
    sloppiness: the graph has branches, and a request routed to the parametric path legitimately
    never populates ``pool`` or ``evidence``. Making them required would force every branch to
    invent empty values, and an empty ``CandidatePool`` is indistinguishable from a retrieval
    that found nothing.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    trace_id: str
    principal: Principal
    policy: TenantPolicy
    budget: Budget

    raw_query: str
    session: SessionContext | None = None

    analysis: QueryAnalysis | None = None
    strategy: StrategyDecision | None = None
    variants: QueryVariants | None = None
    plan: RetrievalPlan | None = None
    pool: CandidatePool | None = None
    evidence: tuple[EvidenceGroup, ...] = ()
    parametric: ParametricSignal | None = None
    bundle: ContextBundle | None = None
    validation: ContextValidation | None = None
    decision: KnowledgeDecision | None = None
    spec: ModelSpec | None = None
    result: AnswerEnvelope | None = None

    guardrail_verdicts: tuple[GuardrailVerdict, ...] = ()
    node_timings: dict[str, int] = Field(default_factory=dict)
    #: Drained to the event bus after the response is sent, never during.
    events: tuple[DomainEvent, ...] = ()

    def advanced(self, **updates: object) -> RequestState:
        """The next state, with ``updates`` applied.

        The only way a node should produce state. Returning a new instance keeps every
        intermediate step recordable, which is what replay needs.
        """
        return self.model_copy(update=updates)

    def with_event(self, event: DomainEvent) -> RequestState:
        return self.model_copy(update={"events": (*self.events, event)})

    def with_verdict(self, verdict: GuardrailVerdict) -> RequestState:
        return self.model_copy(update={"guardrail_verdicts": (*self.guardrail_verdicts, verdict)})

    def with_timing(self, node_id: str, elapsed_ms: int) -> RequestState:
        return self.model_copy(update={"node_timings": {**self.node_timings, node_id: elapsed_ms}})

    @property
    def blocked(self) -> bool:
        return any(v.blocked for v in self.guardrail_verdicts)

    @property
    def has_usable_evidence(self) -> bool:
        return bool(self.evidence)

    @property
    def total_elapsed_ms(self) -> int:
        """Sum of recorded node timings.

        Less than the request's wall time, and deliberately so: the gap between this and the
        measured total is the orchestration overhead, and that gap is worth being able to see.
        """
        return sum(self.node_timings.values())


class NodeResult(BaseModel):
    """What one node returns to the interpreter.

    The node hands back the state it produced plus how it went. Separating the two means the
    interpreter can enforce the budget, record the timing, and route to a fallback without the
    node needing to know that any of that happens.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    node_id: str
    status: NodeStatus
    state: RequestState
    elapsed_ms: int = Field(default=0, ge=0)
    usd_spent: float = Field(default=0.0, ge=0.0)
    #: Where to go next, when the node overrides the graph's declared edge. Used by the three
    #: cycles the graph genuinely needs: clarification, re-retrieval, and regeneration.
    next_node: str | None = None
    error_reason_code: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in (NodeStatus.OK, NodeStatus.FELL_BACK, NodeStatus.SKIPPED)
