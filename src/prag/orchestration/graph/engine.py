"""The graph interpreter.

Roughly the smallest thing that can run a typed state graph correctly, and it is meant to stay
that way. The engine is node-agnostic: any feature only one graph needs belongs in a node, not
here. An engine whose line count grows faster than the node count has become a framework, and
the effort it absorbs is effort that belonged in the domain.

What it does, and why each part is not the node's job:

**Enforces the budget between every node.** Not at the edges. Checking only at the end pays the
full cost and then times out anyway, having produced nothing; checking between nodes means a
request that cannot finish well can still finish usefully.

**Derives each node's deadline from what the request has left.** A node declares a timeout, and
the engine hands it the smaller of that and the remaining budget. Fixed per-node timeouts are
how a cascading timeout is built: every node is individually within its limit while the sum
exceeds what the client will wait for.

**Enforces the reads and writes declarations.** A node that writes a field it did not declare
fails its own test rather than someone else's, and the declarations are what let a graph be
checked statically instead of by running it.

**Records every step.** The trace is the product. A request that cannot be reconstructed from
its recorded steps cannot be debugged, replayed, or regression-tested.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable  # runtime: the Condition alias below is evaluated
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from prag.core.budget import BudgetController, DegradationLevel
from prag.core.errors import (
    AbstentionRequired,
    ConfigurationError,
    DeadlineExceeded,
    PragError,
)
from prag.core.models.state import NodeResult, NodeStatus, RequestState

if TYPE_CHECKING:
    from collections.abc import Mapping

    from prag.core.protocols.crosscutting import GraphNode
    from prag.orchestration.graph.definition import GraphDefinition

__all__ = ["Condition", "GraphEngine", "GraphRun", "StepRecord"]

#: A named predicate over the state, used to decide which edge to take. Synchronous and pure:
#: an edge condition that performs I/O would put an un-budgeted call between every two nodes.
Condition = Callable[[RequestState], bool]


@dataclass(frozen=True, slots=True)
class StepRecord:
    """What happened at one node.

    Kept even for skipped and failed nodes. "The reranker did not run" is a fact worth having in
    a trace, and its absence is indistinguishable from a reranker that ran and found nothing.
    """

    node_id: str
    status: NodeStatus
    elapsed_ms: int
    degradation_level: int
    next_node: str | None
    error_reason_code: str | None = None
    #: Set when this step ran because another node's fallback pointed here.
    fell_back_from: str | None = None


@dataclass(slots=True)
class GraphRun:
    """The outcome of one traversal."""

    state: RequestState
    steps: list[StepRecord] = field(default_factory=list)
    completed: bool = False
    terminal_node: str | None = None

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(s.node_id for s in self.steps)

    @property
    def total_elapsed_ms(self) -> int:
        return sum(s.elapsed_ms for s in self.steps)


class GraphEngine:
    """Executes a graph definition against a request state."""

    def __init__(
        self,
        definition: GraphDefinition,
        nodes: Mapping[str, GraphNode],
        *,
        budget_controller: BudgetController | None = None,
        conditions: Mapping[str, Condition] | None = None,
    ) -> None:
        self._definition = definition
        self._nodes = dict(nodes)
        self._budget = budget_controller or BudgetController()
        self._conditions = dict(conditions or {})
        self._abstain_node = definition.abstain_node
        self._validate()

    def _validate(self) -> None:
        """Fail at construction, not on the request that first hits the gap."""
        missing_nodes = sorted(set(self._definition.node_ids) - set(self._nodes))
        if missing_nodes:
            raise ConfigurationError(
                "graph declares nodes with no implementation",
                graph_id=self._definition.graph_id,
                missing=missing_nodes,
            )

        missing_conditions = sorted(self._definition.condition_names() - set(self._conditions))
        if missing_conditions:
            raise ConfigurationError(
                "graph references unregistered edge conditions",
                graph_id=self._definition.graph_id,
                missing=missing_conditions,
            )

        for node_id, node in self._nodes.items():
            if node.node_id != node_id:
                raise ConfigurationError(
                    "node registered under a different id than it declares",
                    registered_as=node_id,
                    declares=node.node_id,
                )
            declared = self._definition.fallback_from(node_id)
            if node.fallback_node is not None:
                self._definition.require_known(node.fallback_node)
                if declared != node.fallback_node:
                    raise ConfigurationError(
                        "node fallback disagrees with the graph's fallback edge",
                        node_id=node_id,
                        node_says=node.fallback_node,
                        graph_says=declared,
                        hint="declare the failure path as an Edge with kind='fallback'",
                    )
            elif declared is not None:
                raise ConfigurationError(
                    "graph declares a fallback edge the node does not use",
                    node_id=node_id,
                    graph_says=declared,
                )

    async def run(self, state: RequestState) -> GraphRun:
        """Traverse the graph until a terminal node, an abstention, or the step cap."""
        run = GraphRun(state=state)
        current: str | None = self._definition.entry_node
        fell_back_from: str | None = None

        for _ in range(self._definition.max_steps):
            if current is None:
                break

            verdict = self._budget.check(run.state.budget, node_id=current)
            run.state = run.state.advanced(budget=verdict.budget)

            # The ladder's last rung. Divert to the abstention path rather than starting work
            # that cannot finish, which would spend what remains and produce nothing.
            #
            # The abstain node itself is exempt from this check. It runs on an exhausted budget
            # by design — explaining why the system is declining is the one piece of work still
            # worth doing here, and re-gating it would make the abstention path unreachable
            # exactly when it is needed.
            if verdict.plan.must_abstain and current != self._abstain_node:
                if self._abstain_node is None:
                    raise AbstentionRequired(
                        "budget exhausted and this graph has no abstention path",
                        abstention_code="budget_exceeded",
                        suggested_action="retry with a longer deadline or a cheaper SLA tier",
                        node_id=current,
                    )
                current, fell_back_from = self._abstain_node, current
                continue

            result, record = await self._run_node(
                current, run.state, verdict.plan.level, fell_back_from
            )
            run.state = result.state
            run.steps.append(record)
            fell_back_from = None

            if not result.succeeded:
                node = self._nodes[current]
                if node.fallback_node is None:
                    # No fallback means this node's absence makes the request meaningless.
                    # Propagating beats continuing with a state the next node cannot use.
                    raise self._failure_for(result, current)
                current, fell_back_from = node.fallback_node, current
                continue

            if self._definition.is_terminal(current):
                run.completed = True
                run.terminal_node = current
                return run

            # A node may override the declared edge. This is how the three genuine cycles are
            # expressed: re-retrieval, clarification, and regeneration all send execution
            # backwards, and the graph's edges describe the forward path.
            current = result.next_node or self._next_node(current, run.state)

        raise ConfigurationError(
            "graph exceeded its step cap without reaching a terminal node",
            graph_id=self._definition.graph_id,
            max_steps=self._definition.max_steps,
            path=list(run.path),
        )

    async def _run_node(
        self,
        node_id: str,
        state: RequestState,
        level: DegradationLevel,
        fell_back_from: str | None,
    ) -> tuple[NodeResult, StepRecord]:
        node = self._nodes[node_id]
        self._check_reads(node, state)

        # The smaller of what the node asked for and what the request has left.
        deadline = state.budget.deadline_for(node_id, 1.0).narrowed_to(
            node.timeout_ms, label=node_id
        )
        started = time.monotonic()

        try:
            result = await asyncio.wait_for(
                node.run(state), timeout=max(deadline.remaining_ms / 1000.0, 0.001)
            )
        except TimeoutError:
            result = self._timed_out(node_id, state, started)
        except AbstentionRequired:
            # Control flow, not failure. It belongs to the caller of run(), which converts it
            # into an envelope; catching it here would turn a deliberate abstention into a
            # node failure and send it down the fallback path.
            raise
        except PragError as exc:
            result = NodeResult(
                node_id=node_id,
                status=NodeStatus.FAILED,
                state=state,
                elapsed_ms=self._elapsed(started),
                error_reason_code=exc.reason_code,
            )

        if result.succeeded:
            self._check_writes(node, state, result.state)

        elapsed = result.elapsed_ms or self._elapsed(started)
        next_state = result.state.advanced(
            budget=result.state.budget.spend(wall_ms=elapsed, usd=result.usd_spent)
        ).with_timing(node_id, elapsed)

        return (
            result.model_copy(update={"state": next_state}),
            StepRecord(
                node_id=node_id,
                status=result.status,
                elapsed_ms=elapsed,
                degradation_level=int(level),
                next_node=result.next_node,
                error_reason_code=result.error_reason_code,
                fell_back_from=fell_back_from,
            ),
        )

    def _timed_out(self, node_id: str, state: RequestState, started: float) -> NodeResult:
        return NodeResult(
            node_id=node_id,
            status=NodeStatus.TIMED_OUT,
            state=state,
            elapsed_ms=self._elapsed(started),
            error_reason_code="deadline_exceeded",
        )

    @staticmethod
    def _elapsed(started: float) -> int:
        return int((time.monotonic() - started) * 1000)

    def _check_reads(self, node: GraphNode, state: RequestState) -> None:
        """A node must not run before what it reads has been produced.

        Caught here rather than as a ``None`` dereference inside the node, which would report
        the symptom at the wrong layer and blame the node for the graph's ordering.
        """
        missing = sorted(field for field in node.reads if getattr(state, field, None) is None)
        if missing:
            raise ConfigurationError(
                "node scheduled before its inputs were produced",
                node_id=node.node_id,
                missing=missing,
            )

    def _check_writes(self, node: GraphNode, before: RequestState, after: RequestState) -> None:
        """A node must write only what it declared.

        Undeclared writes make the graph's static ordering a lie: a later node would read a
        field nothing claims to produce, and no load-time check could catch it.
        """
        # Bookkeeping the engine itself maintains, which every node is allowed to change.
        engine_owned = {"budget", "node_timings", "events", "guardrail_verdicts"}
        changed = {
            name
            for name in type(before).model_fields
            if name not in engine_owned and getattr(before, name) != getattr(after, name)
        }
        undeclared = sorted(changed - node.writes)
        if undeclared:
            raise ConfigurationError(
                "node wrote fields it did not declare",
                node_id=node.node_id,
                undeclared=undeclared,
                declared=sorted(node.writes),
            )

    def _next_node(self, node_id: str, state: RequestState) -> str | None:
        """The first edge whose condition holds, evaluated in declaration order."""
        for edge in self._definition.edges_from(node_id):
            if edge.is_default or self._conditions[edge.when or ""](state):
                return edge.to_node
        return None

    @staticmethod
    def _failure_for(result: NodeResult, node_id: str) -> PragError:
        if result.status is NodeStatus.TIMED_OUT:
            return DeadlineExceeded("node exceeded its deadline", node_id=node_id)
        return PragError(
            "node failed with no fallback",
            node_id=node_id,
            reason=result.error_reason_code,
        )
