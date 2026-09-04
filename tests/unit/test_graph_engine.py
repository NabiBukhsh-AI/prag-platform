"""The graph engine, exercised against toy graphs.

Toy nodes rather than real ones, deliberately. The build order puts the engine and its full
test suite before any real node exists, so that a failure here is unambiguously the engine's.
Testing an interpreter through the subsystems it interprets makes every failure two hypotheses.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from prag.core.budget import BudgetController, DegradationLevel
from prag.core.errors import AbstentionRequired, ConfigurationError, DeadlineExceeded, PragError
from prag.core.models.common import SlaTier, VolatilityClass
from prag.core.models.identity import Budget, Principal, TenantPolicy, UtilityWeights
from prag.core.models.query import (
    Ambiguity,
    BudgetClass,
    FieldPrediction,
    QueryAnalysis,
    QueryQuality,
    QueryStructure,
    SafetyPreflags,
    Temporality,
)
from prag.core.models.state import NodeResult, NodeStatus, RequestState
from prag.orchestration import Edge, GraphDefinition, GraphEngine

# ---------------------------------------------------------------------------
# Toy nodes
# ---------------------------------------------------------------------------


class ToyNode:
    """A node that writes one declared field, with configurable behaviour."""

    def __init__(
        self,
        node_id: str,
        *,
        writes: frozenset[str] = frozenset(),
        reads: frozenset[str] = frozenset(),
        timeout_ms: int = 1000,
        fallback_node: str | None = None,
        status: NodeStatus = NodeStatus.OK,
        value: Any = None,
        next_node: str | None = None,
        sleep_s: float = 0.0,
        raises: Exception | None = None,
        usd: float = 0.0,
        writes_undeclared: str | None = None,
    ) -> None:
        self.node_id = node_id
        self.reads = reads
        self.writes = writes
        self.timeout_ms = timeout_ms
        self.fallback_node = fallback_node
        self._status = status
        self._value = value
        self._next = next_node
        self._sleep_s = sleep_s
        self._raises = raises
        self._usd = usd
        self._writes_undeclared = writes_undeclared
        self.call_count = 0

    async def run(self, state: RequestState) -> NodeResult:
        self.call_count += 1
        if self._sleep_s:
            await asyncio.sleep(self._sleep_s)
        if self._raises is not None:
            raise self._raises

        updates: dict[str, Any] = {}
        for field in self.writes:
            updates[field] = self._value if self._value is not None else f"{self.node_id}-out"
        if self._writes_undeclared:
            updates[self._writes_undeclared] = "sneaky"

        return NodeResult(
            node_id=self.node_id,
            status=self._status,
            state=state.advanced(**updates) if updates else state,
            next_node=self._next,
            usd_spent=self._usd,
        )


def an_analysis(request_id: str = "req-1") -> QueryAnalysis:
    return QueryAnalysis(
        request_id=request_id,
        raw_query="q",
        normalized_query="q",
        language="en",
        intent=FieldPrediction(value="lookup", confidence=0.9),
        domain=FieldPrediction(value="ops", confidence=0.9),
        complexity=FieldPrediction(value="simple_factual", confidence=0.9),
        knowledge_requirements={},
        temporality=Temporality(
            volatility_class=VolatilityClass.SLOW,
            estimated_half_life_days=180.0,
            confidence=0.8,
        ),
        structure=QueryStructure(multi_hop=FieldPrediction(value=False, confidence=0.9)),
        ambiguity=Ambiguity(),
        query_quality=QueryQuality(needs_rewrite=False, score=0.9),
        budget_class=BudgetClass(latency_tier=SlaTier.STANDARD, cost_tier=SlaTier.STANDARD),
        safety_preflags=SafetyPreflags(),
        router_uncertainty=0.1,
        classifier_tier_used="T1",
        analysis_latency_ms=17,
    )


@pytest.fixture
def state() -> RequestState:
    return RequestState(
        request_id="req-1",
        trace_id="t" * 32,
        principal=Principal(tenant_id="tenant-a", user_id="u1"),
        policy=TenantPolicy(
            tenant_id="tenant-a",
            config_version="c1",
            utility_weights=UtilityWeights(quality=0.6, latency=0.2, cost=0.2),
        ),
        budget=Budget(
            wall_ms_total=5000,
            wall_ms_remaining=5000,
            usd_total=0.10,
            max_tokens_in=8000,
            max_tokens_out=1024,
        ),
        raw_query="what is the escalation policy",
    )


def two_node_graph(**overrides: Any) -> GraphDefinition:
    base: dict[str, Any] = {
        "graph_id": "toy",
        "entry_node": "analyze",
        "node_ids": ("analyze", "answer"),
        "edges": (Edge(from_node="analyze", to_node="answer"),),
        "terminal_nodes": frozenset({"answer"}),
    }
    return GraphDefinition(**{**base, **overrides})


# ---------------------------------------------------------------------------
# Definition validation
# ---------------------------------------------------------------------------


class TestGraphDefinition:
    """Every check here would otherwise surface on the first request to take a broken edge."""

    def test_valid_graph_loads(self) -> None:
        assert two_node_graph().graph_id == "toy"

    def test_entry_node_must_exist(self) -> None:
        with pytest.raises(ValidationError, match="entry node"):
            two_node_graph(entry_node="missing")

    def test_edge_endpoints_must_exist(self) -> None:
        with pytest.raises(ValidationError, match="edge to unknown node"):
            two_node_graph(edges=(Edge(from_node="analyze", to_node="nowhere"),))

    def test_a_graph_needs_a_terminal_node(self) -> None:
        """Otherwise it could never finish."""
        with pytest.raises(ValidationError, match="no terminal node"):
            two_node_graph(terminal_nodes=frozenset())

    def test_dead_ends_are_rejected(self) -> None:
        """A non-terminal node with no outgoing edge is a request that arrives and sticks."""
        with pytest.raises(ValidationError, match="no outgoing edge"):
            GraphDefinition(
                graph_id="toy",
                entry_node="a",
                node_ids=("a", "b", "c"),
                edges=(Edge(from_node="a", to_node="b"), Edge(from_node="a", to_node="c")),
                terminal_nodes=frozenset({"b"}),
            )

    def test_unreachable_nodes_are_rejected(self) -> None:
        """A node nothing routes to is dead code that looks like coverage."""
        with pytest.raises(ValidationError, match="unreachable"):
            GraphDefinition(
                graph_id="toy",
                entry_node="a",
                node_ids=("a", "b", "orphan"),
                edges=(Edge(from_node="a", to_node="b"),),
                terminal_nodes=frozenset({"b", "orphan"}),
            )

    def test_edges_keep_declaration_order(self) -> None:
        """First matching condition wins, so order is the policy."""
        definition = GraphDefinition(
            graph_id="toy",
            entry_node="a",
            node_ids=("a", "b", "c"),
            edges=(
                Edge(from_node="a", to_node="b", when="is_special"),
                Edge(from_node="a", to_node="c"),
            ),
            terminal_nodes=frozenset({"b", "c"}),
        )
        assert [e.to_node for e in definition.edges_from("a")] == ["b", "c"]
        assert definition.condition_names() == frozenset({"is_special"})

    def test_definition_is_serializable(self) -> None:
        """It is data, so it can be written to a file and diffed across versions."""
        original = two_node_graph()
        assert GraphDefinition.model_validate(original.model_dump()) == original


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------


class TestEngineConstruction:
    def test_missing_node_implementation_fails_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="no implementation"):
            GraphEngine(two_node_graph(), {"analyze": ToyNode("analyze")})

    def test_unregistered_condition_fails_at_construction(self) -> None:
        """Not on the one request that takes that edge."""
        definition = two_node_graph(
            edges=(Edge(from_node="analyze", to_node="answer", when="unknown_condition"),)
        )
        with pytest.raises(ConfigurationError, match="unregistered edge conditions"):
            GraphEngine(definition, {"analyze": ToyNode("analyze"), "answer": ToyNode("answer")})

    def test_node_registered_under_a_mismatched_id_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="different id"):
            GraphEngine(
                two_node_graph(),
                {"analyze": ToyNode("something-else"), "answer": ToyNode("answer")},
            )

    def test_fallback_must_name_a_known_node(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown node"):
            GraphEngine(
                two_node_graph(),
                {
                    "analyze": ToyNode("analyze", fallback_node="ghost"),
                    "answer": ToyNode("answer"),
                },
            )

    def test_node_fallback_must_match_the_declared_edge(self) -> None:
        """The serialized graph and the running graph must agree about where failures go.

        A fallback living only on the node implementation is invisible in the definition, which
        is exactly the divergence that makes a reviewed graph and a deployed graph two things.
        """
        definition = GraphDefinition(
            graph_id="mismatched",
            entry_node="a",
            node_ids=("a", "b", "c"),
            edges=(
                Edge(from_node="a", to_node="b"),
                Edge(from_node="a", to_node="c", kind="fallback"),
                Edge(from_node="c", to_node="b"),
            ),
            terminal_nodes=frozenset({"b"}),
        )
        with pytest.raises(ConfigurationError, match="disagrees"):
            GraphEngine(
                definition,
                {
                    "a": ToyNode("a", fallback_node="b"),
                    "b": ToyNode("b"),
                    "c": ToyNode("c"),
                },
            )

    def test_declared_fallback_edge_needs_a_node_that_uses_it(self) -> None:
        """A fallback edge nothing takes is a failure path that only looks like it exists."""
        definition = GraphDefinition(
            graph_id="orphan-fallback",
            entry_node="a",
            node_ids=("a", "b", "c"),
            edges=(
                Edge(from_node="a", to_node="b"),
                Edge(from_node="a", to_node="c", kind="fallback"),
                Edge(from_node="c", to_node="b"),
            ),
            terminal_nodes=frozenset({"b"}),
        )
        with pytest.raises(ConfigurationError, match="does not use"):
            GraphEngine(
                definition,
                {"a": ToyNode("a"), "b": ToyNode("b"), "c": ToyNode("c")},
            )


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


class TestTraversal:
    async def test_runs_the_two_node_graph(self, state: RequestState) -> None:
        engine = GraphEngine(
            two_node_graph(),
            {
                "analyze": ToyNode("analyze", writes=frozenset({"analysis"}), value=an_analysis()),
                "answer": ToyNode("answer"),
            },
        )
        run = await engine.run(state)

        assert run.completed
        assert run.terminal_node == "answer"
        assert run.path == ("analyze", "answer")
        assert run.state.analysis is not None

    async def test_records_a_step_per_node(self, state: RequestState) -> None:
        """The trace is the product; a request that cannot be reconstructed cannot be debugged."""
        engine = GraphEngine(
            two_node_graph(), {"analyze": ToyNode("analyze"), "answer": ToyNode("answer")}
        )
        run = await engine.run(state)

        assert len(run.steps) == 2
        assert all(s.status is NodeStatus.OK for s in run.steps)
        assert set(run.state.node_timings) == {"analyze", "answer"}

    async def test_conditional_edges_take_the_first_match(self, state: RequestState) -> None:
        definition = GraphDefinition(
            graph_id="branching",
            entry_node="start",
            node_ids=("start", "special", "ordinary"),
            edges=(
                Edge(from_node="start", to_node="special", when="is_special"),
                Edge(from_node="start", to_node="ordinary"),
            ),
            terminal_nodes=frozenset({"special", "ordinary"}),
        )
        nodes = {
            "start": ToyNode("start", writes=frozenset({"raw_query"}), value="special please"),
            "special": ToyNode("special"),
            "ordinary": ToyNode("ordinary"),
        }
        engine = GraphEngine(
            definition,
            nodes,
            conditions={"is_special": lambda s: "special" in s.raw_query},
        )
        run = await engine.run(state)
        assert run.terminal_node == "special"

    async def test_default_edge_is_taken_when_no_condition_matches(
        self, state: RequestState
    ) -> None:
        definition = GraphDefinition(
            graph_id="branching",
            entry_node="start",
            node_ids=("start", "special", "ordinary"),
            edges=(
                Edge(from_node="start", to_node="special", when="is_special"),
                Edge(from_node="start", to_node="ordinary"),
            ),
            terminal_nodes=frozenset({"special", "ordinary"}),
        )
        engine = GraphEngine(
            definition,
            {
                "start": ToyNode("start"),
                "special": ToyNode("special"),
                "ordinary": ToyNode("ordinary"),
            },
            conditions={"is_special": lambda s: False},
        )
        assert (await engine.run(state)).terminal_node == "ordinary"

    async def test_node_can_override_the_edge(self, state: RequestState) -> None:
        """How the three genuine cycles are expressed.

        Re-retrieval, clarification, and regeneration all send execution backwards; the graph's
        edges describe the forward path.
        """
        definition = GraphDefinition(
            graph_id="cyclic",
            entry_node="retrieve",
            node_ids=("retrieve", "validate", "answer"),
            edges=(
                Edge(from_node="retrieve", to_node="validate"),
                Edge(from_node="validate", to_node="answer"),
            ),
            terminal_nodes=frozenset({"answer"}),
        )
        retrieve = ToyNode("retrieve")
        # Sends execution back to retrieve exactly once, the way a coverage failure would.
        validate = _RetryOnce("validate", back_to="retrieve")
        engine = GraphEngine(
            definition,
            {"retrieve": retrieve, "validate": validate, "answer": ToyNode("answer")},
        )
        run = await engine.run(state)

        assert run.completed
        assert run.path == ("retrieve", "validate", "retrieve", "validate", "answer")
        assert retrieve.call_count == 2

    async def test_step_cap_stops_a_runaway_cycle(self, state: RequestState) -> None:
        """Reaching it is a bug in the graph, so the engine raises rather than answering."""
        definition = GraphDefinition(
            graph_id="looping",
            entry_node="a",
            node_ids=("a", "b", "done"),
            edges=(Edge(from_node="a", to_node="b"), Edge(from_node="b", to_node="done")),
            terminal_nodes=frozenset({"done"}),
            max_steps=6,
        )
        engine = GraphEngine(
            definition,
            {
                "a": ToyNode("a"),
                # Always overrides the edge to "done", so the terminal is never reached.
                "b": ToyNode("b", next_node="a"),
                "done": ToyNode("done"),
            },
        )
        with pytest.raises(ConfigurationError, match="step cap"):
            await engine.run(state)


class _RetryOnce:
    """Sends execution backwards on its first call only."""

    def __init__(self, node_id: str, *, back_to: str) -> None:
        self.node_id = node_id
        self.reads = frozenset()
        self.writes = frozenset()
        self.timeout_ms = 1000
        self.fallback_node = None
        self._back_to = back_to
        self.call_count = 0

    async def run(self, state: RequestState) -> NodeResult:
        self.call_count += 1
        return NodeResult(
            node_id=self.node_id,
            status=NodeStatus.OK,
            state=state,
            next_node=self._back_to if self.call_count == 1 else None,
        )


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


class TestDeclarations:
    async def test_undeclared_write_is_rejected(self, state: RequestState) -> None:
        """Undeclared writes make the graph's static ordering a lie."""
        engine = GraphEngine(
            two_node_graph(),
            {
                "analyze": ToyNode("analyze", writes_undeclared="raw_query"),
                "answer": ToyNode("answer"),
            },
        )
        with pytest.raises(ConfigurationError, match="did not declare"):
            await engine.run(state)

    async def test_declared_write_is_allowed(self, state: RequestState) -> None:
        engine = GraphEngine(
            two_node_graph(),
            {
                "analyze": ToyNode("analyze", writes=frozenset({"raw_query"}), value="rewritten"),
                "answer": ToyNode("answer"),
            },
        )
        assert (await engine.run(state)).state.raw_query == "rewritten"

    async def test_node_scheduled_before_its_inputs_is_rejected(self, state: RequestState) -> None:
        """Reported at the graph layer rather than as a None dereference inside the node."""
        engine = GraphEngine(
            two_node_graph(),
            {
                "analyze": ToyNode("analyze", reads=frozenset({"decision"})),
                "answer": ToyNode("answer"),
            },
        )
        with pytest.raises(ConfigurationError, match="before its inputs"):
            await engine.run(state)

    async def test_engine_owned_fields_need_no_declaration(self, state: RequestState) -> None:
        """Budget and timings are the engine's bookkeeping, not the node's output."""
        engine = GraphEngine(
            two_node_graph(),
            {"analyze": ToyNode("analyze", usd=0.002), "answer": ToyNode("answer")},
        )
        run = await engine.run(state)
        assert run.state.budget.usd_spent == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# Failure and degradation
# ---------------------------------------------------------------------------


class TestFailureHandling:
    async def test_fallback_is_taken_on_failure(self, state: RequestState) -> None:
        definition = GraphDefinition(
            graph_id="with-fallback",
            entry_node="primary",
            node_ids=("primary", "backup", "answer"),
            edges=(
                Edge(from_node="primary", to_node="answer"),
                Edge(from_node="primary", to_node="backup", kind="fallback"),
                Edge(from_node="backup", to_node="answer"),
            ),
            terminal_nodes=frozenset({"answer"}),
        )
        engine = GraphEngine(
            definition,
            {
                "primary": ToyNode("primary", status=NodeStatus.FAILED, fallback_node="backup"),
                "backup": ToyNode("backup"),
                "answer": ToyNode("answer"),
            },
        )
        run = await engine.run(state)

        assert run.completed
        assert run.path == ("primary", "backup", "answer")
        assert run.steps[1].fell_back_from == "primary"

    async def test_failure_without_a_fallback_propagates(self, state: RequestState) -> None:
        """No fallback means this node's absence makes the request meaningless."""
        engine = GraphEngine(
            two_node_graph(),
            {
                "analyze": ToyNode("analyze", status=NodeStatus.FAILED),
                "answer": ToyNode("answer"),
            },
        )
        with pytest.raises(PragError, match="no fallback"):
            await engine.run(state)

    async def test_timeout_becomes_a_typed_failure(self, state: RequestState) -> None:
        engine = GraphEngine(
            two_node_graph(),
            {
                "analyze": ToyNode("analyze", timeout_ms=10, sleep_s=0.5),
                "answer": ToyNode("answer"),
            },
        )
        with pytest.raises(DeadlineExceeded):
            await engine.run(state)

    async def test_timeout_routes_to_the_fallback(self, state: RequestState) -> None:
        definition = GraphDefinition(
            graph_id="slow",
            entry_node="slow",
            node_ids=("slow", "fast", "answer"),
            edges=(
                Edge(from_node="slow", to_node="answer"),
                Edge(from_node="slow", to_node="fast", kind="fallback"),
                Edge(from_node="fast", to_node="answer"),
            ),
            terminal_nodes=frozenset({"answer"}),
        )
        engine = GraphEngine(
            definition,
            {
                "slow": ToyNode("slow", timeout_ms=10, sleep_s=0.5, fallback_node="fast"),
                "fast": ToyNode("fast"),
                "answer": ToyNode("answer"),
            },
        )
        run = await engine.run(state)

        assert run.completed
        assert run.steps[0].status is NodeStatus.TIMED_OUT
        assert run.steps[0].error_reason_code == "deadline_exceeded"

    async def test_typed_errors_become_node_failures(self, state: RequestState) -> None:
        from prag.core.errors import SourceUnavailable

        definition = GraphDefinition(
            graph_id="erroring",
            entry_node="boom",
            node_ids=("boom", "backup", "answer"),
            edges=(
                Edge(from_node="boom", to_node="answer"),
                Edge(from_node="boom", to_node="backup", kind="fallback"),
                Edge(from_node="backup", to_node="answer"),
            ),
            terminal_nodes=frozenset({"answer"}),
        )
        engine = GraphEngine(
            definition,
            {
                "boom": ToyNode("boom", raises=SourceUnavailable("down"), fallback_node="backup"),
                "backup": ToyNode("backup"),
                "answer": ToyNode("answer"),
            },
        )
        run = await engine.run(state)
        assert run.steps[0].error_reason_code == "source_unavailable"

    async def test_abstention_is_not_a_node_failure(self, state: RequestState) -> None:
        """Control flow, not failure.

        Catching it as a failure would send a deliberate abstention down the fallback path.
        """
        engine = GraphEngine(
            two_node_graph(
                edges=(
                    Edge(from_node="analyze", to_node="answer"),
                    Edge(from_node="analyze", to_node="answer", kind="fallback"),
                )
            ),
            {
                "analyze": ToyNode(
                    "analyze",
                    raises=AbstentionRequired("no evidence", abstention_code="x"),
                    fallback_node="answer",
                ),
                "answer": ToyNode("answer"),
            },
        )
        with pytest.raises(AbstentionRequired):
            await engine.run(state)


class TestBudgetIntegration:
    def _tight(self, state: RequestState, remaining: int) -> RequestState:
        return state.advanced(
            budget=state.budget.model_copy(update={"wall_ms_remaining": remaining})
        )

    async def test_degradation_level_is_recorded_per_step(self, state: RequestState) -> None:
        """Budget-driven quality loss must be visible, or it looks like a regression."""
        engine = GraphEngine(
            two_node_graph(), {"analyze": ToyNode("analyze"), "answer": ToyNode("answer")}
        )
        run = await engine.run(self._tight(state, 1000))

        assert run.steps[0].degradation_level >= int(DegradationLevel.SKIP_RERANK)

    async def test_exhausted_budget_routes_to_the_abstain_node(self, state: RequestState) -> None:
        definition = GraphDefinition(
            graph_id="with-abstain",
            entry_node="analyze",
            node_ids=("analyze", "answer", "abstain"),
            edges=(Edge(from_node="analyze", to_node="answer"),),
            terminal_nodes=frozenset({"answer", "abstain"}),
            abstain_node="abstain",
        )
        engine = GraphEngine(
            definition,
            {
                "analyze": ToyNode("analyze"),
                "answer": ToyNode("answer"),
                "abstain": ToyNode("abstain"),
            },
        )
        run = await engine.run(self._tight(state, 0))

        assert run.terminal_node == "abstain"
        assert run.path == ("abstain",), "no work starts that cannot finish"

    async def test_exhausted_budget_without_an_abstain_node_raises(
        self, state: RequestState
    ) -> None:
        engine = GraphEngine(
            two_node_graph(), {"analyze": ToyNode("analyze"), "answer": ToyNode("answer")}
        )
        with pytest.raises(AbstentionRequired) as excinfo:
            await engine.run(self._tight(state, 0))
        assert excinfo.value.abstention_code == "budget_exceeded"

    async def test_node_deadline_is_capped_by_the_remaining_budget(
        self, state: RequestState
    ) -> None:
        """A node cannot be granted more time than the request has left.

        The node declares a 60 second timeout and sleeps for half a second. With 300 ms of
        budget left it still times out, because the engine hands it the smaller of the two.
        Without that cap a single generous node timeout would overrun the whole request.

        It then abstains rather than falling back, and that is correct: the timeout consumed
        what remained, so there is genuinely nothing left to run the fallback with.
        """
        definition = GraphDefinition(
            graph_id="capped",
            entry_node="slow",
            node_ids=("slow", "backup", "answer", "abstain"),
            edges=(
                Edge(from_node="slow", to_node="answer"),
                Edge(from_node="slow", to_node="backup", kind="fallback"),
                Edge(from_node="backup", to_node="answer"),
            ),
            terminal_nodes=frozenset({"answer", "abstain"}),
            abstain_node="abstain",
        )
        engine = GraphEngine(
            definition,
            {
                "slow": ToyNode("slow", timeout_ms=60_000, sleep_s=0.5, fallback_node="backup"),
                "backup": ToyNode("backup"),
                "answer": ToyNode("answer"),
                "abstain": ToyNode("abstain"),
            },
            budget_controller=BudgetController(),
        )
        run = await engine.run(self._tight(state, 300))

        assert run.steps[0].status is NodeStatus.TIMED_OUT, "the budget capped the 60s timeout"
        assert run.steps[0].elapsed_ms < 500, "it did not wait the full half second"
        assert run.terminal_node == "abstain"

    async def test_abstain_node_runs_on_an_exhausted_budget(self, state: RequestState) -> None:
        """The abstention path is exempt from the gate that sent execution to it.

        Re-gating it would make the abstention path unreachable exactly when it is needed, and
        explaining why the system is declining is the one piece of work still worth doing.
        """
        definition = GraphDefinition(
            graph_id="abstaining",
            entry_node="work",
            node_ids=("work", "answer", "abstain"),
            edges=(Edge(from_node="work", to_node="answer"),),
            terminal_nodes=frozenset({"answer", "abstain"}),
            abstain_node="abstain",
        )
        abstain = ToyNode("abstain")
        engine = GraphEngine(
            definition,
            {"work": ToyNode("work"), "answer": ToyNode("answer"), "abstain": abstain},
        )
        run = await engine.run(self._tight(state, 0))

        assert abstain.call_count == 1, "the abstention node must actually run"
        assert run.completed

    async def test_budget_is_spent_across_nodes(self, state: RequestState) -> None:
        engine = GraphEngine(
            two_node_graph(),
            {
                "analyze": ToyNode("analyze", usd=0.001),
                "answer": ToyNode("answer", usd=0.002),
            },
        )
        run = await engine.run(state)
        assert run.state.budget.usd_spent == pytest.approx(0.003)
