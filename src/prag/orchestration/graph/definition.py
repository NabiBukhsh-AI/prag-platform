"""Graph definitions: the shape of a request path, as data.

A definition is serializable and executes nothing. That separation is what makes the request
path testable: a graph can be loaded, validated, diffed against another version, and asserted
on without a database, a model, or a network.

Conditions are *named*, not inlined as lambdas. A graph carrying Python callables cannot be
written to a file, cannot be compared across versions, and cannot be reviewed by anyone who is
not reading the code that built it. The engine resolves names against a registry at construction
time, so an unknown condition fails at load rather than on the one request that takes that edge.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prag.core.errors import ConfigurationError

__all__ = ["Edge", "GraphDefinition"]


class Edge(BaseModel):
    """A directed transition between two nodes.

    ``when`` names a condition in the engine's registry. Edges out of a node are evaluated in
    declaration order and the first matching one is taken, so an unconditional edge acts as a
    default and belongs last. That ordering is explicit rather than alphabetical because the
    order *is* the policy.
    """

    model_config = ConfigDict(frozen=True)

    from_node: str
    to_node: str
    when: str | None = None
    #: ``normal`` edges are taken by condition evaluation on success. ``fallback`` edges are
    #: taken by the engine when a node fails, and are never evaluated as conditions.
    #:
    #: Declaring the error path here rather than only on the node implementation is what keeps
    #: the graph complete as data: a definition whose failure routing is invisible cannot be
    #: diffed, reviewed, or reasoned about without reading every node class.
    kind: Literal["normal", "fallback"] = "normal"
    #: Why this edge exists. Rendered into graph documentation and into the trace, so a path
    #: taken during an incident explains itself without a reader reconstructing the intent.
    reason: str | None = None

    @property
    def is_default(self) -> bool:
        return self.when is None


class GraphDefinition(BaseModel):
    """One route class, expressed as nodes and edges.

    Cyclic by design. The request path genuinely needs three cycles — clarification,
    re-retrieval on insufficient evidence, and regeneration on grounding failure — and a DAG
    cannot express any of them. ``max_steps`` is what keeps a cycle from being unbounded, and it
    is a property of the graph rather than a global constant because a multi-hop graph
    legitimately takes more steps than a simple lookup.
    """

    model_config = ConfigDict(frozen=True)

    graph_id: str
    entry_node: str
    node_ids: tuple[str, ...]
    edges: tuple[Edge, ...] = ()
    #: Nodes after which execution stops. More than one is normal: a graph terminates at an
    #: answer, at an abstention, or at a clarification request, and those are different ends.
    terminal_nodes: frozenset[str] = frozenset()
    #: Where the engine jumps when the budget ladder reaches its last rung. Part of the graph's
    #: shape rather than an engine setting, because whether a route class has an abstention path
    #: is a property of that route. Optional: an ingestion graph legitimately has none.
    abstain_node: str | None = None
    #: Hard cap on node executions, so a cycle cannot run forever. Reaching it is a bug in the
    #: graph rather than an expected outcome, and the engine raises rather than answering.
    max_steps: int = Field(default=32, gt=0)

    @model_validator(mode="after")
    def _structurally_valid(self) -> GraphDefinition:
        """Reject a malformed graph at load time rather than mid-request.

        Every check here would otherwise surface as an exception on whichever request first
        took the broken edge, which may be days after the change that broke it.
        """
        known = set(self.node_ids)
        if not known:
            raise ValueError(f"graph {self.graph_id!r} declares no nodes")
        if self.entry_node not in known:
            raise ValueError(f"entry node {self.entry_node!r} is not among the declared nodes")

        if self.abstain_node is not None and self.abstain_node not in known:
            raise ValueError(f"abstain node {self.abstain_node!r} is not among the declared nodes")

        unknown_terminals = self.terminal_nodes - known
        if unknown_terminals:
            raise ValueError(f"terminal nodes not declared: {sorted(unknown_terminals)}")
        if not self.terminal_nodes:
            raise ValueError(f"graph {self.graph_id!r} has no terminal node and could never finish")

        for edge in self.edges:
            if edge.from_node not in known:
                raise ValueError(f"edge from unknown node {edge.from_node!r}")
            if edge.to_node not in known:
                raise ValueError(f"edge to unknown node {edge.to_node!r}")

        # A non-terminal node with no outgoing edge is a dead end: execution would arrive and
        # have nowhere to go. Catching it here beats discovering it as a stuck request.
        dead_ends = sorted(
            node
            for node in known
            if node not in self.terminal_nodes
            and not any(e.from_node == node and e.kind == "normal" for e in self.edges)
        )
        if dead_ends:
            raise ValueError(f"non-terminal nodes with no outgoing edge: {dead_ends}")

        unreachable = sorted(known - self._reachable())
        if unreachable:
            raise ValueError(f"nodes unreachable from the entry node: {unreachable}")

        return self

    def _reachable(self) -> set[str]:
        """Nodes the engine can actually arrive at.

        Follows fallback edges as well as normal ones, and seeds the frontier with the abstain
        node, because both are real ways execution reaches a node. Counting only normal edges
        would flag every error-handling node as unreachable.
        """
        seeds = [self.entry_node] + ([self.abstain_node] if self.abstain_node else [])
        reached = set(seeds)
        frontier = list(seeds)
        while frontier:
            current = frontier.pop()
            for edge in self.edges:
                if edge.from_node == current and edge.to_node not in reached:
                    reached.add(edge.to_node)
                    frontier.append(edge.to_node)
        return reached

    def edges_from(self, node_id: str) -> tuple[Edge, ...]:
        """Normal outgoing edges, in declaration order.

        Order matters: the first matching condition wins, so a default edge placed first would
        shadow every conditional edge after it. Fallback edges are excluded because they are
        taken on failure, not chosen on success.
        """
        return tuple(e for e in self.edges if e.from_node == node_id and e.kind == "normal")

    def fallback_from(self, node_id: str) -> str | None:
        """The declared fallback target for a node, if it has one."""
        return next(
            (e.to_node for e in self.edges if e.from_node == node_id and e.kind == "fallback"),
            None,
        )

    def condition_names(self) -> frozenset[str]:
        """Every condition this graph references, for validation against the registry."""
        return frozenset(e.when for e in self.edges if e.when is not None and e.kind == "normal")

    def is_terminal(self, node_id: str) -> bool:
        return node_id in self.terminal_nodes

    def require_known(self, node_id: str) -> None:
        if node_id not in self.node_ids:
            raise ConfigurationError("unknown node", graph_id=self.graph_id, node_id=node_id)
