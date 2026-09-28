"""Executing a retrieval plan.

Legs run concurrently and are individually deadlined. Concurrency is the whole reason the plan
budget and the leg timeouts are separate numbers: a leg timeout bounds one slow source, the plan
budget bounds the wall-clock the stage may take, and with parallel legs the second is not the sum
of the first.

**A slow leg never delays a fast one past the plan budget.** When the budget expires, whatever
has returned is used and the rest is abandoned. That is not a degradation to apologise for — it
is the design: an answer built from two of three sources, with a coverage warning, beats a
request that spent its whole budget waiting for a source whose breaker should have been open.

**One source failing is not the request failing.** The partial-results policy decides, and it
consults which legs were marked required. Everything required marked optional means a total
retrieval failure looks healthy; everything marked required means one flaky source takes the
request down with it.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from prag.core.errors import DeadlineExceeded, RetrievalTotalFailure, SourceUnavailable
from prag.core.models.retrieval import (
    CandidatePool,
    LegResult,
    LegStatus,
    PartialResultsPolicy,
)
from prag.retrieval.fusion import fuse_legs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from prag.core.models.identity import Budget, Principal
    from prag.core.models.retrieval import RetrievalLeg, RetrievalPlan
    from prag.core.protocols.retrieval import KnowledgeSource

__all__ = ["CircuitBreaker", "ParallelRetrievalOrchestrator"]


class CircuitBreaker:
    """Per-source failure isolation.

    Not optional at this dependency count. Without one, an unhealthy source turns every request
    into a timeout, and the degradation ladder never fires because nothing reports failure fast
    enough to trigger it — the request just waits.

    A half-open probe is what lets recovery happen without a deploy. Refusing traffic forever
    after a transient outage is its own outage.
    """

    __slots__ = ("_cooldown_s", "_failures", "_opened_at", "_probes", "_threshold")

    def __init__(self, *, failure_threshold: int = 5, cooldown_s: float = 30.0) -> None:
        self._threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}
        self._probes: dict[str, int] = {}

    def is_open(self, source_id: str) -> bool:
        opened = self._opened_at.get(source_id)
        if opened is None:
            return False
        if time.monotonic() - opened >= self._cooldown_s:
            # Half-open: let one request through to find out whether the source recovered.
            # Reopening on its failure costs one request; staying closed costs every request.
            self._opened_at.pop(source_id, None)
            self._failures[source_id] = self._threshold - 1
            self._probes[source_id] = self._probes.get(source_id, 0) + 1
            return False
        return True

    def record_success(self, source_id: str) -> None:
        self._failures.pop(source_id, None)
        self._opened_at.pop(source_id, None)

    def record_failure(self, source_id: str) -> None:
        count = self._failures.get(source_id, 0) + 1
        self._failures[source_id] = count
        if count >= self._threshold:
            self._opened_at[source_id] = time.monotonic()

    def state(self, source_id: str) -> str:
        """Breaker state, for the trace.

        A silently open breaker means quietly serving from fewer sources than the plan asked
        for, which shows up as a slow quality decline nobody can attribute.
        """
        if source_id in self._opened_at:
            return "open"
        return "half_open" if self._failures.get(source_id) else "closed"


class ParallelRetrievalOrchestrator:
    """Runs every leg of a plan concurrently, within the plan budget."""

    def __init__(
        self,
        sources: Mapping[str, KnowledgeSource],
        *,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._sources = dict(sources)
        self._breaker = breaker or CircuitBreaker()

    async def execute(
        self, plan: RetrievalPlan, principal: Principal, budget: Budget
    ) -> CandidatePool:
        """Execute the plan and return the fused candidate pool."""
        # The plan's own budget, capped by what the request has left. A plan that asked for
        # 260 ms cannot have it when 80 ms remain, and honouring the plan's number would spend
        # time belonging to generation.
        wall_ms = min(plan.budget.wall_ms, max(1, budget.wall_ms_remaining))

        tasks = [asyncio.create_task(self._run_leg(leg, principal, wall_ms)) for leg in plan.legs]

        try:
            # A single gather with one deadline over all legs, rather than awaiting each in
            # turn. Sequential awaits would make the stage cost the sum of the legs, which is
            # the whole thing parallelism exists to avoid.
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=wall_ms / 1000.0
            )
        except TimeoutError:
            results = []
            for leg, task in zip(plan.legs, tasks, strict=True):
                if task.done() and not task.cancelled():
                    outcome = task.exception() or task.result()
                    results.append(outcome)
                else:
                    task.cancel()
                    results.append(self._timed_out(leg))

        leg_results = tuple(
            self._as_leg_result(leg, outcome)
            for leg, outcome in zip(plan.legs, results, strict=True)
        )
        self._enforce_policy(plan, leg_results)

        return CandidatePool(
            plan_id=plan.plan_id,
            leg_results=leg_results,
            candidates=fuse_legs(leg_results, plan.fusion)[: plan.budget.max_candidates],
        )

    async def _run_leg(self, leg: RetrievalLeg, principal: Principal, wall_ms: int) -> LegResult:
        source = self._sources.get(leg.source_id)
        if source is None:
            raise SourceUnavailable(
                "leg names a source that is not registered",
                source_id=leg.source_id,
                leg_id=leg.leg_id,
            )

        if self._breaker.is_open(leg.source_id):
            # Never attempted. Distinct from a failure: nothing was tried, so the source's own
            # health is not further implicated and the trace should not say it was.
            return LegResult(
                leg_id=leg.leg_id,
                source_id=leg.source_id,
                status=LegStatus.SKIPPED_BREAKER_OPEN,
                latency_ms=0,
                error_reason_code="breaker_open",
            )

        from prag.core.models.common import Deadline

        deadline = Deadline.in_ms(min(leg.timeout_ms, wall_ms), label=leg.leg_id)
        try:
            result = await source.retrieve(leg, principal, deadline)
        except DeadlineExceeded:
            # A timeout is not the source failing, so it does not count toward the breaker.
            # Counting it would let one slow request open the breaker on a healthy source and
            # take it down for every other request.
            return self._timed_out(leg)
        except Exception:
            self._breaker.record_failure(leg.source_id)
            raise

        self._breaker.record_success(leg.source_id)
        return result

    @staticmethod
    def _timed_out(leg: RetrievalLeg) -> LegResult:
        return LegResult(
            leg_id=leg.leg_id,
            source_id=leg.source_id,
            status=LegStatus.TIMED_OUT,
            latency_ms=leg.timeout_ms,
            error_reason_code="deadline_exceeded",
        )

    @staticmethod
    def _as_leg_result(leg: RetrievalLeg, outcome: object) -> LegResult:
        """Normalise a gather outcome into a leg result.

        An exception becomes a FAILED leg rather than propagating, because whether one leg's
        failure fails the request is the plan's policy decision and not this method's.
        """
        if isinstance(outcome, LegResult):
            return outcome
        if isinstance(outcome, BaseException):
            return LegResult(
                leg_id=leg.leg_id,
                source_id=leg.source_id,
                status=LegStatus.FAILED,
                latency_ms=0,
                error_reason_code=getattr(outcome, "reason_code", "retrieval_failed"),
            )
        return LegResult(
            leg_id=leg.leg_id,
            source_id=leg.source_id,
            status=LegStatus.FAILED,
            latency_ms=0,
            error_reason_code="retrieval_failed",
        )

    @staticmethod
    def _enforce_policy(plan: RetrievalPlan, results: tuple[LegResult, ...]) -> None:
        """Decide whether what came back is enough to proceed on."""
        by_id = {result.leg_id: result for result in results}
        failed_required = [
            leg_id
            for leg_id in plan.required_leg_ids
            if leg_id in by_id and not by_id[leg_id].usable
        ]

        if plan.partial_results_policy is PartialResultsPolicy.ALL_OR_NOTHING:
            broken = [r.leg_id for r in results if not r.usable]
            if broken:
                raise RetrievalTotalFailure(
                    "plan requires every leg and some failed",
                    plan_id=plan.plan_id,
                    failed=broken,
                )
            return

        if failed_required:
            # Every required leg failed means the caller decides: answer parametrically with
            # explicit marking that the documents could not be reached, or abstain if the query
            # targets private data. Both are better than a confident answer built on nothing.
            raise RetrievalTotalFailure(
                "every required leg failed",
                plan_id=plan.plan_id,
                failed=failed_required,
            )
