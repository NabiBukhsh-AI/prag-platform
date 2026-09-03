"""Budget enforcement and the degradation ladder.

The controller is consulted between every node, not at the edges. Checking only at the end
produces the worst possible outcome: the full cost is paid and then the client times out anyway,
having received nothing. Checking between nodes means a request that cannot finish well can
still finish usefully.

The ladder's ordering is not arbitrary. Each rung sheds the most cost for the least quality, so
a request degrades along the cheapest axis available before touching anything that matters more:

===== =================================================== ====================================
Level Action                                              What it costs
===== =================================================== ====================================
0     Nothing                                             --
1     Skip reranking, rank on fusion scores               Precision in the top-k ordering
2     Halve the evidence budget, skip compression         Coverage on broad queries
3     Drop optional retrieval legs                        Recall from secondary sources
4     Downgrade to the cheaper model profile              Reasoning quality
5     Return a partial answer with a budget warning       Completeness, stated explicitly
6     Abstain, reason ``budget_exceeded``                 The answer
===== =================================================== ====================================

Every transition is logged, which is the point. Budget-driven quality loss that is invisible
shows up in evaluation as an unexplained regression, and a week is then spent looking for a bug
in the reranker that was never run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from prag.core.models.identity import Budget

__all__ = [
    "BudgetController",
    "BudgetVerdict",
    "DegradationLevel",
    "DegradationPlan",
]


class DegradationLevel(IntEnum):
    """Rungs of the ladder. Higher is more degraded; a request never climbs back down."""

    NONE = 0
    SKIP_RERANK = 1
    REDUCE_EVIDENCE = 2
    DROP_OPTIONAL_LEGS = 3
    CHEAPER_MODEL = 4
    PARTIAL_ANSWER = 5
    ABSTAIN = 6


@dataclass(frozen=True, slots=True)
class DegradationPlan:
    """What is switched off at a given level.

    Cumulative: level 3 also implies levels 1 and 2. Expressing it as a set of booleans derived
    from one number means a node asks "may I rerank" rather than comparing against a level it
    would have to know the meaning of.
    """

    level: DegradationLevel

    @property
    def rerank_allowed(self) -> bool:
        return self.level < DegradationLevel.SKIP_RERANK

    @property
    def compression_allowed(self) -> bool:
        return self.level < DegradationLevel.REDUCE_EVIDENCE

    @property
    def evidence_budget_multiplier(self) -> float:
        """Fraction of the configured evidence budget still available."""
        return 0.5 if self.level >= DegradationLevel.REDUCE_EVIDENCE else 1.0

    @property
    def optional_legs_allowed(self) -> bool:
        return self.level < DegradationLevel.DROP_OPTIONAL_LEGS

    @property
    def premium_model_allowed(self) -> bool:
        return self.level < DegradationLevel.CHEAPER_MODEL

    @property
    def must_warn(self) -> bool:
        """Whether the response has to carry a budget warning.

        From level 5 the answer is knowingly incomplete, and an incomplete answer presented as
        complete is worse than a slow one.
        """
        return self.level >= DegradationLevel.PARTIAL_ANSWER

    @property
    def must_abstain(self) -> bool:
        return self.level >= DegradationLevel.ABSTAIN


@dataclass(frozen=True, slots=True)
class BudgetVerdict:
    """The controller's answer at one checkpoint."""

    budget: Budget
    plan: DegradationPlan
    #: True when this check moved the request to a new rung, so the caller knows to log and emit
    #: the degradation event exactly once rather than on every subsequent check.
    changed: bool
    reason: str | None = None

    @property
    def proceed(self) -> bool:
        return not self.plan.must_abstain


class BudgetController:
    """Decides how degraded a request should be, given what it has left.

    Pure with respect to the budget: it returns a new one rather than mutating, so a checkpoint
    can be evaluated in a test without constructing a request. Configuration comes in through
    the constructor because the thresholds are tenant-tunable and must not be constants buried
    in a method.

    The thresholds are fractions of the *original* budget still remaining. Fractions rather than
    absolute values because the same ladder has to work for a 650 ms interactive request and a
    30 s batch one, and an absolute threshold that suits one is nonsense for the other.
    """

    #: Remaining-fraction at which each rung engages. Ordered high to low, and read in that
    #: order, so a request that has already burned 95 percent of its budget lands directly on
    #: the rung that matches rather than walking there one node at a time.
    DEFAULT_THRESHOLDS: ClassVar[dict[DegradationLevel, float]] = {
        DegradationLevel.SKIP_RERANK: 0.50,
        DegradationLevel.REDUCE_EVIDENCE: 0.35,
        DegradationLevel.DROP_OPTIONAL_LEGS: 0.25,
        DegradationLevel.CHEAPER_MODEL: 0.15,
        DegradationLevel.PARTIAL_ANSWER: 0.08,
        DegradationLevel.ABSTAIN: 0.0,
    }

    def __init__(self, thresholds: dict[DegradationLevel, float] | None = None) -> None:
        self._thresholds = dict(thresholds or self.DEFAULT_THRESHOLDS)
        for level, fraction in self._thresholds.items():
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"threshold for {level.name} must be a fraction in [0, 1], got {fraction}"
                )

    def check(self, budget: Budget, *, node_id: str) -> BudgetVerdict:
        """Evaluate the budget at a checkpoint and return the level to run at.

        Called between nodes by the interpreter. ``node_id`` names the node about to run, so a
        degradation event says which step the request could not afford rather than only that it
        ran out somewhere.
        """
        wall_fraction = budget.wall_ms_remaining / budget.wall_ms_total
        cost_fraction = budget.usd_remaining / budget.usd_total if budget.usd_total > 0 else 1.0

        # Whichever resource is scarcer sets the level. A request with plenty of time and no
        # money left is just as constrained as the reverse, and averaging the two would let a
        # surplus of one hide the exhaustion of the other.
        scarcest = min(wall_fraction, cost_fraction)
        binding = "wall_clock" if wall_fraction <= cost_fraction else "cost"

        target = self._level_for(scarcest)
        # Never un-degrade: the level already reached is a floor.
        effective = DegradationLevel(max(target, budget.degradation_level))
        changed = effective > budget.degradation_level

        return BudgetVerdict(
            budget=budget.degraded_to(effective) if changed else budget,
            plan=DegradationPlan(level=effective),
            changed=changed,
            reason=(f"{binding} at {scarcest:.0%} of budget before {node_id}" if changed else None),
        )

    def _level_for(self, remaining_fraction: float) -> DegradationLevel:
        """The most degraded rung whose threshold this fraction has crossed."""
        level = DegradationLevel.NONE
        for candidate, threshold in sorted(self._thresholds.items()):
            if remaining_fraction <= threshold:
                level = max(level, candidate)
        return level

    def plan_for(self, budget: Budget) -> DegradationPlan:
        """The plan matching a budget's current level, without re-evaluating thresholds.

        For nodes that need to know what they are allowed to do but are not themselves a
        checkpoint. A reranker asking "am I allowed to run" should not be able to change the
        request's degradation level by asking.
        """
        return DegradationPlan(level=DegradationLevel(budget.degradation_level))
