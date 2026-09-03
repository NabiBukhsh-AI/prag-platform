"""The budget controller and the degradation ladder."""

from __future__ import annotations

import pytest

from prag.core.budget import BudgetController, DegradationLevel, DegradationPlan
from prag.core.models.identity import Budget


def a_budget(*, wall_remaining: int = 1000, usd_spent: float = 0.0) -> Budget:
    return Budget(
        wall_ms_total=1000,
        wall_ms_remaining=wall_remaining,
        usd_total=0.10,
        usd_spent=usd_spent,
        max_tokens_in=8000,
        max_tokens_out=1024,
    )


class TestBudgetArithmetic:
    def test_spending_returns_a_new_budget(self) -> None:
        """Frozen, so an overspend cannot appear retroactively in a recorded state."""
        original = a_budget()
        after = original.spend(wall_ms=300, usd=0.01)

        assert original.wall_ms_remaining == 1000, "the original must be untouched"
        assert after.wall_ms_remaining == 700
        assert after.usd_spent == pytest.approx(0.01)
        assert after is not original

    def test_remaining_clamps_at_zero(self) -> None:
        """An overrun is real, but a negative remainder makes every share() nonsense."""
        after = a_budget().spend(wall_ms=5000)
        assert after.wall_ms_remaining == 0
        assert after.wall_exhausted
        assert after.exhausted

    def test_cost_exhaustion_is_independent_of_time(self) -> None:
        broke = a_budget(usd_spent=0.10)
        assert broke.cost_exhausted
        assert not broke.wall_exhausted
        assert broke.exhausted, "either resource running out exhausts the request"

    def test_negative_spend_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            a_budget().spend(wall_ms=-1)

    def test_deadline_is_a_share_of_what_remains(self) -> None:
        """Derived from the remainder, which is what prevents cascading timeouts."""
        deadline = a_budget(wall_remaining=800).deadline_for("retrieve", 0.25)
        assert 150 < deadline.remaining_ms <= 200
        assert deadline.label == "retrieve"

    @pytest.mark.parametrize("share", [0.0, -0.5, 1.5])
    def test_invalid_share_is_rejected(self, share: float) -> None:
        with pytest.raises(ValueError, match="share"):
            a_budget().deadline_for("node", share)


class TestDegradationPlan:
    """Each rung is cumulative: level 3 also implies levels 1 and 2."""

    def test_undegraded_allows_everything(self) -> None:
        plan = DegradationPlan(level=DegradationLevel.NONE)
        assert plan.rerank_allowed
        assert plan.compression_allowed
        assert plan.optional_legs_allowed
        assert plan.premium_model_allowed
        assert plan.evidence_budget_multiplier == 1.0
        assert not plan.must_warn
        assert not plan.must_abstain

    def test_rungs_are_cumulative(self) -> None:
        plan = DegradationPlan(level=DegradationLevel.DROP_OPTIONAL_LEGS)
        assert not plan.rerank_allowed, "level 3 implies level 1"
        assert not plan.compression_allowed, "level 3 implies level 2"
        assert not plan.optional_legs_allowed
        assert plan.premium_model_allowed, "level 4 has not been reached"
        assert plan.evidence_budget_multiplier == 0.5

    def test_partial_answer_must_warn(self) -> None:
        """An incomplete answer presented as complete is worse than a slow one."""
        assert DegradationPlan(level=DegradationLevel.PARTIAL_ANSWER).must_warn

    def test_abstain_is_terminal(self) -> None:
        plan = DegradationPlan(level=DegradationLevel.ABSTAIN)
        assert plan.must_abstain
        assert plan.must_warn


class TestBudgetController:
    def test_fresh_budget_is_undegraded(self) -> None:
        verdict = BudgetController().check(a_budget(), node_id="analyze")
        assert verdict.plan.level is DegradationLevel.NONE
        assert not verdict.changed
        assert verdict.reason is None
        assert verdict.proceed

    @pytest.mark.parametrize(
        ("remaining", "expected"),
        [
            (1000, DegradationLevel.NONE),
            (500, DegradationLevel.SKIP_RERANK),
            (350, DegradationLevel.REDUCE_EVIDENCE),
            (250, DegradationLevel.DROP_OPTIONAL_LEGS),
            (150, DegradationLevel.CHEAPER_MODEL),
            (80, DegradationLevel.PARTIAL_ANSWER),
            (0, DegradationLevel.ABSTAIN),
        ],
    )
    def test_ladder_engages_at_each_threshold(
        self, remaining: int, expected: DegradationLevel
    ) -> None:
        verdict = BudgetController().check(a_budget(wall_remaining=remaining), node_id="n")
        assert verdict.plan.level is expected

    def test_jumps_straight_to_the_matching_rung(self) -> None:
        """A request that has burned 95 percent of its budget lands where it belongs.

        Walking down one rung per node would spend several more nodes discovering what the
        first check already knew.
        """
        verdict = BudgetController().check(a_budget(wall_remaining=50), node_id="generate")
        assert verdict.plan.level is DegradationLevel.PARTIAL_ANSWER

    def test_cost_can_be_the_binding_constraint(self) -> None:
        """Plenty of time, no money left: just as constrained, and the reason says so."""
        verdict = BudgetController().check(
            a_budget(wall_remaining=1000, usd_spent=0.095), node_id="rerank"
        )
        assert verdict.plan.level is DegradationLevel.PARTIAL_ANSWER
        assert verdict.reason is not None
        assert "cost" in verdict.reason

    def test_scarcest_resource_wins(self) -> None:
        """A surplus of one resource must not hide the exhaustion of the other."""
        verdict = BudgetController().check(a_budget(wall_remaining=200, usd_spent=0.0), node_id="n")
        assert verdict.reason is not None
        assert "wall_clock" in verdict.reason

    def test_never_un_degrades(self) -> None:
        """A request whose level oscillates produces variance no evaluation can explain."""
        already = a_budget(wall_remaining=1000).degraded_to(DegradationLevel.CHEAPER_MODEL)
        verdict = BudgetController().check(already, node_id="n")

        assert verdict.plan.level is DegradationLevel.CHEAPER_MODEL
        assert not verdict.changed, "staying put is not a transition"

    def test_changed_fires_once_per_transition(self) -> None:
        """So the degradation event is emitted once, not on every subsequent check."""
        controller = BudgetController()
        first = controller.check(a_budget(wall_remaining=400), node_id="n1")
        assert first.changed

        second = controller.check(first.budget, node_id="n2")
        assert not second.changed
        assert second.plan.level is first.plan.level

    def test_verdict_carries_the_updated_budget(self) -> None:
        verdict = BudgetController().check(a_budget(wall_remaining=300), node_id="n")
        assert verdict.budget.degradation_level == int(verdict.plan.level)

    def test_zero_cost_budget_does_not_divide_by_zero(self) -> None:
        """A free local model is a legitimate configuration, not an error."""
        free = Budget(
            wall_ms_total=1000,
            wall_ms_remaining=1000,
            usd_total=0.0,
            max_tokens_in=100,
            max_tokens_out=100,
        )
        assert BudgetController().check(free, node_id="n").plan.level is DegradationLevel.NONE

    def test_custom_thresholds_are_validated(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            BudgetController({DegradationLevel.SKIP_RERANK: 1.5})

    def test_plan_for_does_not_change_the_level(self) -> None:
        """Asking what you may do must not change what you may do."""
        controller = BudgetController()
        budget = a_budget(wall_remaining=100)
        plan = controller.plan_for(budget)

        assert plan.level is DegradationLevel.NONE, "reads the recorded level, not the clock"
        assert budget.degradation_level == 0
