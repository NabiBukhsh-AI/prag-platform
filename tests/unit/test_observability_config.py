"""The alert rules and dashboard reference only metrics and labels that exist.

An alert on a renamed metric does not fail — it stops firing. This is the test that turns that
silence into a red build.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from prag.observability import METRICS

DEPLOY = Path(__file__).resolve().parents[2] / "deploy" / "observability"
_SERIES = re.compile(r"\b(prag_[a-z_]+?)(?:_bucket|_sum|_count)?\b(?:\{([^}]*)\})?")
_LABEL = re.compile(r"(\w+)\s*(?:=~|!=|!~|=)")
#: Labels Prometheus adds or aggregations introduce, not declared on the metric itself.
_BUILTIN_LABELS = {"le"}


def expressions() -> list[tuple[str, str]]:
    alerts = (DEPLOY / "alerts.yml").read_text(encoding="utf-8")
    dashboard = json.loads((DEPLOY / "dashboard.json").read_text(encoding="utf-8"))
    rule = re.compile(r"expr:\s*>?-?\s*\n?((?:.+\n?)+?)(?=\s+(?:for|labels):)")
    exprs = [("alerts.yml", m) for m in rule.findall(alerts)]
    exprs += [
        (f"dashboard: {panel['title']}", target["expr"])
        for panel in dashboard["panels"]
        for target in panel["targets"]
    ]
    return exprs


def test_the_files_define_something() -> None:
    assert len(expressions()) >= 15


@pytest.mark.parametrize(("where", "expr"), expressions())
def test_every_series_and_label_is_declared(where: str, expr: str) -> None:
    for name, selector in _SERIES.findall(expr):
        assert name in METRICS, f"{where}: {name} is not a declared metric"
        declared = set(METRICS[name].labels) | _BUILTIN_LABELS
        for label in _LABEL.findall(selector):
            assert label in declared, f"{where}: {name} has no label {label!r}"


def test_filtered_label_values_are_ones_the_code_emits() -> None:
    """A renamed reason code silences an alert exactly as a renamed metric would."""
    source = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (Path(__file__).resolve().parents[2] / "src" / "prag").rglob("*.py")
    )
    for where, expr in expressions():
        for label, value in re.findall(r'(reason|outcome)="([^"]+)"', expr):
            assert f'"{value}"' in source, f"{where}: nothing emits {label}={value!r}"


def test_by_clauses_group_on_declared_labels() -> None:
    all_labels = {label for m in METRICS.values() for label in m.labels} | _BUILTIN_LABELS
    for where, expr in expressions():
        for group in re.findall(r"(?:by|ignoring)\s*\(([^)]*)\)", expr):
            for label in (g.strip() for g in group.split(",") if g.strip()):
                assert label in all_labels, f"{where}: groups on unknown label {label!r}"
