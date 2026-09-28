#!/usr/bin/env python
"""Run the evaluation suite and enforce the regression gate.

Scores the golden and adversarial sets against a freshly seeded platform, prints a scorecard per
set, and exits non-zero if any gated metric is below its floor or has regressed past tolerance
from the committed baseline. That exit code is the CI gate: a regression on faithfulness,
citation precision or the adversarial set fails the build.

    python scripts/run_eval.py                    # score and gate
    python scripts/run_eval.py --update-baseline  # score, gate, then record this run as baseline
    python scripts/run_eval.py --dataset-dir DIR  # a private dataset with the same layout
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prag.api import build_platform
from prag.api.evaluation import ingest_corpus, run_case
from prag.config import PragSettings
from prag.config.schema import GuardrailsConfig
from prag.evaluation import (
    GateResult,
    Scorecard,
    load_cases,
    regression_gate,
    score_samples,
    standard_metrics,
)

ROOT = Path(__file__).resolve().parent.parent


def floors(settings: PragSettings) -> dict[str, dict[str, float]]:
    """Per-dataset floors. The adversarial floor is 100 percent: one miss is an incident."""
    evaluation = settings.evaluation
    return {
        "golden": {
            "hit_rate@5": 0.9,
            "mrr": 0.7,
            "groundedness": evaluation.faithfulness_min,
            "citation_precision": evaluation.citation_precision_min,
            "abstention_correct": 1.0,
        },
        "adversarial": {"adversarial": evaluation.adversarial_pass_rate_min},
    }


def render(card: Scorecard, gate: GateResult) -> str:
    lines = [f"\n== {card.dataset_id} =="]
    for metric_id, s in card.summaries.items():
        rate = "" if s.pass_rate is None else f"  pass {s.pass_rate:6.1%}"
        lines.append(f"  {metric_id:<20} {s.mean:6.3f}  n={s.n:<3}{rate}")
    for miss in card.failures():
        lines.append(f"  miss: {miss.metric_id} on {miss.sample_id} ({miss.score:.2f})")
    lines.extend(f"  UNGATED {u}" for u in gate.ungated)
    lines.extend(f"  FAIL {f}" for f in gate.failures)
    lines.append("  gate: PASS" if gate.passed else "  gate: FAIL")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "eval" / "seed")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()
    dataset_dir: Path = args.dataset_dir

    canaries_path = dataset_dir / "canaries.json"
    canaries = json.loads(canaries_path.read_text("utf-8")) if canaries_path.exists() else {}
    settings = PragSettings(guardrails=GuardrailsConfig(canaries=canaries))
    platform = build_platform(settings)
    indexed = await ingest_corpus(platform, dataset_dir / "corpus")
    print(f"indexed {indexed} chunks from {dataset_dir / 'corpus'}")

    passed = True
    for name, dataset_floors in floors(settings).items():
        cases = load_cases(dataset_dir / f"{name}.jsonl")
        samples = [await run_case(platform, case) for case in cases]
        card = await score_samples(standard_metrics(), samples, dataset_id=f"seed.{name}")

        baseline_path = dataset_dir / "baselines" / f"{name}.json"
        baseline = (
            Scorecard.from_json(baseline_path.read_text("utf-8"))
            if baseline_path.exists()
            else None
        )
        gate = regression_gate(
            card,
            floors=dataset_floors,
            baseline=baseline,
            judge_calibrated=settings.evaluation.judge_calibrated,
        )
        print(render(card, gate))
        passed = passed and gate.passed

        if args.update_baseline:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(card.to_json() + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
