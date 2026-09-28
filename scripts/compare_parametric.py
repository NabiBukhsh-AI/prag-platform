#!/usr/bin/env python
"""Parametric versus non-parametric on the same corpus subset (the Phase 4 deliverable).

Runs the seed golden set twice over the same tenant corpus: once with the parametric tier off,
once with adapters trained on each seed document and promoted through shadow. Reports, per query
and in total, which route answered, cost, latency, faithfulness (groundedness) and completeness.

READ THIS BEFORE READING THE NUMBERS. The parametric side uses the local stand-in model
(``prag.parametric.local``), which memorises QA pairs rather than learning weights. The numbers
measure the platform's routing, gating, shadowing and cost accounting — whether the right route
was taken, whether a parametric answer was verified, what each path cost — and say nothing about
how well a real LoRA adapter would learn this corpus. That comparison needs the GPU trainer and
vLLM serving, run through this same script.

    python scripts/compare_parametric.py
    python scripts/compare_parametric.py --json report.json
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prag.api import build_platform
from prag.config import PragSettings
from prag.config.schema import (
    AdapterSelectionConfig,
    EvaluationConfig,
    ParametricConfig,
    RoutingConfig,
)
from prag.core.errors import AbstentionRequired
from prag.core.models.common import Deadline, EmbeddingPurpose, VolatilityClass
from prag.core.models.parametric import KnowledgeClass, KnowledgeRecord, ParametricEconomics
from prag.evaluation import load_cases
from prag.parametric import (
    EligibilityGate,
    MemorizingTrainer,
    Passage,
    cluster_passages,
    parameterize_cluster,
    promote_from_shadow,
)

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "eval" / "seed"
TENANT = "tenant-local"
DOCUMENTS = ("runbook.incident-response", "policy.access-control")


def settings(parametric: bool) -> PragSettings:
    base = {"routing": RoutingConfig(exploration_fraction=0.0)}
    if not parametric:
        return PragSettings(**base)
    return PragSettings(
        **base,
        # 0.35 rather than the production 0.62: the local hashing embedder scores query-to-centroid
        # similarity around 0.1-0.5, where a semantic embedder scores far higher.
        parametric=ParametricConfig(
            enabled=True, selection=AdapterSelectionConfig(min_coverage_similarity=0.35)
        ),
        # Required by the config before the tier can be switched on. No judge scores are used.
        evaluation=EvaluationConfig(judge_calibrated=True),
    )


async def seed(platform) -> None:
    for name in DOCUMENTS:
        text = (SEED / "corpus" / TENANT / "public" / f"{name}.md").read_text(encoding="utf-8")
        await platform.ingest_markdown(text, document_id=name, tenant_id=TENANT)


async def train_adapters(platform) -> list[str]:
    trained = []
    for name in DOCUMENTS:
        text = (SEED / "corpus" / TENANT / "public" / f"{name}.md").read_text(encoding="utf-8")
        paragraphs = [p for p in text.split("\n\n") if len(p.split()) > 12]
        vectors = await platform.embedder.embed(
            paragraphs, EmbeddingPurpose.DOCUMENT, Deadline.in_ms(5_000)
        )
        passages = [
            Passage(f"{name}#{i}", name, p, tuple(v))
            for i, (p, v) in enumerate(zip(paragraphs, vectors, strict=True))
        ]
        (cluster,) = cluster_passages(passages, threshold=0.0)
        cluster = dataclasses.replace(cluster, cluster_id=name)
        report = await parameterize_cluster(
            cluster,
            knowledge=KnowledgeRecord(
                record_id=name,
                knowledge_class=KnowledgeClass.NON_PARAMETRIC_STATIC,
                tenant_id=TENANT,
                acl_narrower_than_tenant=False,
                volatility_class=VolatilityClass.STATIC,
                estimated_half_life_days=365.0,
            ),
            economics=ParametricEconomics(
                training_gpu_cost_usd=20.0,
                storage_cost_per_period_usd=1.0,
                expected_queries_per_period=10_000,
                evidence_tokens_displaced=3_000,
                prefill_cost_per_token_usd=0.000_003,
                retrieval_infra_cost_per_query_usd=0.000_5,
                cluster_coherence_score=0.8,
            ),
            gate=EligibilityGate(),
            registry=platform.parametric.registry,
            tenant_scope=TENANT,
            base_model_id="base.lora",
            base_model_version="1",
            embedding_model_version=platform.embedder.model_version,
            domain="general",
            trainer=MemorizingTrainer(),
        )
        if report.record is None:
            print(f"  {name}: not parameterized ({', '.join(report.refused)})")
            continue
        await promote_from_shadow(
            platform.parametric.registry, report.record,
            parametric_faithfulness=1.0, baseline_faithfulness=1.0,
        )
        trained.append(report.record.adapter_id)
        print(f"  {name}: adapter {report.record.adapter_id}, probes {report.probes}")
    return trained


async def run_case(platform, case) -> dict:
    state = platform.request_state(case.query, {"x-tenant-id": TENANT, "x-user-id": "compare"})
    started = time.perf_counter()
    try:
        run = await platform.answer(state)
    except AbstentionRequired:
        return {"route": "abstained", "latency_ms": (time.perf_counter() - started) * 1000}
    latency = (time.perf_counter() - started) * 1000
    envelope = run.state.result
    answer = envelope.answer if envelope else ""
    facts = case.expected_facts
    return {
        "route": "parametric" if run.path[-1] == "shadow" else "grounded",
        "path": list(run.path),
        "latency_ms": round(latency, 2),
        "usd": run.state.budget.usd_spent,
        "groundedness": envelope.grounding.groundedness if envelope else 0.0,
        "completeness": (
            sum(1 for f in facts if f.casefold() in answer.casefold()) / len(facts)
            if facts else None
        ),
        "conflicts": sum(1 for e in run.state.events if e.kind == "parametric_retrieval_conflict"),
    }


def summarise(rows: list[dict]) -> dict:
    answered = [r for r in rows if r["route"] != "abstained"]
    complete = [r["completeness"] for r in answered if r.get("completeness") is not None]
    return {
        "answered": len(answered),
        "parametric_served": sum(1 for r in rows if r["route"] == "parametric"),
        "usd_per_query": sum(r.get("usd", 0.0) for r in answered) / max(1, len(answered)),
        "mean_latency_ms": sum(r["latency_ms"] for r in rows) / max(1, len(rows)),
        "mean_groundedness": sum(r["groundedness"] for r in answered) / max(1, len(answered)),
        "mean_completeness": sum(complete) / max(1, len(complete)),
        "conflicts": sum(r.get("conflicts", 0) for r in rows),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Parametric versus non-parametric comparison.")
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args()

    cases = [
        c for c in load_cases(SEED / "golden.jsonl")
        if c.tenant_id == TENANT and not c.acl_hashes and not c.expect_abstain
    ]

    baseline = build_platform(settings(parametric=False))
    await seed(baseline)
    parametric = build_platform(settings(parametric=True))
    await seed(parametric)
    print("Training adapters (local stand-in model — see this script's docstring):")
    await train_adapters(parametric)

    rows = []
    for case in cases:
        rows.append(
            {
                "case_id": case.case_id,
                "baseline": await run_case(baseline, case),
                "parametric": await run_case(parametric, case),
            }
        )

    print(f"\n{'case':<24} {'route':<11} {'usd':>9} {'ms':>7} {'ground':>7} {'complete':>9}")
    for row in rows:
        for side in ("baseline", "parametric"):
            r = row[side]
            print(
                f"{row['case_id'] if side == 'baseline' else '':<24} "
                f"{r['route']:<11} {r.get('usd', 0.0):>9.5f} {r['latency_ms']:>7.2f} "
                f"{r.get('groundedness', 0.0):>7.2f} "
                f"{'' if r.get('completeness') is None else format(r['completeness'], '.2f'):>9}"
            )

    summary = {
        "baseline": summarise([r["baseline"] for r in rows]),
        "parametric": summarise([r["parametric"] for r in rows]),
    }
    print("\nSummary (stand-in model: mechanics and cost accounting, not LoRA quality)")
    for side, values in summary.items():
        print(f"  {side:<11} " + "  ".join(f"{k}={v:.4g}" for k, v in values.items()))

    if args.json:
        args.json.write_text(
            json.dumps({"cases": rows, "summary": summary}, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
