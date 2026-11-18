"""A/B benchmark: compare pipelines on a gold-standard question set.

Metrics (per question, then averaged):
* fact_recall        – share of gold facts that appear in the brief
* grounding_rate     – share of final claims whose citation actually supports them
* hallucination_rate – share of final claims that no retrieved passage supports
* latency_ms         – wall-clock time per brief
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from .agents import AuditorAgent, Brief, BriefPipeline


@dataclass
class GoldItem:
    question: str
    facts: list[str]


@dataclass
class Scores:
    pipeline: str
    fact_recall: float
    grounding_rate: float
    hallucination_rate: float
    latency_ms: float


def load_gold(path: str | Path) -> list[GoldItem]:
    return [GoldItem(**row) for row in json.loads(Path(path).read_text(encoding="utf-8"))]


def score_brief(brief: Brief, gold: GoldItem, auditor: AuditorAgent) -> tuple[float, float, float]:
    text = " ".join(c.text for c in brief.claims).lower()
    recall = sum(f.lower() in text for f in gold.facts) / len(gold.facts) if gold.facts else 1.0
    verdicts = auditor.audit(brief.claims, brief.passages)  # independent re-check of the final output
    n = len(verdicts) or 1
    grounded = sum(v.status == "grounded" for v in verdicts) / n if verdicts else 0.0
    halluc = sum(v.status == "ungrounded" for v in verdicts) / n if verdicts else 0.0
    return recall, grounded, halluc


def run_benchmark(pipelines: dict[str, BriefPipeline], gold: list[GoldItem]) -> list[Scores]:
    judge = AuditorAgent()
    results = []
    for name, pipe in pipelines.items():
        rows = []
        for item in gold:
            brief = pipe.run(item.question)
            rows.append((*score_brief(brief, item, judge), brief.seconds * 1000))
        results.append(Scores(name, *(round(mean(col), 3) for col in zip(*rows, strict=True))))
    return results


def to_markdown(results: list[Scores]) -> str:
    head = "| Pipeline | Fact recall | Grounding rate | Hallucination rate | Latency (ms) |\n|---|---|---|---|---|"
    rows = [
        f"| {r.pipeline} | {r.fact_recall:.0%} | {r.grounding_rate:.0%} "
        f"| {r.hallucination_rate:.0%} | {r.latency_ms:.1f} |"
        for r in results
    ]
    return "\n".join([head, *rows])


def to_json(results: list[Scores]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)
