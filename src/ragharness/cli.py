"""Command line entry point.

    rag-harness ask "How long are records kept?"
    rag-harness bench
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agents import BriefPipeline
from .chunker import load_corpus
from .evaluation import load_gold, run_benchmark, to_markdown
from .llm import ExtractiveLLM, LLMClient, NoisyLLM
from .retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "datasets" / "corpus"
DEFAULT_GOLD = ROOT / "datasets" / "gold_qa.json"


def _llm() -> LLMClient:
    if os.environ.get("ANTHROPIC_API_KEY"):
        from .llm import AnthropicLLM

        return AnthropicLLM()
    return ExtractiveLLM()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag-harness")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    bench = sub.add_parser("bench")
    bench.add_argument("--gold", default=str(DEFAULT_GOLD))
    args = ap.parse_args(argv)

    retriever = HybridRetriever(load_corpus(args.corpus))
    if args.cmd == "ask":
        brief = BriefPipeline(retriever, _llm()).run(args.question)
        print(brief.to_markdown())
        print("\n" + "\n".join(f"  · {line}" for line in brief.log))
        return 0

    noisy = NoisyLLM(ExtractiveLLM(), hallucination_rate=0.6, miscite_rate=0.4)
    pipelines = {
        "A: noisy model, no auditor": BriefPipeline(retriever, noisy, audit=False),
        "B: noisy model + auditor": BriefPipeline(retriever, NoisyLLM(ExtractiveLLM(), 0.6, 0.4)),
        "C: baseline model + auditor": BriefPipeline(retriever, ExtractiveLLM()),
    }
    print(to_markdown(run_benchmark(pipelines, load_gold(args.gold))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
