"""Model clients. Every client turns (question, passages) into cited claims.

* ``ExtractiveLLM``  – deterministic, offline; picks the best-supported sentences. Used in tests/CI.
* ``NoisyLLM``       – wraps any client and injects unsupported or mis-cited claims, to stress-test the auditor.
* ``AnthropicLLM``   – calls Claude through the official SDK when ``ANTHROPIC_API_KEY`` is set.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Protocol

from .chunker import Chunk, split_sentences
from .retriever import tokenize


@dataclass
class Claim:
    text: str
    citations: list[str] = field(default_factory=list)


class LLMClient(Protocol):
    name: str

    def draft(self, question: str, passages: list[Chunk]) -> list[Claim]: ...


class ExtractiveLLM:
    """Selects the sentences that best answer the question and cites their source chunk."""

    name = "extractive-baseline"

    def __init__(self, max_claims: int = 3):
        self.max_claims = max_claims

    def draft(self, question: str, passages: list[Chunk]) -> list[Claim]:
        q = set(tokenize(question))
        scored: list[tuple[float, str, str]] = []
        for chunk in passages:
            for sent in split_sentences(chunk.text):
                toks = set(tokenize(sent))
                if toks:
                    scored.append((len(q & toks) / len(q | toks), sent, chunk.id))
        scored.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        claims: list[Claim] = []
        floor = scored[0][0] * 0.5 if scored else 0.0  # keep only sentences close to the best match
        for score, sent, cid in scored:
            if score <= 0 or score < floor or sent in seen:
                continue
            seen.add(sent)
            claims.append(Claim(sent, [cid]))
            if len(claims) == self.max_claims:
                break
        return claims


class NoisyLLM:
    """Simulates common LLM failure modes: invented facts and wrong citations."""

    FABRICATIONS = (
        "This policy was approved unanimously by the board in 2019.",
        "Exceptions are granted automatically after 90 days.",
        "Violations carry a mandatory fine of $250,000.",
    )

    def __init__(self, inner: LLMClient, hallucination_rate: float = 0.5, miscite_rate: float = 0.3, seed: int = 7):
        self.inner = inner
        self.name = f"noisy({inner.name})"
        self.h, self.m = hallucination_rate, miscite_rate
        self.rng = random.Random(seed)

    def draft(self, question: str, passages: list[Chunk]) -> list[Claim]:
        claims = self.inner.draft(question, passages)
        ids = [p.id for p in passages]
        for c in claims:
            if len(ids) > 1 and self.rng.random() < self.m:
                c.citations = [self.rng.choice([i for i in ids if i not in c.citations])]
        if passages and self.rng.random() < self.h:
            claims.append(Claim(self.rng.choice(self.FABRICATIONS), [passages[0].id]))
        return claims


class AnthropicLLM:
    """Claude-backed drafting. Requires ``pip install anthropic`` and ``ANTHROPIC_API_KEY``."""

    PROMPT = (
        "Answer the question using ONLY the passages. Return JSON: "
        '{{"claims": [{{"text": "...", "citations": ["<passage id>"]}}]}}. '
        "Every claim must cite at least one passage id.\n\nQuestion: {q}\n\nPassages:\n{p}"
    )

    def __init__(self, model: str | None = None):
        import anthropic  # optional dependency

        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model or os.environ.get("RAG_MODEL", "claude-sonnet-4-5")
        self.name = self.model

    def draft(self, question: str, passages: list[Chunk]) -> list[Claim]:
        body = "\n".join(f"[{p.id}] ({p.section}) {p.text}" for p in passages)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{"role": "user", "content": self.PROMPT.format(q=question, p=body)}],
        )
        raw = msg.content[0].text
        match = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(match.group(0)) if match else {"claims": []}
        return [Claim(c["text"], list(c.get("citations", []))) for c in data.get("claims", [])]
