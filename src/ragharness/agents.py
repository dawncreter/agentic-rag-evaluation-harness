"""Research -> Audit -> Revise loop.

The graph is three nodes with a conditional edge, written as plain Python so it runs anywhere.
Each node is a pure function of ``BriefState``; the same functions can be registered as
LangGraph nodes without changes.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .chunker import Chunk
from .llm import Claim, LLMClient
from .retriever import HybridRetriever, tokenize

_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


def _numbers(text: str) -> set[str]:
    return {n.rstrip(",") for n in _NUMBER.findall(text)}


@dataclass
class Verdict:
    claim: Claim
    status: str  # "grounded" | "miscited" | "ungrounded"
    support: float
    best_chunk: str | None


@dataclass
class BriefState:
    question: str
    passages: list[Chunk] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    revisions: int = 0
    log: list[str] = field(default_factory=list)


class AuditorAgent:
    """Checks every claim against source text.

    support(claim, chunk) = |content tokens of claim found in chunk| / |content tokens of claim|,
    and every number in the claim (amounts, percentages, days) must appear verbatim in the chunk.
    """

    def __init__(self, threshold: float = 0.6):
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    @staticmethod
    def support(claim: str, chunk_text: str) -> float:
        ct = set(tokenize(claim))
        if not ct:
            return 0.0
        nums = _numbers(claim)
        if nums and not nums <= _numbers(chunk_text):
            return 0.0
        return len(ct & set(tokenize(chunk_text))) / len(ct)

    def audit(self, claims: list[Claim], passages: list[Chunk]) -> list[Verdict]:
        by_id = {p.id: p for p in passages}
        verdicts = []
        for c in claims:
            cited = max((self.support(c.text, by_id[i].text) for i in c.citations if i in by_id), default=0.0)
            best_id, best = None, 0.0
            for p in passages:
                s = self.support(c.text, p.text)
                if s > best:
                    best_id, best = p.id, s
            if cited >= self.threshold:
                verdicts.append(Verdict(c, "grounded", cited, best_id))
            elif best >= self.threshold:
                verdicts.append(Verdict(c, "miscited", best, best_id))
            else:
                verdicts.append(Verdict(c, "ungrounded", best, best_id))
        return verdicts


class BriefPipeline:
    """Orchestrates retrieval, drafting, auditing and revision."""

    def __init__(self, retriever: HybridRetriever, llm: LLMClient, auditor: AuditorAgent | None = None,
                 k: int = 4, max_revisions: int = 2, audit: bool = True):
        self.retriever, self.llm = retriever, llm
        self.auditor = auditor or AuditorAgent()
        self.k, self.max_revisions, self.audit_enabled = k, max_revisions, audit

    # --- nodes -----------------------------------------------------------
    def research(self, s: BriefState) -> BriefState:
        s.passages = [h.chunk for h in self.retriever.search(s.question, self.k)]
        s.claims = self.llm.draft(s.question, s.passages)
        s.log.append(f"research: {len(s.passages)} passages, {len(s.claims)} claims")
        return s

    def audit(self, s: BriefState) -> BriefState:
        s.verdicts = self.auditor.audit(s.claims, s.passages)
        counts = {k: sum(v.status == k for v in s.verdicts) for k in ("grounded", "miscited", "ungrounded")}
        s.log.append(f"audit: {counts}")
        return s

    def revise(self, s: BriefState) -> BriefState:
        kept = []
        for v in s.verdicts:
            if v.status == "grounded":
                kept.append(v.claim)
            elif v.status == "miscited" and v.best_chunk:
                kept.append(Claim(v.claim.text, [v.best_chunk]))  # re-anchor to the supporting source
        s.claims, s.revisions = kept, s.revisions + 1
        s.log.append(f"revise: kept {len(kept)} claims")
        return s

    @staticmethod
    def needs_revision(s: BriefState) -> bool:
        return any(v.status != "grounded" for v in s.verdicts)

    # --- graph -----------------------------------------------------------
    def run(self, question: str) -> Brief:
        start = time.perf_counter()
        s = self.research(BriefState(question))
        if self.audit_enabled:
            s = self.audit(s)
            while self.needs_revision(s) and s.revisions < self.max_revisions:
                s = self.audit(self.revise(s))
        return Brief(question, s.claims, s.passages, s.verdicts, s.log, time.perf_counter() - start)


@dataclass
class Brief:
    question: str
    claims: list[Claim]
    passages: list[Chunk]
    verdicts: list[Verdict]
    log: list[str]
    seconds: float

    def to_markdown(self) -> str:
        lines = [f"### {self.question}", ""]
        lines += [f"- {c.text} [{', '.join(c.citations)}]" for c in self.claims] or ["- No grounded answer found."]
        return "\n".join(lines)
