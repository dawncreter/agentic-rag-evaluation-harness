"""Hybrid retrieval: BM25 keyword scoring + TF-IDF cosine, fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .chunker import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was "
    "were what when where which who why will with within does do should must can our we".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, tokenize and drop stopwords."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25:
    """Okapi BM25 over a fixed set of chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text + " " + c.section)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if chunks else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for d, dl in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in q:
                tf = d.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


class TfidfCosine:
    """Sparse TF-IDF vectors with cosine similarity (a dependency-free stand-in for dense embeddings).

    Swap in a real embedding model by implementing the same ``scores(query)`` method.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        tokenized = [tokenize(c.text) for c in chunks]
        df: Counter[str] = Counter()
        for toks in tokenized:
            df.update(set(toks))
        n = max(len(chunks), 1)
        self.idf = {t: math.log((n + 1) / (f + 1)) + 1 for t, f in df.items()}
        self.vecs = [self._vec(toks) for toks in tokenized]

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        v = {t: c * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def scores(self, query: str) -> list[float]:
        q = self._vec(tokenize(query))
        return [sum(w * v.get(t, 0.0) for t, w in q.items()) for v in self.vecs]


class HybridRetriever:
    """Fuse keyword and vector rankings with Reciprocal Rank Fusion (RRF)."""

    def __init__(self, chunks: list[Chunk], rrf_k: int = 60):
        if not chunks:
            raise ValueError("HybridRetriever needs at least one chunk")
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.bm25 = BM25(chunks)
        self.dense = TfidfCosine(chunks)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        fused = [0.0] * len(self.chunks)
        for scores in (self.bm25.scores(query), self.dense.scores(query)):
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for rank, i in enumerate(ranked):
                if scores[i] > 0:
                    fused[i] += 1.0 / (self.rrf_k + rank + 1)
        order = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)
        return [Hit(self.chunks[i], fused[i]) for i in order[:k] if fused[i] > 0]
