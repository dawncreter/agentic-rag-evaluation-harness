from pathlib import Path

import pytest

from ragharness import (
    AuditorAgent,
    BriefPipeline,
    Chunk,
    Claim,
    ExtractiveLLM,
    HybridRetriever,
    NoisyLLM,
    chunk_markdown,
    load_corpus,
)
from ragharness.chunker import split_sentences
from ragharness.cli import main
from ragharness.evaluation import load_gold, run_benchmark, to_json, to_markdown

DATA = Path(__file__).resolve().parents[1] / "datasets"


@pytest.fixture(scope="module")
def chunks():
    return load_corpus(DATA / "corpus")


@pytest.fixture(scope="module")
def retriever(chunks):
    return HybridRetriever(chunks)


# --- chunking --------------------------------------------------------------
def test_chunks_respect_sections_and_size():
    text = "# A\n" + " ".join(f"Sentence number {i} is here." for i in range(40)) + "\n# B\nShort section."
    out = chunk_markdown("doc", text, max_words=30)
    assert {c.section for c in out} == {"A", "B"}
    assert all(len(c.text.split()) <= 36 for c in out)  # window + one overlap sentence
    assert out[0].id == "doc#0" and len({c.id for c in out}) == len(out)


def test_chunk_overlap_carries_context():
    text = "# S\n" + " ".join(f"Fact {i} is stated clearly here." for i in range(12))
    out = chunk_markdown("d", text, max_words=20, overlap=1)
    last_of_first = split_sentences(out[0].text)[-1]
    assert out[1].text.startswith(last_of_first)


def test_chunker_rejects_tiny_windows():
    with pytest.raises(ValueError):
        chunk_markdown("d", "text", max_words=5)


def test_split_sentences_keeps_decimals():
    assert split_sentences("TLS 1.2 is required. AES-256 is used.") == ["TLS 1.2 is required.", "AES-256 is used."]
    assert split_sentences("   ") == []


# --- retrieval -------------------------------------------------------------
def test_hybrid_retriever_finds_the_right_section(retriever):
    hits = retriever.search("How long are contract records retained?", k=3)
    assert hits and hits[0].chunk.section == "Retention periods"


def test_retriever_returns_nothing_for_unrelated_query(retriever):
    assert retriever.search("zebra quantum banana") == []


def test_retriever_requires_chunks():
    with pytest.raises(ValueError):
        HybridRetriever([])


# --- auditor ---------------------------------------------------------------
def _chunk(cid, text):
    return Chunk(cid, "d", "s", text)


def test_auditor_classifies_grounded_miscited_and_ungrounded():
    p1 = _chunk("d#0", "Contract records are retained for 7 years after close-out.")
    p2 = _chunk("d#1", "Routine email is deleted after 3 years.")
    claims = [
        Claim("Contract records are retained for 7 years after close-out.", ["d#0"]),
        Claim("Routine email is deleted after 3 years.", ["d#0"]),
        Claim("Violations carry a mandatory fine of $250,000.", ["d#0"]),
    ]
    statuses = [v.status for v in AuditorAgent().audit(claims, [p1, p2])]
    assert statuses == ["grounded", "miscited", "ungrounded"]


def test_auditor_rejects_changed_numbers():
    p = _chunk("d#0", "Contract records are retained for 7 years.")
    assert AuditorAgent.support("Contract records are retained for 10 years.", p.text) == 0.0
    assert AuditorAgent.support("", p.text) == 0.0


def test_auditor_threshold_validation():
    with pytest.raises(ValueError):
        AuditorAgent(threshold=0)


# --- pipeline --------------------------------------------------------------
def test_pipeline_removes_hallucinations_and_fixes_citations(retriever):
    noisy = NoisyLLM(ExtractiveLLM(), hallucination_rate=1.0, miscite_rate=1.0, seed=1)
    brief = BriefPipeline(retriever, noisy).run("Who approves exceptions to the records retention policy?")
    assert brief.claims, "audited brief should keep supported claims"
    assert all(v.status == "grounded" for v in AuditorAgent().audit(brief.claims, brief.passages))
    assert not any(f in brief.to_markdown() for f in NoisyLLM.FABRICATIONS)
    assert any(line.startswith("revise") for line in brief.log)


def test_pipeline_without_audit_keeps_noise(retriever):
    noisy = NoisyLLM(ExtractiveLLM(), hallucination_rate=1.0, miscite_rate=0.0, seed=1)
    brief = BriefPipeline(retriever, noisy, audit=False).run("Who owns the compliance matrix?")
    assert any(c.text in NoisyLLM.FABRICATIONS for c in brief.claims)


def test_empty_brief_markdown():
    from ragharness.agents import Brief

    assert "No grounded answer" in Brief("q", [], [], [], [], 0.0).to_markdown()


# --- benchmark & cli ---------------------------------------------------------
def test_benchmark_shows_auditor_benefit(retriever):
    gold = load_gold(DATA / "gold_qa.json")
    res = run_benchmark({
        "no-audit": BriefPipeline(retriever, NoisyLLM(ExtractiveLLM(), 1.0, 0.5), audit=False),
        "audit": BriefPipeline(retriever, NoisyLLM(ExtractiveLLM(), 1.0, 0.5)),
    }, gold)
    no_audit, audit = res
    assert audit.hallucination_rate == 0 < no_audit.hallucination_rate
    assert audit.grounding_rate > no_audit.grounding_rate
    assert audit.fact_recall >= 0.8
    assert "| audit |" in to_markdown(res) and '"pipeline": "audit"' in to_json(res)


def test_cli_ask_and_bench(capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["ask", "What encryption is used for data at rest?"]) == 0
    assert "AES-256" in capsys.readouterr().out
    assert main(["bench"]) == 0
    assert "Hallucination rate" in capsys.readouterr().out
