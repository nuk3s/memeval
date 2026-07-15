import re
from pathlib import Path

from conftest import FakeAdapter

from memeval.events import RunLog
from memeval.rank import load_jsonl, run_rank
from memeval.runner import Runner

EX = Path(__file__).resolve().parent.parent / "examples"


def _corpus():
    return load_jsonl(EX / "corpus.jsonl")


def _gold():
    return load_jsonl(EX / "gold.jsonl")


def test_corpus_shape_and_size():
    docs = _corpus()
    assert len(docs) >= 110
    ids = [d["id"] for d in docs]
    assert len(set(ids)) == len(ids), "duplicate ids"
    assert all(d["content"].strip() for d in docs)


def test_gold_shape_and_size():
    g = _gold()
    assert len(g) >= 100
    queries = [x["query"] for x in g]
    assert len(set(queries)) == len(queries), "duplicate queries"
    for x in g:
        re.compile(x["answer"])


def test_every_answer_matches_a_doc():
    docs = _corpus()
    for x in _gold():
        pat = re.compile(x["answer"], re.I)
        assert any(pat.search(d["content"]) for d in docs), \
            f"gold answer matches no corpus doc: {x['query']!r}"


# the 6 designed stale pairs are the only sanctioned 2-doc matches
STALE_PAIR_QUERIES = {
    "what does Northwind use for its session cache after the switch?",
    "which search engine does Northwind run today?",
    "where does media and artifact storage live now?",
    "after Jenkins was retired, what runs the pipelines?",
    "what does monitoring run on since the Nagios retirement?",
    "what is the anonymous rate limit per minute these days?",
}


def test_answer_collisions_are_only_stale_pairs():
    docs = _corpus()
    for x in _gold():
        pat = re.compile(x["answer"], re.I)
        matches = [d["id"] for d in docs if pat.search(d["content"])]
        allowed = 2 if x["query"] in STALE_PAIR_QUERIES else 1
        assert len(matches) <= allowed, \
            f"answer regex too loose ({len(matches)} docs): {x['query']!r} -> {matches}"


def test_fake_adapter_floor(tmp_path):
    # word-overlap floor: catches gold items lexically unmoored from their doc.
    # A real embedding retriever does far better; this is a smoke floor, not a target.
    log = RunLog(tmp_path / "run.jsonl")
    runner = Runner(FakeAdapter(), log, poll_interval=0.05, ready_timeout=5)
    s = run_rank(runner, _corpus(), _gold(), k=10)
    log.close()
    assert s is not None and s["errors"] == 0 and s["n"] == len(_gold())
    assert s["hit@k"] >= int(0.5 * s["n"]), s
