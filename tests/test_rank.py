import pytest
from conftest import FakeAdapter

from memeval.events import RunLog, read_events
from memeval.rank import run_rank, summarize
from memeval.runner import Runner

CORPUS = [
    {"id": "d1", "content": "the checkout service runs on port 8080 with six workers"},
    {"id": "d2", "content": "the user database is PostgreSQL sixteen on the db-main node"},
    {"id": "d3", "content": "the cache layer switched from Redis to Memcached in March"},
]
GOLD = [
    {"query": "what port does the checkout service use?", "answer": "8080"},
    {"query": "which database holds the users?", "answer": "PostgreSQL|db-main"},
    {"query": "what cache is in use now?", "answer": "Memcached"},
]


def run(tmp_path, adapter, **kw):
    log = RunLog(tmp_path / "run.jsonl")
    runner = Runner(adapter, log, poll_interval=0.05, ready_timeout=3)
    s = run_rank(runner, CORPUS, GOLD, **kw)
    log.close()
    return s, read_events(log.path)


def test_perfect_rank_run(tmp_path):
    s, evs = run(tmp_path, FakeAdapter())
    assert s["n"] == 3 and s["hit@1"] == 3 and s["mrr"] == 1.0 and s["errors"] == 0
    kinds = [e["ev"] for e in evs]
    assert kinds[0] == "retain" and "ready" in kinds and kinds[-1] == "summary"
    assert kinds.count("query") == 3


def test_error_row_counts_not_crashes(tmp_path):
    s, evs = run(tmp_path, FakeAdapter(fail_queries=("cache",)))
    assert s["errors"] == 1 and s["n"] == 3
    assert s["hit@1"] == 2  # the two healthy queries still score


def test_repeat_produces_per_rep_blocks(tmp_path):
    s, evs = run(tmp_path, FakeAdapter(), repeat=2)
    assert s["n"] == 6 and set(s["per_rep"]) == {"1", "2"}
    assert s["per_rep"]["1"]["hit@1"] == 3
    assert [e for e in evs if e["ev"] == "query" and e["rep"] == 2]


def test_aborted_retain_returns_none(tmp_path):
    class Boom(FakeAdapter):
        def retain(self, items):
            raise RuntimeError("dead backend")
    s, evs = run(tmp_path, Boom())
    assert s is None
    assert evs[-1]["ev"] == "retain" and evs[-1]["ok"] is False


def test_bare_exception_still_counts_as_error(tmp_path):
    class BareFail(FakeAdapter):
        def recall(self, query, k=10):
            raise ConnectionError()  # stringifies to ""
    s, evs = run(tmp_path, BareFail())
    assert s["errors"] == 3 and s["hit@k"] == 0
    qevs = [e for e in evs if e["ev"] == "query"]
    assert all(e["error"] for e in qevs), "error message must never be empty"


def test_summarize_percentiles():
    rows = [{"phase": "rank", "rep": 1, "query": str(i), "rank": 1,
             "ms": ms, "hits": [], "error": None}
            for i, ms in enumerate([100, 110, 120, 130, 140, 150, 160, 170, 180, 1000])]
    s = summarize(rows, k=10)
    assert s["p50_ms"] == 145
    assert s["p95_ms"] == 1000
    assert s["hit@1"] == 10


def test_load_gold_and_corpus_reject_malformed_rows(tmp_path):
    from memeval.rank import load_corpus, load_gold
    p = tmp_path / "rows.jsonl"
    p.write_text('{"query": "q1", "answer": "a"}\n{"query": "q2"}\n')
    with pytest.raises(ValueError, match="gold row 2: missing or empty 'answer'"):
        load_gold(p)
    p.write_text('{"id": "d1", "content": "   "}\n')
    with pytest.raises(ValueError, match="corpus row 1: missing or empty 'content'"):
        load_corpus(p)
    p.write_text('"just a string"\n')
    with pytest.raises(ValueError, match="expected a JSON object, got str"):
        load_corpus(p)


def test_load_gold_keeps_bad_regex_as_runtime_error_row(tmp_path):
    # README contract: a malformed answer regex is an error ROW, not a rejected file
    from memeval.rank import load_gold
    p = tmp_path / "g.jsonl"
    p.write_text('{"query": "q1", "answer": "8080("}\n')
    assert load_gold(p)[0]["answer"] == "8080("


def test_load_jsonl_reports_line_number(tmp_path):
    from memeval.rank import load_jsonl
    p = tmp_path / "c.jsonl"
    p.write_text('{"a": 1}\n\n{broken\n')
    with pytest.raises(ValueError, match="line 3"):
        load_jsonl(p)
