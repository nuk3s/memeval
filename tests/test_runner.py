from conftest import FakeAdapter

from memeval.events import RunLog, read_events
from memeval.runner import Runner


def make(tmp_path, adapter, **kw):
    log = RunLog(tmp_path / "run.jsonl")
    return Runner(adapter, log, **kw), log


CORPUS = [
    {"id": "d1", "content": "the checkout service runs on port 8080 with six workers"},
    {"id": "d2", "content": "the user database is PostgreSQL sixteen on the db-main node"},
]


def test_retain_ok_emits_event(tmp_path):
    r, log = make(tmp_path, FakeAdapter())
    assert r.retain(CORPUS) is True
    ev = read_events(log.path)[0]
    assert ev["ev"] == "retain" and ev["ok"] is True and ev["n_items"] == 2
    assert ev["error"] is None


def test_retain_failure_is_captured(tmp_path):
    class Boom(FakeAdapter):
        def retain(self, items):
            raise RuntimeError("nope")
    r, log = make(tmp_path, Boom())
    assert r.retain(CORPUS) is False
    ev = read_events(log.path)[0]
    assert ev["ok"] is False and "nope" in ev["error"]


def test_wait_ready_polls_through_index_delay(tmp_path):
    r, log = make(tmp_path, FakeAdapter(index_delay=0.3), ready_timeout=5, poll_interval=0.1)
    r.retain(CORPUS)
    missing = r.wait_ready(CORPUS)
    assert missing == []
    ready = [e for e in read_events(log.path) if e["ev"] == "ready"][0]
    assert ready["verified_n"] == 2 and ready["missing"] == []


def test_wait_ready_reports_silent_write_loss(tmp_path):
    r, log = make(tmp_path, FakeAdapter(drop_ids={"d2"}), ready_timeout=1, poll_interval=0.2)
    r.retain(CORPUS)
    missing = r.wait_ready(CORPUS)
    assert missing == ["d2"]


def test_wait_ready_zero_timeout_still_polls_once(tmp_path):
    r, log = make(tmp_path, FakeAdapter(), ready_timeout=0)
    r.retain(CORPUS)
    assert r.wait_ready(CORPUS) == []


def test_wait_ready_sleep_clamped_to_budget(tmp_path):
    import time as _t
    r, log = make(tmp_path, FakeAdapter(drop_ids={"d1", "d2"}),
                  ready_timeout=0.3, poll_interval=5.0)
    r.retain(CORPUS)
    t0 = _t.monotonic()
    missing = r.wait_ready(CORPUS)
    assert _t.monotonic() - t0 < 2.0, "sleep must be clamped to the remaining timeout budget"
    assert missing == ["d1", "d2"]


def test_query_ranks_against_pattern(tmp_path):
    r, log = make(tmp_path, FakeAdapter())
    r.retain(CORPUS)
    row = r.query("what port does the checkout service use?", 5, phase="rank", pattern="8080")
    assert row["rank"] == 1 and row["error"] is None and row["ms"] >= 0
    ev = [e for e in read_events(log.path) if e["ev"] == "query"][0]
    assert ev["rank"] == 1 and ev["hits"][0]["text"]


def test_query_error_becomes_row_not_crash(tmp_path):
    r, log = make(tmp_path, FakeAdapter(fail_queries=("port",)), retries=1)
    r.retain(CORPUS)
    row = r.query("what port does the checkout service use?", 5, phase="rank", pattern="8080")
    assert row["rank"] is None and "injected failure" in row["error"]


def test_query_bad_pattern_becomes_error_row(tmp_path):
    r, log = make(tmp_path, FakeAdapter())
    r.retain(CORPUS)
    row = r.query("what port does the checkout service use?", 5, phase="rank", pattern="8080(")
    assert row["rank"] is None and "bad answer pattern" in row["error"]


def test_query_retries_zero_still_calls_once(tmp_path):
    r, log = make(tmp_path, FakeAdapter(), retries=0)
    r.retain(CORPUS)
    row = r.query("what port does the checkout service use?", 5, phase="rank", pattern="8080")
    assert row["rank"] == 1 and row["error"] is None


def test_query_retry_recovers(tmp_path):
    class FlakyOnce(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def recall(self, query, k=10):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient")
            return super().recall(query, k)
    r, log = make(tmp_path, FlakyOnce(), retries=1)
    r.retain(CORPUS)
    row = r.query("what port does the checkout service use?", 5, phase="rank", pattern="8080")
    assert row["error"] is None and row["rank"] == 1


def test_hit_text_truncated_in_event_but_full_in_row(tmp_path):
    a = FakeAdapter()
    long = "checkout " + "x" * 500
    a.retain([{"content": long}])
    r, log = make(tmp_path, a)
    row = r.query("checkout", 5, phase="rank")
    assert len(row["hits"][0]["text"]) > 300           # full text for in-process consumers
    ev = [e for e in read_events(log.path) if e["ev"] == "query"][0]
    assert len(ev["hits"][0]["text"]) == 300           # truncated in the log


def test_query_survives_poisoned_hits(tmp_path):
    class Poisoned(FakeAdapter):
        def recall(self, query, k=10):
            return [{"text": 123, "score": None}, "garbage", {"score": 0.5}]
    r, log = make(tmp_path, Poisoned())
    r.retain(CORPUS)
    row = r.query("anything", 5, phase="rank", pattern="123")
    assert row["error"] is None and row["rank"] == 1  # str(123) == "123" matched
    assert row["hits"][1]["text"] == ""  # {"score": 0.5} entry normalized, "garbage" dropped


def test_normalize_hits_coerces_score(tmp_path):
    class WeirdScore(FakeAdapter):
        def recall(self, query, k=10):
            return [{"text": "checkout on port 8080", "score": "0.9"},
                    {"text": "other", "score": {"weird": 1}}]
    r, log = make(tmp_path, WeirdScore())
    r.retain(CORPUS)
    row = r.query("checkout", 5, phase="rank")
    assert row["hits"][0]["score"] == 0.9
    assert row["hits"][1]["score"] is None
