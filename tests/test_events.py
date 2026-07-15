import json

from memeval.events import RunLog, file_sha256, read_events, redact, run_path


def test_runlog_roundtrip(tmp_path):
    p = tmp_path / "run.jsonl"
    log = RunLog(p)
    log.emit("run_meta", run_id="r1", kind="rank")
    log.emit("query", query="q", rank=1, ms=12)
    log.close()
    evs = read_events(p)
    assert [e["ev"] for e in evs] == ["run_meta", "query"]
    assert evs[1]["rank"] == 1


def test_redact_drops_secret_values():
    cfg = {"url": "http://x", "token": "s3cret", "api_key": "k", "bank": "b", "password": "p"}
    r = redact(cfg)
    assert r["url"] == "http://x" and r["bank"] == "b"
    assert r["token"] == "***" and r["api_key"] == "***" and r["password"] == "***"


def test_redact_keeps_empty_secret_as_is():
    assert redact({"token": ""})["token"] == ""


def test_redact_recurses_into_nested_dicts():
    r = redact({"auth": {"token": "s3cret", "region": "us"}, "url": "http://x"})
    assert r["auth"]["token"] == "***" and r["auth"]["region"] == "us"


def test_run_path_shape(tmp_path):
    p = run_path(tmp_path / "runs", "hindsight", "rank")
    assert p.parent.is_dir()
    name = p.name
    assert name.endswith("_hindsight_rank.jsonl") and name[:4].isdigit()


def test_run_path_collision_uniquified(tmp_path):
    p1 = run_path(tmp_path / "runs", "fake", "ab-leg")
    p1.touch()  # RunLog creates the file on open; simulate leg A already writing
    p2 = run_path(tmp_path / "runs", "fake", "ab-leg")
    assert p1 != p2 and p2.name.endswith("-2.jsonl")


def test_file_sha256(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text("hello\n")
    digest = file_sha256(f)
    assert len(digest) == 64
    assert digest == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"


def test_events_are_one_json_per_line(tmp_path):
    p = tmp_path / "run.jsonl"
    log = RunLog(p)
    log.emit("run_meta", note="a\nb")  # newline inside a field must stay escaped
    log.close()
    lines = p.read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["note"] == "a\nb"


def test_read_events_skips_truncated_line(tmp_path):
    p = tmp_path / "run.jsonl"
    p.write_text('{"ev": "run_meta", "run_id": "r1"}\n{"ev": "query", "ra')
    evs = read_events(p)
    assert len(evs) == 1 and evs[0]["run_id"] == "r1"


def test_runlog_context_manager(tmp_path):
    p = tmp_path / "run.jsonl"
    with RunLog(p) as log:
        log.emit("run_meta", run_id="r1")
    assert read_events(p) == [{"ev": "run_meta", "run_id": "r1"}]
