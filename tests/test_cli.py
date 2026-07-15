import json

import pytest
from conftest import FakeAdapter

import memeval.cli as cli
from memeval.events import read_events

CORPUS = [
    {"id": "d1", "content": "the checkout service runs on port 8080 with six workers"},
    {"id": "d2", "content": "the user database is PostgreSQL sixteen on the db-main node"},
]
GOLD = [
    {"query": "what port does the checkout service use?", "answer": "8080"},
    {"query": "which database holds the users?", "answer": "PostgreSQL|db-main"},
]


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setitem(cli.REGISTRY, "fake", FakeAdapter)
    return "fake"


@pytest.fixture
def data(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    gold = tmp_path / "gold.jsonl"
    corpus.write_text("\n".join(json.dumps(c) for c in CORPUS))
    gold.write_text("\n".join(json.dumps(g) for g in GOLD))
    return corpus, gold


def test_rank_end_to_end_writes_run_log(tmp_path, capsys, fake_registry, data):
    corpus, gold = data
    runs = tmp_path / "runs"
    cli.main(["rank", "--adapter", "fake", "--corpus", str(corpus),
              "--gold", str(gold), "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert "hit@1" in out and "run log:" in out
    logs = list(runs.glob("*_fake_rank.jsonl"))
    assert len(logs) == 1
    evs = read_events(logs[0])
    assert evs[0]["ev"] == "run_meta" and evs[0]["adapter"] == "fake"
    assert evs[0]["schema"] == 1 and "corpus_sha256" in evs[0]
    assert evs[-1]["ev"] == "summary"


def test_ab_prints_delta(tmp_path, capsys, fake_registry, data):
    corpus, gold = data
    cli.main(["ab", "--a", "fake", "--b", "fake", "--corpus", str(corpus),
              "--gold", str(gold), "--runs-dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert "Δ" in out
    assert len(list((tmp_path / "runs").glob("*_ab-leg*.jsonl"))) == 2


def test_runs_listing(tmp_path, capsys, fake_registry, data):
    corpus, gold = data
    runs = tmp_path / "runs"
    cli.main(["rank", "--adapter", "fake", "--corpus", str(corpus),
              "--gold", str(gold), "--runs-dir", str(runs)])
    capsys.readouterr()
    cli.main(["runs", "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert "fake" in out and "rank" in out and "hit@1" in out


def test_runs_listing_tolerates_foreign_files(tmp_path, capsys, fake_registry, data):
    corpus, gold = data
    runs = tmp_path / "runs"
    cli.main(["rank", "--adapter", "fake", "--corpus", str(corpus),
              "--gold", str(gold), "--runs-dir", str(runs)])
    capsys.readouterr()
    (runs / "foreign.jsonl").write_text('{"some": "other schema"}\n')
    cli.main(["runs", "--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert "fake" in out and "hit@1" in out  # real run still listed, no crash


def test_bad_gold_file_exits_cleanly_without_run_log(tmp_path, fake_registry, data):
    corpus, _ = data
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json at all\n")
    runs = tmp_path / "runs"
    with pytest.raises(SystemExit) as ex:
        cli.main(["rank", "--adapter", "fake", "--corpus", str(corpus),
                  "--gold", str(bad), "--runs-dir", str(runs)])
    assert "cannot load gold" in str(ex.value)
    assert not list(runs.glob("*.jsonl")) if runs.is_dir() else True  # no orphan log


def test_adapters_listing(capsys):
    cli.main(["adapters"])
    out = capsys.readouterr().out
    assert "hindsight" in out and "tested" in out and "example-rest" in out


def test_unknown_adapter_exits(tmp_path, data):
    corpus, gold = data
    with pytest.raises(SystemExit):
        cli.main(["rank", "--adapter", "nope", "--corpus", str(corpus),
                  "--gold", str(gold), "--runs-dir", str(tmp_path / "runs")])
