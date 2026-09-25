import json

import pytest

pytest.importorskip("matplotlib")

from conftest import FakeAdapter  # noqa: E402

import memeval.cli as cli  # noqa: E402

CORPUS = [{"id": "d1", "content": "the checkout service runs on port 8080 with six workers"},
          {"id": "d2", "content": "the user database is PostgreSQL sixteen on the db-main node"}]
GOLD = [{"query": "what port does the checkout service use?", "answer": "8080"},
        {"query": "which database holds the users?", "answer": "PostgreSQL|db-main"}]


@pytest.fixture
def run_log(tmp_path, monkeypatch):
    monkeypatch.setitem(cli.REGISTRY, "fake", FakeAdapter)
    corpus, gold = tmp_path / "c.jsonl", tmp_path / "g.jsonl"
    corpus.write_text("\n".join(json.dumps(c) for c in CORPUS))
    gold.write_text("\n".join(json.dumps(g) for g in GOLD))
    runs = tmp_path / "runs"
    cli.main(["rank", "--adapter", "fake", "--corpus", str(corpus),
              "--gold", str(gold), "--runs-dir", str(runs)])
    return next(runs.glob("*.jsonl"))


def test_single_run_chart_svg(run_log, tmp_path):
    from memeval.render.charts import render_chart
    out = render_chart(str(run_log), out=str(tmp_path / "c.svg"))
    text = (tmp_path / "c.svg").read_text()
    assert "<svg" in text and out.endswith("c.svg")


def test_default_out_path_under_charts_dir(run_log, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from memeval.render.charts import render_chart
    out = render_chart(str(run_log))
    assert out == f"charts/{run_log.stem}.svg"


def test_ab_chart(run_log, tmp_path):
    from memeval.render.charts import render_ab_chart
    render_ab_chart(str(run_log), str(run_log), out=str(tmp_path / "ab.svg"))
    assert "<svg" in (tmp_path / "ab.svg").read_text()


def test_ab_chart_single_legend_outside(run_log, tmp_path):
    from memeval.render.charts import render_ab_chart
    render_ab_chart(str(run_log), str(run_log), out=str(tmp_path / "ab.svg"))
    svg = (tmp_path / "ab.svg").read_text()
    assert 'id="legend_1"' in svg
    assert 'id="legend_2"' not in svg  # one shared figure legend, not one per panel


def test_cli_chart_command(run_log, tmp_path, capsys):
    cli.main(["chart", str(run_log), "-o", str(tmp_path / "x.svg")])
    assert "wrote" in capsys.readouterr().out


def test_chart_rejects_adversarial_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(cli.REGISTRY, "fake", FakeAdapter)
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({
        "corpus": [{"content": "the checkout service runs on port 8080"}],
        "checks": [{"query": "what runs on port 8080?", "forbidden": [], "expected": []}],
        "consolidate_wait": 0}))
    runs = tmp_path / "runs"
    cli.main(["adversarial", "--adapter", "fake", "--probe", str(probe),
              "--runs-dir", str(runs)])
    capsys.readouterr()
    log = next(runs.glob("*_adversarial.jsonl"))
    from memeval.render.charts import render_chart
    with pytest.raises(ValueError, match="adversarial"):
        render_chart(str(log))
    with pytest.raises(SystemExit):
        cli.main(["chart", str(log)])


def test_rank_bucket_counts_correct(tmp_path, monkeypatch):
    monkeypatch.setitem(cli.REGISTRY, "fake",
                        lambda: FakeAdapter(fail_queries=("database",)))
    corpus = tmp_path / "c.jsonl"
    gold = tmp_path / "g.jsonl"
    corpus.write_text("\n".join(json.dumps(c) for c in CORPUS))
    gold.write_text("\n".join(json.dumps(g) for g in [
        GOLD[0],                                                  # hits rank 1
        {"query": "which database holds the users?", "answer": "PostgreSQL"},  # errors
        {"query": "what is the flurble quantum?", "answer": "zzz"},            # misses
    ]))
    runs = tmp_path / "runs"
    cli.main(["rank", "--adapter", "fake", "--corpus", str(corpus),
              "--gold", str(gold), "--runs-dir", str(runs), "--ready-timeout", "1"])
    from memeval.render.charts import _load, _rank_counts
    _, queries, _ = _load(str(next(runs.glob("*.jsonl"))))
    buckets, counts = _rank_counts(queries, 10)
    assert buckets == ["1", "2", "3", ">3", "miss", "error"]
    assert counts == [1, 0, 0, 0, 1, 1]


def test_chart_all_errors_still_renders(tmp_path, monkeypatch):
    monkeypatch.setitem(cli.REGISTRY, "fake",
                        lambda: FakeAdapter(fail_queries=("the",)))
    corpus = tmp_path / "c.jsonl"
    gold = tmp_path / "g.jsonl"
    corpus.write_text("\n".join(json.dumps(c) for c in CORPUS))
    gold.write_text(json.dumps(GOLD[0]))
    runs = tmp_path / "runs"
    cli.main(["rank", "--adapter", "fake", "--corpus", str(corpus),
              "--gold", str(gold), "--runs-dir", str(runs), "--ready-timeout", "1"])
    from memeval.render.charts import render_chart
    render_chart(str(next(runs.glob("*.jsonl"))), out=str(tmp_path / "e.svg"))
    assert "<svg" in (tmp_path / "e.svg").read_text()


def test_chart_missing_file_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit) as ex:
        cli.main(["chart", str(tmp_path / "nope.jsonl")])
    assert "chart failed" in str(ex.value) and "nope.jsonl" in str(ex.value)


def test_chart_rejects_run_and_ab_together(run_log):
    with pytest.raises(SystemExit) as ex:
        cli.main(["chart", str(run_log), "--ab", str(run_log), str(run_log)])
    assert "not both" in str(ex.value)
