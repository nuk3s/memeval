import pytest
from conftest import FakeAdapter

import memeval.cli as cli


def test_doctor_pass_exits_zero(monkeypatch, capsys):
    monkeypatch.setitem(cli.REGISTRY, "fake", FakeAdapter)
    with pytest.raises(SystemExit) as ex:
        cli.main(["doctor", "--adapter", "fake", "--ready-timeout", "3"])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    assert "retain" in out and "recall roundtrip" in out
    assert "3/3 sentinels visible" in out
    assert "supersede" in out and "unsupported" in out  # FakeAdapter doesn't implement it


def test_doctor_fails_on_broken_recall(monkeypatch, capsys):
    def broken():
        return FakeAdapter(fail_queries=("sentinel",))
    monkeypatch.setitem(cli.REGISTRY, "fake", broken)
    with pytest.raises(SystemExit) as ex:
        cli.main(["doctor", "--adapter", "fake", "--ready-timeout", "1"])
    assert ex.value.code == 1
    assert "FAIL" in capsys.readouterr().out


def test_doctor_partial_visibility_exits_one(monkeypatch, capsys):
    class DropThird(FakeAdapter):
        def retain(self, items):
            super().retain([i for i in items if not (i.get("id") or "").endswith("-3")])
    monkeypatch.setitem(cli.REGISTRY, "fake", DropThird)
    with pytest.raises(SystemExit) as ex:
        cli.main(["doctor", "--adapter", "fake", "--ready-timeout", "1"])
    assert ex.value.code == 1
    out = capsys.readouterr().out
    assert "PARTIAL" in out and "2/3" in out


def test_doctor_supersede_failure_gates_exit(monkeypatch, capsys):
    class BadSupersede(FakeAdapter):
        def supersede(self, doc_id, content, tags=None):
            raise RuntimeError("500 from server")
    monkeypatch.setitem(cli.REGISTRY, "fake", BadSupersede)
    with pytest.raises(SystemExit) as ex:
        cli.main(["doctor", "--adapter", "fake", "--ready-timeout", "3"])
    assert ex.value.code == 1
    assert "FAIL" in capsys.readouterr().out


def test_doctor_transient_recall_error_recovers(monkeypatch, capsys):
    class FlakyRecall(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def recall(self, query, k=10):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("blip")
            return super().recall(query, k)
    monkeypatch.setitem(cli.REGISTRY, "fake", FlakyRecall)
    with pytest.raises(SystemExit) as ex:
        cli.main(["doctor", "--adapter", "fake", "--ready-timeout", "5"])
    assert ex.value.code == 0
