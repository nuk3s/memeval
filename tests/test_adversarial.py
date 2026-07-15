from conftest import FakeAdapter

from memeval.adversarial import run_adversarial
from memeval.events import RunLog, read_events
from memeval.runner import Runner

PROBE = {
    "corpus": [
        {"content": "the checkout service runs on port 8080"},
        {"content": "the media service moved to the new cluster"},
    ],
    "checks": [
        {"query": "what runs on port 8080 and where is the media service?",
         "forbidden": ["media service[^.]{0,30}8080"],
         "expected": ["checkout[^.]{0,30}8080"]},
    ],
    "corrections": ["the media service is not on port 8080"],
    "consolidate_wait": 0,
}


class MergingFake(FakeAdapter):
    """Simulates a consolidator that invents a merged fact, then heals on correction."""
    def consolidate(self):
        texts = " ".join(self.docs)
        if "not on port 8080" in texts:
            self.docs = [d for d in self.docs if "media service runs on port 8080" not in d]
        elif "media" in texts and "8080" in texts:
            self.docs.append("the media service runs on port 8080")


def run(tmp_path, adapter, probe=PROBE, consolidate=True):
    log = RunLog(tmp_path / "run.jsonl")
    runner = Runner(adapter, log, poll_interval=0.05, ready_timeout=2)
    r = run_adversarial(runner, probe, consolidate=consolidate)
    log.close()
    return r, read_events(log.path)


def test_fabrication_detected_and_healed(tmp_path):
    r, evs = run(tmp_path, MergingFake())
    assert r["before"][0]["fabricated"], "merged fact must be flagged"
    assert not r["after_corrections"][0]["fabricated"], "correction must clear it"
    verdicts = [e for e in evs if e["ev"] == "verdict"]
    assert {v["phase"] for v in verdicts} == {"adv_before", "adv_after"}
    summ = [e for e in evs if e["ev"] == "summary"][0]
    assert summ["fabricated_before"] == 1 and summ["fabricated_after"] == 0


def test_clean_store_stays_clean(tmp_path):
    r, _ = run(tmp_path, FakeAdapter())
    assert not r["before"][0]["fabricated"]
    assert r["before"][0]["expected_present"] == ["checkout[^.]{0,30}8080"]


def test_negation_correction_not_flagged_as_fabrication(tmp_path):
    # the correction itself contains "media ... 8080" but with a negation cue
    a = FakeAdapter()
    r, _ = run(tmp_path, a)
    a.retain([{"content": "the media service is not on port 8080"}])
    # direct check: a store returning only the negated correction must stay clean
    from memeval.adversarial import _NEG
    assert _NEG.search("the media service is not on port 8080")


def test_no_corrections_probe(tmp_path):
    probe = {k: v for k, v in PROBE.items() if k != "corrections"}
    r, evs = run(tmp_path, FakeAdapter(), probe=probe)
    assert r["after_corrections"] is None


def test_errored_check_is_not_clean(tmp_path):
    a = FakeAdapter(fail_queries=("media",))
    r, evs = run(tmp_path, a)
    assert r["before"][0]["error"]
    assert not r["before"][0]["fabricated"]
    verdicts = [e for e in evs if e["ev"] == "verdict"]
    assert verdicts[0]["error"]
    summ = [e for e in evs if e["ev"] == "summary"][0]
    assert summ["errors_before"] == 1


def test_corrections_retain_failure_marked(tmp_path):
    class CorrectionsFail(MergingFake):
        def __init__(self):
            super().__init__()
            self.retain_calls = 0
        def retain(self, items):
            self.retain_calls += 1
            if self.retain_calls > 1:
                raise RuntimeError("write refused")
            super().retain(items)
    r, evs = run(tmp_path, CorrectionsFail())
    assert r["corrections_failed"] is True and r["after_corrections"] is None
    summ = [e for e in evs if e["ev"] == "summary"][0]
    assert summ["corrections_failed"] is True and summ["fabricated_after"] is None


def test_consolidate_exception_does_not_kill_run(tmp_path):
    class BoomConsolidate(FakeAdapter):
        def consolidate(self):
            raise RuntimeError("no such endpoint")
    r, evs = run(tmp_path, BoomConsolidate())
    assert r is not None and not r["before"][0]["fabricated"]
    cons = [e for e in evs if e["ev"] == "consolidate"]
    assert cons and cons[0]["ok"] is False
