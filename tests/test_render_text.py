from memeval.render.text import bar, format_ab, format_adversarial, format_summary

SUMMARY = {"n": 8, "k": 10, "hit@1": 7, "hit@3": 8, "hit@k": 8,
           "mrr": 0.906, "p50_ms": 148, "p95_ms": 310, "errors": 0}


def test_bar_full_and_empty():
    assert bar(8, 8) == "▇" * 8
    assert bar(0, 8) == "░" * 8
    assert bar(0, 0) == "░" * 8      # zero denominator must not divide


def test_format_summary_contains_metrics_and_bars():
    out = format_summary(SUMMARY, label="hindsight")
    assert "[hindsight]" in out and "7/8" in out and "MRR 0.906" in out
    assert "▇" in out and "p95 310ms" in out


def test_format_summary_per_rep():
    s = dict(SUMMARY)
    s["per_rep"] = {"1": {"n": 4, "hit@1": 4, "hit@3": 4, "hit@k": 4, "mrr": 1.0},
                    "2": {"n": 4, "hit@1": 3, "hit@3": 4, "hit@k": 4, "mrr": 0.875}}
    out = format_summary(s)
    assert "rep 1" in out and "rep 2" in out


def test_format_ab_delta_line():
    a = dict(SUMMARY)
    b = dict(SUMMARY, **{"hit@1": 5, "mrr": 0.706, "p50_ms": 200})
    out = format_ab("hindsight", a, "mem0", b)
    assert "Δ" in out and "+2" in out and "+0.200" in out and "-52ms" in out


def test_format_adversarial_verdict_lines():
    result = {
        "before": [{"phase": "adv_before", "query": "q1", "fabricated": ["media.*8080"],
                    "expected_present": [], "top3": ["the media service runs on port 8080"],
                    "error": None}],
        "after_corrections": [{"phase": "adv_after", "query": "q1", "fabricated": [],
                               "expected_present": [], "top3": ["corrected"], "error": None}],
        "corrections_failed": False,
    }
    out = format_adversarial(result)
    assert "FABRICATED" in out and "self-heal: yes" in out


def test_format_adversarial_error_state():
    result = {
        "before": [{"phase": "adv_before", "query": "q1", "fabricated": [],
                    "expected_present": [], "top3": [], "error": "ConnectionError"}],
        "after_corrections": None,
        "corrections_failed": False,
    }
    out = format_adversarial(result)
    assert "[ERROR]" in out and "ConnectionError" in out
    assert "clean" not in out  # an errored check must not read as clean


def test_format_adversarial_corrections_failed():
    result = {
        "before": [{"phase": "adv_before", "query": "q1", "fabricated": ["x"],
                    "expected_present": [], "top3": [], "error": None}],
        "after_corrections": None,
        "corrections_failed": True,
    }
    out = format_adversarial(result)
    assert "corrections retain FAILED" in out


def test_format_adversarial_after_errors_mean_unknown_heal():
    result = {
        "before": [{"phase": "adv_before", "query": "q1", "fabricated": ["x"],
                    "expected_present": [], "top3": [], "error": None}],
        "after_corrections": [{"phase": "adv_after", "query": "q1", "fabricated": [],
                               "expected_present": [], "top3": [], "error": "timeout"}],
        "corrections_failed": False,
    }
    out = format_adversarial(result)
    assert "self-heal: UNKNOWN" in out
