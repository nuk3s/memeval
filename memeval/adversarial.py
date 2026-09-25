"""Adversarial eval: retain facts that invite a bad merge, consolidate, and check
whether recall asserts a statement no source made — then whether corrections clear it."""
import json
import re
import time

# A correction restates the false claim in order to deny it, so a bare regex can't
# tell a fabrication from its own correction. Count a forbidden pattern only when it
# matches an AFFIRMATIVE statement (no negation cue). Heuristic: a fabrication that
# happens to contain "not" elsewhere slips past. Documented in the README.
_NEG = re.compile(
    r"\b(not|no|never|false|isn'?t|aren'?t|doesn'?t|don'?t|without|lacks|nor)\b", re.I)


def load_probe(path):
    with open(path, encoding="utf-8") as f:
        return validate_probe(json.load(f))


def _nonempty_str(v):
    return isinstance(v, str) and bool(v.strip())


def validate_probe(probe):
    """Shape-check the probe at load time, so a typo fails before retain writes into
    the store rather than as a KeyError or re.error halfway through the run."""
    if not isinstance(probe, dict):
        raise ValueError("probe must be a JSON object")
    corpus = probe.get("corpus")
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("probe 'corpus' must be a non-empty list")
    for n, d in enumerate(corpus, 1):
        if not isinstance(d, dict) or not _nonempty_str(d.get("content")):
            raise ValueError(f"probe corpus item {n}: missing or empty 'content'")
    checks = probe.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("probe 'checks' must be a non-empty list")
    for n, c in enumerate(checks, 1):
        if not isinstance(c, dict) or not _nonempty_str(c.get("query")):
            raise ValueError(f"probe check {n}: missing or empty 'query'")
        for key in ("forbidden", "expected"):
            pats = c.get(key, [])
            if not isinstance(pats, list) or not all(isinstance(p, str) for p in pats):
                raise ValueError(f"probe check {n}: {key!r} must be a list of regex strings")
            for p in pats:
                try:
                    re.compile(p)
                except re.error as e:
                    raise ValueError(f"probe check {n}: bad {key} regex {p!r}: {e}") from None
    corrections = probe.get("corrections", [])
    if not isinstance(corrections, list) or not all(isinstance(x, str) for x in corrections):
        raise ValueError("probe 'corrections' must be a list of strings")
    wait = probe.get("consolidate_wait", 2)
    if isinstance(wait, bool) or not isinstance(wait, (int, float)) or wait < 0:
        raise ValueError("probe 'consolidate_wait' must be a non-negative number of seconds")
    return probe


def _check(runner, checks, phase):
    results = []
    for c in checks:
        row = runner.query(c["query"], 10, phase=phase)
        texts = [h.get("text", "") for h in row["hits"]]  # FULL texts (event is truncated)
        affirmative = [t for t in texts if not _NEG.search(t)]
        fabricated = [f for f in c.get("forbidden", [])
                      if any(re.search(f, t, re.I) for t in affirmative)]
        present = [e for e in c.get("expected", [])
                   if any(re.search(e, t, re.I) for t in texts)]
        res = {"phase": phase, "query": c["query"], "fabricated": fabricated,
               "expected_present": present, "top3": [t[:100] for t in texts[:3]],
               "error": row["error"]}
        runner.log.emit("verdict", **res)
        results.append(res)
    return results


def run_adversarial(runner, probe, consolidate=True):
    """Returns {"before": [...], "after_corrections": [...]|None, "corrections_failed": bool},
    or None on aborted retain."""
    wait = probe.get("consolidate_wait", 2)
    if not runner.retain(probe["corpus"]):
        return None
    if consolidate:
        runner.consolidate()
        time.sleep(wait)
    runner.wait_ready(probe["corpus"])
    before = _check(runner, probe["checks"], "adv_before")

    after = None
    corrections_failed = False
    if probe.get("corrections"):
        if runner.retain([{"content": c} for c in probe["corrections"]]):
            if consolidate:
                runner.consolidate()
                time.sleep(wait)
            after = _check(runner, probe["checks"], "adv_after")
        else:
            corrections_failed = True

    runner.log.emit(
        "summary", kind="adversarial",
        fabricated_before=sum(1 for c in before if c["fabricated"]),
        fabricated_after=(sum(1 for c in after if c["fabricated"]) if after is not None else None),
        errors_before=sum(1 for c in before if c["error"]),
        errors_after=(sum(1 for c in after if c["error"]) if after is not None else None),
        corrections_failed=corrections_failed)
    return {"before": before, "after_corrections": after, "corrections_failed": corrections_failed}
