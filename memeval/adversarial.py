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
        return json.load(f)


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
