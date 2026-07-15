"""Rank eval: load a corpus, run gold queries, record where the right fact lands."""
import json
import statistics


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# nearest-rank with round(); differs from numpy's linear interpolation —
# chosen for stdlib simplicity
def _p95(latencies):
    if not latencies:
        return 0
    s = sorted(latencies)
    return round(s[min(len(s) - 1, round(0.95 * (len(s) - 1)))])


def _block(rows):
    m = len(rows)
    ranked = [r["rank"] for r in rows if r["rank"]]
    return {
        "n": m,
        "hit@1": sum(1 for x in ranked if x == 1),
        "hit@3": sum(1 for x in ranked if x <= 3),
        "hit@k": len(ranked),
        "mrr": round(sum(1 / x for x in ranked) / m, 3) if m else 0.0,
    }


def summarize(rows, k):
    latencies = [r["ms"] for r in rows if r["error"] is None]
    s = _block(rows)
    s["k"] = k
    s["p50_ms"] = round(statistics.median(latencies)) if latencies else 0
    s["p95_ms"] = _p95(latencies)
    s["errors"] = sum(1 for r in rows if r["error"] is not None)
    reps = {}
    for r in rows:
        reps.setdefault(r["rep"], []).append(r)
    if len(reps) > 1:
        s["per_rep"] = {str(rep): _block(rs) for rep, rs in sorted(reps.items())}
    return s


def run_rank(runner, corpus, gold, k=10, repeat=1):
    """Returns the summary dict, or None if retain failed (aborted run)."""
    if not runner.retain(corpus):
        return None
    runner.wait_ready(corpus)
    rows = []
    for rep in range(1, repeat + 1):
        if repeat > 1:
            runner.echo(f"— repeat {rep}/{repeat}")
        for g in gold:
            rows.append(runner.query(g["query"], k, phase="rank",
                                     rep=rep, pattern=g["answer"]))
    s = summarize(rows, k)
    # **s keys must not collide with 'ev'/'kind'; emit() fails loud on collision
    runner.log.emit("summary", kind="rank", **s)
    return s
