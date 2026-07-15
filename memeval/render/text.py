"""Plain-text presentation. Stdlib only — always available."""


def bar(num, den, width=8):
    if den <= 0:
        return "░" * width
    filled = round(width * num / den)
    return "▇" * filled + "░" * (width - filled)


def format_summary(s, label=""):
    head = f"rank eval{f' [{label}]' if label else ''}"
    lines = [
        head,
        f"  hit@1 {bar(s['hit@1'], s['n'])} {s['hit@1']}/{s['n']}",
        f"  hit@3 {bar(s['hit@3'], s['n'])} {s['hit@3']}/{s['n']}",
        f"  hit@k {bar(s['hit@k'], s['n'])} {s['hit@k']}/{s['n']}  (k={s['k']})",
        f"  MRR {s['mrr']}   p50 {s['p50_ms']}ms   p95 {s['p95_ms']}ms   errors {s['errors']}",
    ]
    for rep, b in s.get("per_rep", {}).items():
        lines.append(f"    rep {rep}: hit@1 {b['hit@1']}/{b['n']}  MRR {b['mrr']}")
    return "\n".join(lines)


def format_ab(label_a, a, label_b, b):
    return (f"  Δ ({label_a} − {label_b}): "
            f"hit@1 {a['hit@1'] - b['hit@1']:+d}  "
            f"MRR {a['mrr'] - b['mrr']:+.3f}  "
            f"p50 {a['p50_ms'] - b['p50_ms']:+d}ms")


def format_adversarial(result):
    lines = ["adversarial eval (consolidation / self-heal)"]

    def block(label, checks):
        lines.append(f"  {label}:")
        for c in checks:
            if c.get("error"):
                verdict = "ERROR"
            elif c["fabricated"]:
                verdict = "FABRICATED"
            else:
                verdict = "clean"
            lines.append(f"    [{verdict}] {c['query'][:60]}")
            if c.get("error"):
                lines.append(f"       error: {c['error']}")
            for f in c["fabricated"]:
                lines.append(f"       forbidden match: {f}")
            for t in c["top3"]:
                lines.append(f"       · {t}")

    block("after initial retain + consolidate", result["before"])
    after = result["after_corrections"]
    if result.get("corrections_failed"):
        lines.append("  ── corrections retain FAILED — self-heal unknown")
    elif after is not None:
        block("after corrections + re-consolidate", after)
        errs = sum(1 for c in after if c.get("error"))
        healed = all(not c["fabricated"] for c in after)
        if errs:
            lines.append(f"  ── self-heal: UNKNOWN ({errs} check(s) errored)")
        elif any(c["fabricated"] for c in result["before"]):
            verdict = "yes" if healed else "NO — fabrication survived corrections"
            lines.append(f"  ── self-heal: {verdict}")
        elif not healed:
            lines.append("  ── corrections introduced a fabrication")
        else:
            lines.append("  ── no fabrication observed")
    else:
        any_fab = any(c["fabricated"] for c in result["before"])
        verdict = "fabrication observed" if any_fab else "no fabrication observed"
        lines.append(f"  ── {verdict} (no corrections in probe)")
    return "\n".join(lines)
