"""Chart artifacts from run logs. Requires the [charts] extra (matplotlib)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..events import read_events  # noqa: E402

ACCENT = "#4C6EF5"
ACCENT_B = "#12B886"
MISS = "#E8590C"
ERR = "#C92A2A"
GRID = "#E9ECEF"


def _load(run_path):
    evs = read_events(run_path)
    meta = evs[0] if evs and evs[0].get("ev") == "run_meta" else {}
    queries = [e for e in evs if e.get("ev") == "query"]
    summary = next((e for e in reversed(evs) if e.get("ev") == "summary"), {})
    return meta, queries, summary


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _rank_counts(queries, k):
    buckets = ["1", "2", "3", ">3", "miss", "error"]
    counts = [0, 0, 0, 0, 0, 0]
    for q in queries:
        r = q.get("rank")
        if q.get("error"):
            counts[5] += 1
        elif not r:
            counts[4] += 1
        elif r <= 3:
            counts[r - 1] += 1
        else:
            counts[3] += 1
    return buckets, counts


def render_chart(run_path, out=None, png=False):
    meta, queries, s = _load(run_path)
    kind = meta.get("kind")
    if kind not in ("rank", "ab-leg"):
        raise ValueError(f"chart supports rank/ab-leg run logs, not {kind!r} "
                         "(adversarial runs have no rank data)")
    k = meta.get("k") or 10
    label = meta.get("adapter", "?")
    out = out or f"charts/{Path(run_path).stem}.{'png' if png else 'svg'}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    buckets, counts = _rank_counts(queries, k)
    colors = [ACCENT] * 4 + [MISS, ERR]
    ax1.bar(buckets, counts, color=colors)
    ax1.set_title("where the gold answer ranked")
    ax1.set_ylabel("queries")
    _style(ax1)

    lat = sorted(q["ms"] for q in queries if not q.get("error"))
    ax2.bar(range(len(lat)), lat, color=ACCENT, width=0.9)
    for val, name in ((s.get("p50_ms"), "p50"), (s.get("p95_ms"), "p95")):
        if val:
            ax2.axhline(val, color=MISS, linewidth=1, linestyle="--")
            ax2.annotate(f"{name} {val}ms", (0, val), fontsize=8, color=MISS,
                         xytext=(2, 3), textcoords="offset points")
    ax2.set_title("recall latency (sorted)")
    ax2.set_ylabel("ms")
    ax2.set_xticks([])
    _style(ax2)

    fig.suptitle(
        f"memeval rank [{label}]  ·  hit@1 {s.get('hit@1', '?')}/{s.get('n', '?')}  "
        f"MRR {s.get('mrr', '?')}  ·  errors {s.get('errors', '?')}  ·  "
        f"{Path(run_path).stem}", fontsize=10)
    fig.savefig(out)
    plt.close(fig)
    return out


def render_ab_chart(run_a, run_b, out=None, png=False):
    (ma, _, sa), (mb, _, sb) = _load(run_a), _load(run_b)
    for m in (ma, mb):
        kind = m.get("kind")
        if kind not in ("rank", "ab-leg"):
            raise ValueError(f"chart supports rank/ab-leg run logs, not {kind!r} "
                             "(adversarial runs have no rank data)")
    la, lb = ma.get("adapter", "A"), mb.get("adapter", "B")
    out = out or f"charts/ab_{la}_vs_{lb}.{'png' if png else 'svg'}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    metrics = ["hit@1", "hit@3", "hit@k"]
    n_a, n_b = max(sa.get("n", 1), 1), max(sb.get("n", 1), 1)
    va = [sa.get(m, 0) / n_a for m in metrics]
    vb = [sb.get(m, 0) / n_b for m in metrics]
    x = range(len(metrics))
    ax1.bar([i - 0.18 for i in x], va, width=0.36, color=ACCENT, label=la)
    ax1.bar([i + 0.18 for i in x], vb, width=0.36, color=ACCENT_B, label=lb)
    ax1.set_xticks(list(x), metrics)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("hit rate")
    _style(ax1)

    lat_metrics = ["p50_ms", "p95_ms"]
    ax2.bar([i - 0.18 for i in range(2)], [sa.get(m, 0) for m in lat_metrics],
            width=0.36, color=ACCENT, label=la)
    ax2.bar([i + 0.18 for i in range(2)], [sb.get(m, 0) for m in lat_metrics],
            width=0.36, color=ACCENT_B, label=lb)
    ax2.set_xticks([0, 1], ["p50", "p95"])
    ax2.set_title("recall latency (ms)")
    _style(ax2)

    fig.legend(*ax1.get_legend_handles_labels(), loc="outside upper right",
               frameon=False, fontsize=8, ncols=2)

    fig.suptitle(f"memeval A/B  ·  {la} vs {lb}", fontsize=10)
    fig.savefig(out)
    plt.close(fig)
    return out
