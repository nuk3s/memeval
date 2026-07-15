"""Step-through replay of a recorded run. Requires the [tui] extra (Textual)."""
from textual.app import App
from textual.markup import escape
from textual.widgets import Footer, Static

from .text import bar


class ReplayApp(App):
    """Replays run-log events one step at a time.

    Steps are the retain/ready/query/verdict/summary/consolidate events;
    run_meta feeds the header.
    """

    BINDINGS = [
        ("space", "step", "step"),
        ("right", "step", "next"),
        ("left", "back", "prev"),
        ("a", "toggle_auto", "auto-advance"),
        ("q", "quit", "quit"),
    ]

    CSS = """
    #head  { height: 2; padding: 0 1; background: $panel; color: $text; }
    #main  { padding: 1 2; }
    #stats { dock: bottom; height: 2; padding: 0 1; background: $panel; }
    """

    def __init__(self, events):
        super().__init__()
        self.meta = events[0] if events and events[0].get("ev") == "run_meta" else {}
        self.steps = [e for e in events
                      if e.get("ev") in ("retain", "ready", "query", "verdict",
                                         "summary", "consolidate")]
        self.idx = 0
        self._timer = None

    def compose(self):
        yield Static(id="head")
        yield Static(id="main")
        yield Static(id="stats")
        yield Footer()

    def on_mount(self):
        self.refresh_step()

    def action_step(self):
        if self.idx < len(self.steps) - 1:
            self.idx += 1
            self.refresh_step()
        elif self._timer:
            self._timer.stop()
            self._timer = None
            self.refresh_step()

    def action_back(self):
        if self.idx > 0:
            self.idx -= 1
            self.refresh_step()

    def action_toggle_auto(self):
        if self._timer:
            self._timer.stop()
            self._timer = None
        else:
            self._timer = self.set_interval(1.2, self.action_step)
        self.refresh_step()

    # rendering -----------------------------------------------------------

    def refresh_step(self):
        m = self.meta
        auto = "  ▶ auto" if self._timer else ""
        self.query_one("#head", Static).update(
            f"[b]memeval replay[/b] · {escape(str(m.get('run_id', '?')))} · "
            f"{escape(str(m.get('kind', '?')))} \\[{escape(str(m.get('adapter', '?')))}]  "
            f"step {min(self.idx + 1, len(self.steps))}/{len(self.steps)}{auto}")
        if self.steps:
            self.query_one("#main", Static).update(self._main_text(self.steps[self.idx]))
        else:
            self.query_one("#main", Static).update(
                "no steps recorded in this run (aborted before any eval step?)")
        self.query_one("#stats", Static).update(self._stats_text())

    def _main_text(self, e):
        if e["ev"] == "retain":
            status = ("ok" if e.get("ok")
                      else f"[red]FAILED: {escape(str(e.get('error') or ''))}[/red]")
            return (f"[b]retain[/b]\n\n{e.get('n_items', '?')} items · "
                     f"{e.get('ms', '?')}ms · {status}")
        if e["ev"] == "consolidate":
            status = ("ok" if e.get("ok")
                      else f"[red]failed: {escape(str(e.get('error') or ''))}[/red]")
            return f"[b]consolidate[/b]\n\n{e.get('ms', '?')}ms · {status}"
        if e["ev"] == "ready":
            miss = ", ".join(escape(str(m)) for m in e["missing"]) if e["missing"] else "none"
            return (f"[b]index readiness[/b]\n\n"
                    f"verified {e['verified_n']}/{e['sampled_n']} sampled docs "
                    f"in {e['polled_s']}s\nmissing (silent write loss?): {miss}")
        if e["ev"] == "query":
            head = f"[b]► {escape(e['query'])}[/b]\n"
            if e.get("error"):
                return head + f"\n[red]ERROR: {escape(str(e['error']))}[/red]  ({e['ms']}ms)"
            mark = str(e["rank"]) if e["rank"] else "—"
            lines = [head, f"rank {mark} · {e['ms']}ms\n"]
            for i, h in enumerate(e.get("hits", [])[:5], 1):
                tick = "[green]✓[/green] " if e["rank"] == i else "  "
                score = f"{h['score']:.2f}" if h.get("score") is not None else "  — "
                lines.append(f" {i}  {score}  {tick}{escape(h['text'][:90])}")
            return "\n".join(lines)
        if e["ev"] == "verdict":
            if e.get("error"):
                verdict = "[yellow b]ERROR[/yellow b]"
            elif e["fabricated"]:
                verdict = "[red b]FABRICATED[/red b]"
            else:
                verdict = "[green]clean[/green]"
            lines = [f"[b]{escape(str(e['phase']))}[/b]  {verdict}\n",
                     f"► {escape(e['query'])}\n"]
            if e.get("error"):
                lines.append(f" error: {escape(str(e['error']))}")
            for f in e["fabricated"]:
                lines.append(f" forbidden match: {escape(str(f))}")
            for t in e.get("top3", []):
                lines.append(f"  · {escape(str(t))}")
            return "\n".join(lines)
        if e["ev"] == "summary":
            if e.get("kind") == "adversarial":
                return (f"[b]summary[/b]\n\nfabricated before corrections: "
                        f"{e.get('fabricated_before', '?')}\nfabricated after corrections: "
                        f"{e.get('fabricated_after', '?')}\nchecks errored: "
                        f"{e.get('errors_before', '?')} before / "
                        f"{e.get('errors_after', '?')} after")
            return (f"[b]summary[/b]\n\n"
                    f"hit@1 {bar(e['hit@1'], e['n'])} {e['hit@1']}/{e['n']}\n"
                    f"hit@3 {bar(e['hit@3'], e['n'])} {e['hit@3']}/{e['n']}\n"
                    f"MRR {e['mrr']}   p50 {e['p50_ms']}ms   p95 {e['p95_ms']}ms   "
                    f"errors {e['errors']}")
        return ""

    def _stats_text(self):
        seen = [e for e in self.steps[: self.idx + 1] if e.get("ev") == "query"]
        if not seen:
            return "…"
        hit1 = sum(1 for e in seen if e.get("rank") == 1)
        errs = sum(1 for e in seen if e.get("error"))
        lat = sorted(e["ms"] for e in seen if not e.get("error"))
        # quick midpoint approximation; the summary event carries the real
        # statistics.median
        p50 = lat[len(lat) // 2] if lat else 0
        return (f"so far: hit@1 {bar(hit1, len(seen))} {hit1}/{len(seen)}   "
                f"p50 {p50}ms   errors {errs}")
