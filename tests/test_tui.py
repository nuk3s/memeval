import pytest

pytest.importorskip("textual")

from memeval.render.tui import ReplayApp  # noqa: E402

EVENTS = [
    {"ev": "run_meta", "run_id": "r1", "kind": "rank", "adapter": "fake", "k": 10},
    {"ev": "ready", "polled_s": 0.2, "verified_n": 2, "sampled_n": 2, "missing": []},
    {"ev": "query", "phase": "rank", "rep": 1, "query": "what port does checkout use?",
     "rank": 1, "ms": 140, "error": None,
     "hits": [{"text": "checkout runs on port 8080", "score": 0.9},
              {"text": "batch job at 02:00", "score": 0.2}]},
    {"ev": "query", "phase": "rank", "rep": 1, "query": "which db?", "rank": None,
     "ms": 90, "error": "boom", "hits": []},
    {"ev": "summary", "kind": "rank", "n": 2, "k": 10, "hit@1": 1, "hit@3": 1,
     "hit@k": 1, "mrr": 0.5, "p50_ms": 140, "p95_ms": 140, "errors": 1},
]


@pytest.mark.asyncio
async def test_replay_steps_and_navigates():
    app = ReplayApp(EVENTS)
    async with app.run_test() as pilot:
        assert app.idx == 0                      # first step (ready event)
        await pilot.press("space")
        assert app.idx == 1                      # first query
        await pilot.press("space")
        await pilot.press("space")
        assert app.idx == 3                      # summary; step past end stays put
        await pilot.press("space")
        assert app.idx == 3
        await pilot.press("left")
        assert app.idx == 2
        await pilot.press("q")


@pytest.mark.asyncio
async def test_replay_renders_query_content():
    app = ReplayApp(EVENTS)
    async with app.run_test() as pilot:
        await pilot.press("space")
        text = str(app.query_one("#main").render())
        assert "what port does checkout use?" in text
        assert "8080" in text


@pytest.mark.asyncio
async def test_replay_renders_error_verdict():
    events = [
        {"ev": "run_meta", "run_id": "r2", "kind": "adversarial", "adapter": "fake"},
        {"ev": "verdict", "phase": "adv_before", "query": "q1", "fabricated": [],
         "expected_present": [], "top3": [], "error": "ConnectionError"},
    ]
    app = ReplayApp(events)
    async with app.run_test():
        text = str(app.query_one("#main").render())
        assert "ERROR" in text and "ConnectionError" in text


@pytest.mark.asyncio
async def test_replay_escapes_bracket_content():
    events = [
        {"ev": "run_meta", "run_id": "r3", "kind": "rank", "adapter": "fake"},
        {"ev": "query", "phase": "rank", "rep": 1, "error": None, "rank": 1, "ms": 10,
         "query": "what config lives at path[/etc/app] now?",
         "hits": [{"text": "see config[/etc/app]/prod.yaml", "score": 0.9}]},
    ]
    app = ReplayApp(events)
    async with app.run_test():
        text = str(app.query_one("#main").render())
        assert "etc/app" in text  # renders instead of MarkupError


@pytest.mark.asyncio
async def test_replay_shows_retain_and_consolidate_steps():
    events = [
        {"ev": "run_meta", "run_id": "r4", "kind": "adversarial", "adapter": "fake"},
        {"ev": "retain", "n_items": 2, "ms": 12, "ok": True, "error": None},
        {"ev": "consolidate", "ms": 5, "ok": False, "error": "no such endpoint"},
    ]
    app = ReplayApp(events)
    async with app.run_test() as pilot:
        assert len(app.steps) == 2
        text = str(app.query_one("#main").render())
        assert "retain" in text and "2 items" in text
        await pilot.press("space")
        text = str(app.query_one("#main").render())
        assert "consolidate" in text and "no such endpoint" in text


@pytest.mark.asyncio
async def test_replay_empty_steps_message():
    app = ReplayApp([{"ev": "run_meta", "run_id": "r5", "kind": "rank", "adapter": "fake"}])
    async with app.run_test():
        text = str(app.query_one("#main").render())
        assert "no steps" in text
