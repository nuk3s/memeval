"""Deterministic in-memory adapter. Test scaffolding only — deliberately NOT
registered in memeval.adapters.REGISTRY (spec: no user-facing local baseline)."""
import re
import time

from memeval.adapters.base import MemoryAdapter


class FakeAdapter(MemoryAdapter):
    """Scores recall by word overlap.

    index_delay: models a cold index warming up once — the first retain() arms it;
                 docs retained after warm-up are visible immediately.
    drop_ids:    item ids silently NOT stored (simulates silent write loss)
    fail_queries: words; recall() raises ConnectionError when the query contains one
                 (whole-word match, case-insensitive)
    """

    name = "fake"
    status = "test-only"

    def __init__(self, index_delay=0.0, drop_ids=(), fail_queries=()):
        self.docs = []
        self.index_delay = index_delay
        self.drop_ids = set(drop_ids)
        self.fail_queries = tuple(fail_queries)
        self._stored_at = None

    def retain(self, items):
        if self._stored_at is None:
            self._stored_at = time.monotonic()
        for i in items:
            if i.get("id") in self.drop_ids:
                continue
            self.docs.append(i["content"])

    def recall(self, query, k=10):
        for frag in self.fail_queries:
            if re.search(rf"\b{re.escape(frag)}\b", query, re.I):
                raise ConnectionError(f"injected failure for {frag!r}")
        if self._stored_at is None or time.monotonic() - self._stored_at < self.index_delay:
            visible = []
        else:
            visible = self.docs
        qwords = {w for w in query.lower().split() if len(w) > 3}
        scored = []
        for d in visible:
            dwords = {w.strip(".,") for w in d.lower().split()}
            overlap = len(qwords & dwords)
            if overlap:
                scored.append((overlap, d))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [{"text": d, "score": float(s)} for s, d in scored[:k]]

    def config(self):
        return {"kind": "fake"}
