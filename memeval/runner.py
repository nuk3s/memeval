"""One wrapper around every adapter call: timing, retries, error rows,
readiness polling, event emission. Hardening lives here, written once."""
import re
import time

HIT_TEXT_LIMIT = 300
# ConnectionError and TimeoutError are OSError subclasses — listed anyway for
# readability. urllib.error.URLError is also an OSError subclass, so plain-http
# adapter failures land here too.
RETRYABLE = (ConnectionError, TimeoutError, OSError)


def _ms(t0):
    return round((time.monotonic() - t0) * 1000)


def _coerce_score(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _normalize_hits(hits):
    """Defend the hit contract: adapters/servers can return junk; text must be str."""
    if not isinstance(hits, list):
        return []
    return [{"text": str(h.get("text") or ""), "score": _coerce_score(h.get("score"))}
            for h in hits if isinstance(h, dict)]


def _overlaps(source, text, threshold=0.5):
    """Word-overlap match. Extracting stores (Hindsight, mem0 with infer on)
    rewrite text at retain time, so verbatim containment would false-negative."""
    words = {w.strip(".,;:!?").lower() for w in source.split() if len(w) > 3}
    if not words:
        return False
    tl = text.lower()
    found = sum(1 for w in words if w in tl)
    return found / len(words) >= threshold


class Runner:
    def __init__(self, adapter, log, retries=1, ready_timeout=60.0,
                 poll_interval=1.0, echo=None):
        self.adapter = adapter
        self.log = log
        self.retries = max(0, retries)
        self.ready_timeout = ready_timeout
        self.poll_interval = poll_interval
        self.echo = echo or (lambda s: None)

    def retain(self, items):
        t0 = time.monotonic()
        try:
            self.adapter.retain(items)
        except Exception as e:
            self.log.emit("retain", n_items=len(items), ms=_ms(t0), ok=False,
                          error=str(e) or e.__class__.__name__)
            self.echo(f"retain FAILED: {e}")
            return False
        self.log.emit("retain", n_items=len(items), ms=_ms(t0), ok=True, error=None)
        self.echo(f"retained {len(items)} items in {_ms(t0)}ms")
        return True

    def consolidate(self):
        """Guarded consolidate: an unsupported/failing consolidator never kills the run."""
        t0 = time.monotonic()
        try:
            self.adapter.consolidate()
        except Exception as e:
            self.log.emit("consolidate", ms=_ms(t0), ok=False,
                          error=str(e) or e.__class__.__name__)
            self.echo(f"consolidate failed (continuing): {e}")
            return False
        self.log.emit("consolidate", ms=_ms(t0), ok=True, error=None)
        return True

    def wait_ready(self, corpus, sample=10):
        """Poll until each sampled doc is recallable; report the ones that never are.
        A doc missing at timeout is the 'success:true but not persisted' failure —
        recorded, warned about, and the run continues."""
        docs = corpus[:sample]
        pending = dict(enumerate(docs))
        t0 = time.monotonic()
        while True:
            for idx in list(pending):
                doc = pending[idx]
                try:
                    hits = _normalize_hits(self.adapter.recall(doc["content"][:80], 5))
                except Exception:
                    continue
                if any(_overlaps(doc["content"], h.get("text", "")) for h in hits):
                    del pending[idx]
            if not pending:
                break
            remaining = self.ready_timeout - (time.monotonic() - t0)
            if remaining <= 0:
                break
            time.sleep(min(self.poll_interval, remaining))
        missing = [pending[i].get("id") or f"#{i}" for i in sorted(pending)]
        self.log.emit("ready", polled_s=round(time.monotonic() - t0, 1),
                      verified_n=len(docs) - len(missing), sampled_n=len(docs),
                      missing=missing)
        if missing:
            self.echo(f"WARNING: {len(missing)} doc(s) never became recallable "
                      f"(silent write loss?): {', '.join(missing)}")
        return missing

    def query(self, query, k, phase, rep=1, pattern=None):
        """Recall with retry; returns a row with FULL hit texts. The emitted
        event truncates hit text to HIT_TEXT_LIMIT — verdict regexes must run
        on the row, not the event."""
        hits, ms, err = [], 0, None
        for attempt in range(self.retries + 1):
            t0 = time.monotonic()
            try:
                hits = _normalize_hits(self.adapter.recall(query, k))
                ms, err = _ms(t0), None
                break
            except RETRYABLE as e:
                ms, err = _ms(t0), str(e) or e.__class__.__name__
            except Exception as e:
                ms, err = _ms(t0), str(e) or e.__class__.__name__
                break
        rank = None
        if err is None and pattern:
            try:
                pat = re.compile(pattern, re.I)
                rank = next((i for i, h in enumerate(hits, 1)
                             if pat.search(h.get("text", ""))), None)
            except re.error as e:
                err = f"bad answer pattern {pattern!r}: {e}"
        row = {"phase": phase, "rep": rep, "query": query,
               "rank": rank, "ms": ms, "hits": hits, "error": err}
        self.log.emit("query", phase=phase, rep=rep, query=query, rank=rank, ms=ms,
                      hits=[{"text": h.get("text", "")[:HIT_TEXT_LIMIT],
                             "score": h.get("score")} for h in hits],
                      error=err)
        mark = "×" if err else (str(rank) if rank else "—")
        self.echo(f"  rank {mark:>2}  {ms:>5}ms  {query[:52]}")
        return row
