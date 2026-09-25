"""Run-log primitives. A run is a JSONL file of typed events; everything
(text summary, TUI replay, charts) consumes this one artifact."""
import hashlib
import json
import re
import time
from pathlib import Path

SCHEMA_VERSION = 1

_SECRET = re.compile(r"(token|key|secret|password|credential)", re.I)


def redact(config):
    """Mask secret-looking values (recursively) so adapter config can go into run_meta."""
    out = {}
    for k, v in config.items():
        if isinstance(v, dict):
            out[k] = redact(v)
        else:
            out[k] = "***" if _SECRET.search(k) and v else v
    return out


def utc_stamp():
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def run_path(runs_dir, adapter, kind):
    """Timestamped log path; uniquified because an ab run starts two legs
    with the same adapter name inside the same second."""
    p = Path(runs_dir)
    p.mkdir(parents=True, exist_ok=True)
    base = f"{utc_stamp()}_{adapter}_{kind}"
    path = p / f"{base}.jsonl"
    n = 2
    while path.exists():
        path = p / f"{base}-{n}.jsonl"
        n += 1
    return path


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class RunLog:
    """Append-only JSONL event log for one run.

    Contract: opens the file in append mode, writes one JSON object per line,
    and flushes after every emit. The parent directory must already exist
    (run_path handles that). Call close() when done, or use as a context
    manager (`with RunLog(path) as log:`).
    """

    def __init__(self, path):
        self.path = Path(path)
        self._f = open(self.path, "a", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def emit(self, ev, **fields):
        rec = {"ev": ev, **fields}
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()
        return rec

    def close(self):
        self._f.close()


def read_events(path):
    """Read events, skipping malformed/truncated lines (a killed run can
    leave a partial last line; keep every line that parses)."""
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):  # an event is always an object; a foreign file's
                events.append(rec)     # arrays/scalars would crash every consumer's .get
    return events
