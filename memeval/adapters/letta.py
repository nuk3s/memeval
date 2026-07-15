import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .base import MemoryAdapter


class LettaAdapter(MemoryAdapter):
    """TEMPLATE adapter for a self-hosted Letta (MemGPT) server — written to the
    documented V1 REST API, NOT verified against a live instance.

    Uses the agent archival-memory endpoints of the classic :8283 server. Upstream
    now labels this API "V1 SDK (legacy)" (active development moved to the Letta App
    Server), so verify paths against your server version before trusting results.

    Config from env:
      LETTA_URL       base URL (default http://localhost:8283)
      LETTA_TOKEN     server password; sent as Bearer when set (secure mode)
      LETTA_AGENT_ID  REQUIRED — archival memory hangs off an agent
    """

    name = "letta"
    status = "template"

    def __init__(self):
        self.url = os.environ.get("LETTA_URL", "http://localhost:8283").rstrip("/")
        self.token = os.environ.get("LETTA_TOKEN", "")
        self.agent_id = os.environ.get("LETTA_AGENT_ID", "")

    def _agent_path(self, suffix=""):
        if not self.agent_id:
            raise RuntimeError("LETTA_AGENT_ID is required for the letta adapter")
        agent = urllib.parse.quote(self.agent_id, safe="")
        return f"{self.url}/v1/agents/{agent}/archival-memory{suffix}"

    def _call(self, url, body=None, timeout=120):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data,
                                     method="POST" if body is not None else "GET",
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode()[:200]
            except Exception:
                pass
            raise urllib.error.HTTPError(
                e.url, e.code, f"{e.reason} — {detail}" if detail else e.reason,
                e.headers, None) from None

    def retain(self, items):
        for i in items:
            self._call(self._agent_path(), {"text": i["content"]}, timeout=300)

    def recall(self, query, k=10):
        qs = urllib.parse.urlencode({"query": query, "top_k": k})
        r = self._call(self._agent_path(f"/search?{qs}"))
        return [{"text": m.get("content", ""), "score": None}
                for m in r.get("results", [])[:k]]

    def config(self):
        return {"url": self.url, "agent_id": self.agent_id, "token": self.token}
