import os
import urllib.parse

from ._http import request_json
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
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        return request_json(url, body=body, headers=headers, timeout=timeout)

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
