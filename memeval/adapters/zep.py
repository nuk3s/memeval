import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .base import MemoryAdapter


class ZepAdapter(MemoryAdapter):
    """TEMPLATE adapter for Zep Community Edition — written to the documented CE
    session/fact API, NOT verified against a live instance.

    Honesty notes (verified against source 2026-07-13):
    - CE has NO graph endpoints; POST /api/v2/graph{,/search} is Zep Cloud only.
    - CE search returns LLM-extracted FACTS, not your stored text — rank evals score
      what the extractor produced, and no similarity score is exposed.
    - Upstream has deprecated CE (code moved to legacy/; OSS direction is Graphiti).

    Config from env:
      ZEP_URL      base URL (default http://localhost:8000)
      ZEP_API_KEY  api_secret from zep.yaml; sent as "Authorization: Api-Key <v>"
      ZEP_USER     user_id    (default memeval)
      ZEP_SESSION  session_id (default memeval-scratch)
    """

    name = "zep"
    status = "template"

    def __init__(self):
        self.url = os.environ.get("ZEP_URL", "http://localhost:8000").rstrip("/")
        self.api_key = os.environ.get("ZEP_API_KEY", "")
        self.user = os.environ.get("ZEP_USER", "memeval")
        self.session = os.environ.get("ZEP_SESSION", "memeval-scratch")

    def _call(self, path, body, timeout=120):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     method="POST", headers=headers)
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

    def prepare(self):
        for path, body in (("/api/v2/users", {"user_id": self.user}),
                           ("/api/v2/sessions", {"session_id": self.session,
                                                 "user_id": self.user})):
            try:
                self._call(path, body)
            except urllib.error.HTTPError:
                pass  # already exists — fine; connectivity errors must propagate

    def retain(self, items):
        session = urllib.parse.quote(self.session, safe="")
        for i in items:
            self._call(f"/api/v2/sessions/{session}/memory",
                       {"messages": [{"role": "memeval", "role_type": "user",
                                      "content": i["content"]}]}, timeout=300)

    def recall(self, query, k=10):
        r = self._call(f"/api/v2/sessions/search?limit={k}",
                       {"text": query, "user_id": self.user}, timeout=60)
        out = []
        for m in r.get("results", [])[:k]:
            fact = (m.get("fact") or {}).get("fact", "")
            if fact:
                out.append({"text": fact, "score": None})
        return out

    def config(self):
        return {"url": self.url, "user": self.user, "session": self.session,
                "api_key": self.api_key}
