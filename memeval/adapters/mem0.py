import os

from ._http import request_json
from .base import MemoryAdapter


class Mem0Adapter(MemoryAdapter):
    """Adapter for a self-hosted mem0 OSS REST server (mem0ai/mem0 `server/`).

    Config from env:
      MEM0_URL      base URL          (default http://localhost:8888 — the compose port)
      MEM0_API_KEY  X-API-Key value   (empty = no header; server needs AUTH_DISABLED=true then)
      MEM0_USER     user_id           (default memeval)
      MEM0_INFER    "0" to store verbatim, skipping LLM extraction (default: infer on)

    Auth note: current mem0 server ships auth ON by default (per-user m0sk_ keys or the
    ADMIN_API_KEY env, sent as X-API-Key). For a throwaway eval instance the simplest
    path is AUTH_DISABLED=true in the server env.
    """

    name = "mem0"
    status = "tested"

    def __init__(self):
        self.url = os.environ.get("MEM0_URL", "http://localhost:8888").rstrip("/")
        self.api_key = os.environ.get("MEM0_API_KEY", "")
        self.user = os.environ.get("MEM0_USER", "memeval")
        self.infer = os.environ.get("MEM0_INFER", "1") not in ("0", "false", "no")

    def _call(self, path, body, timeout=120):
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        return request_json(self.url + path, body=body, method="POST",
                            headers=headers, timeout=timeout)

    def retain(self, items):
        for i in items:
            body = {"messages": [{"role": "user", "content": i["content"]}],
                    "user_id": self.user}
            if not self.infer:
                body["infer"] = False
            self._call("/memories", body, timeout=300)

    def recall(self, query, k=10):
        r = self._call("/search", {"query": query,
                                   "filters": {"user_id": self.user},
                                   "top_k": k}, timeout=60)
        return [{"text": m.get("memory", ""), "score": m.get("score")}
                for m in r.get("results", [])[:k]]

    def config(self):
        return {"url": self.url, "user": self.user, "infer": self.infer,
                "api_key": self.api_key}
