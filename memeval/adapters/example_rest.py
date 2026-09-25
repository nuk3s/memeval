import os
from typing import Dict, List

from ._http import request_json
from .base import MemoryAdapter


class ExampleRestAdapter(MemoryAdapter):
    """A template adapter for a generic bearer-token REST memory API.

    This is a starting point, not a working driver for any specific product. Copy it,
    change the two paths and the request/response shapes to match your system, and add
    it to adapters/__init__.py REGISTRY. mem0, Zep, Letta, and a raw pgvector service
    each take about this much code.

    Assumed shape (edit to fit yours):
      POST {base}/memories  {"text": str, "user_id": str}       -> stores one memory
      POST {base}/search    {"query": str, "user_id": str}      -> {"results": [{"memory","score"}]}
    """

    name = "example-rest"

    def __init__(self, url=None, token=None, user=None):
        self.url = (url or os.environ.get("MEMEVAL_REST_URL", "http://localhost:8000")).rstrip("/")
        self.token = token if token is not None else os.environ.get("MEMEVAL_REST_TOKEN", "")
        self.user = user or os.environ.get("MEMEVAL_REST_USER", "memeval")

    def _post(self, path: str, body: Dict, timeout: int = 120):
        return request_json(self.url + path, body=body, method="POST", timeout=timeout,
                            headers={"Authorization": f"Bearer {self.token}"})

    def retain(self, items: List[Dict]) -> None:
        for i in items:
            self._post("/memories", {"text": i["content"], "user_id": self.user})

    def recall(self, query: str, k: int = 10) -> List[Dict]:
        r = self._post("/search", {"query": query, "user_id": self.user})
        out = []
        for m in r.get("results", [])[:k]:
            out.append({"text": m.get("memory", m.get("text", "")), "score": m.get("score")})
        return out
