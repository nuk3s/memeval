import json
import os
import ssl
import urllib.request
from typing import Dict, List, Optional

from .base import MemoryAdapter


class HindsightAdapter(MemoryAdapter):
    """Adapter for a self-hosted Hindsight instance (vectorize-io/hindsight).

    Config from env, or pass to the constructor:
      HINDSIGHT_URL      base URL           (default http://localhost:8888)
      HINDSIGHT_TOKEN    bearer token       (default empty)
      HINDSIGHT_BANK     bank name          (default memeval-scratch)
      HINDSIGHT_INSECURE any non-empty value skips TLS verification (for internal CAs)
      HINDSIGHT_BATCH    items per retain POST (default 20)
      HINDSIGHT_TIMEOUT  seconds per retain POST (default 600)

    Point HINDSIGHT_BANK at a throwaway bank. The eval writes into it.
    """

    name = "hindsight"
    status = "tested"

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None,
                 bank: Optional[str] = None, insecure: Optional[bool] = None):
        self.url = (url or os.environ.get("HINDSIGHT_URL", "http://localhost:8888")).rstrip("/")
        self.token = token if token is not None else os.environ.get("HINDSIGHT_TOKEN", "")
        self.bank = bank or os.environ.get("HINDSIGHT_BANK", "memeval-scratch")
        if insecure is None:
            insecure = bool(os.environ.get("HINDSIGHT_INSECURE", ""))
        self.batch = int(os.environ.get("HINDSIGHT_BATCH", "20"))
        self.timeout = int(os.environ.get("HINDSIGHT_TIMEOUT", "600"))
        self.ctx = ssl.create_default_context()
        if insecure:
            self.ctx.check_hostname = False
            self.ctx.verify_mode = ssl.CERT_NONE

    def _call(self, method: str, path: str, body=None, timeout: int = 300):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.url + path, data=data, method=method,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout, context=self.ctx) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else {}

    def _bank(self, suffix: str) -> str:
        return f"/v1/default/banks/{self.bank}{suffix}"

    def prepare(self, observations: bool = False) -> None:
        # create the bank with observations off by default (see the blog notes)
        self._call("PUT", self._bank(""), {"name": self.bank, "enable_observations": observations})

    def _to_item(self, i: Dict) -> Dict:
        item = {"content": i["content"], "context": i.get("context", "memeval")}
        if i.get("id"):
            item["document_id"] = i["id"]
            item["update_mode"] = "replace"
        if i.get("tags"):
            item["tags"] = i["tags"]
        return item

    def retain(self, items: List[Dict]) -> None:
        # Batched so a large corpus on a slow-extraction store never exceeds a single
        # HTTP timeout; tune HINDSIGHT_BATCH / HINDSIGHT_TIMEOUT.
        for start in range(0, len(items), self.batch):
            chunk = items[start:start + self.batch]
            self._call("POST", self._bank("/memories"),
                       {"items": [self._to_item(i) for i in chunk], "async": False},
                       timeout=self.timeout)

    def recall(self, query: str, k: int = 10) -> List[Dict]:
        r = self._call("POST", self._bank("/memories/recall"), {"query": query, "max_tokens": 1500})
        out = []
        for m in r.get("results", [])[:k]:
            sc = m.get("scores")
            score = sc.get("final") if isinstance(sc, dict) else m.get("score")
            out.append({"text": m.get("text", ""), "score": score})
        return out

    def consolidate(self) -> None:
        try:
            self._call("POST", self._bank("/consolidate"), {})
        except Exception:
            pass  # bank may have observations off; nothing to consolidate

    def supersede(self, doc_id: str, content: str, tags: Optional[list] = None) -> None:
        self.retain([{"content": content, "id": doc_id, "tags": tags or []}])

    def config(self):
        return {"url": self.url, "bank": self.bank, "token": self.token, "batch": self.batch}
