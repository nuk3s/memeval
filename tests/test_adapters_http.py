import io
import json
import urllib.request

import pytest


class FakeHTTP:
    """Monkeypatched urllib.request.urlopen: records requests, returns canned JSON."""

    def __init__(self, responses):
        self.responses = responses  # list of dicts, popped in order (last one sticks)
        self.requests = []          # (method, url, body-dict-or-None, headers)

    def __call__(self, req, timeout=None, context=None):
        body = json.loads(req.data.decode()) if req.data else None
        self.requests.append((req.get_method(), req.full_url, body, dict(req.headers)))
        payload = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        resp = io.BytesIO(json.dumps(payload).encode())
        resp.__enter__ = lambda *a: resp
        resp.__exit__ = lambda *a: False
        return resp


@pytest.fixture
def http(monkeypatch):
    def install(responses):
        fake = FakeHTTP(responses)
        monkeypatch.setattr(urllib.request, "urlopen", fake)
        return fake
    return install


def test_mem0_retain_and_recall(http, monkeypatch):
    monkeypatch.setenv("MEM0_URL", "http://mem0:8888")
    monkeypatch.setenv("MEM0_API_KEY", "k1")
    monkeypatch.setenv("MEM0_USER", "u1")
    from memeval.adapters.mem0 import Mem0Adapter
    fake = http([{"results": [{"id": "1", "memory": "stored", "event": "ADD"}]}])
    a = Mem0Adapter()
    a.retain([{"content": "fact one"}, {"content": "fact two"}])
    assert len(fake.requests) == 2
    method, url, body, headers = fake.requests[0]
    assert method == "POST" and url == "http://mem0:8888/memories"
    assert body["messages"] == [{"role": "user", "content": "fact one"}]
    assert body["user_id"] == "u1"
    assert headers.get("X-api-key") == "k1"  # urllib title-cases header names

    fake = http([{"results": [{"id": "1", "memory": "fact one", "score": 0.9}]}])
    hits = a.recall("fact?", 5)
    method, url, body, _ = fake.requests[0]
    assert url == "http://mem0:8888/search"
    assert body == {"query": "fact?", "filters": {"user_id": "u1"}, "top_k": 5}
    assert hits == [{"text": "fact one", "score": 0.9}]


def test_mem0_infer_flag(http, monkeypatch):
    monkeypatch.setenv("MEM0_INFER", "0")
    from memeval.adapters.mem0 import Mem0Adapter
    fake = http([{"results": []}])
    Mem0Adapter().retain([{"content": "raw"}])
    assert fake.requests[0][2]["infer"] is False


def test_letta_retain_and_recall(http, monkeypatch):
    monkeypatch.setenv("LETTA_URL", "http://letta:8283")
    monkeypatch.setenv("LETTA_AGENT_ID", "agent-1")
    monkeypatch.setenv("LETTA_TOKEN", "pw")
    from memeval.adapters.letta import LettaAdapter
    fake = http([[{"id": "p1", "text": "fact"}]])
    a = LettaAdapter()
    a.retain([{"content": "fact one"}])
    method, url, body, headers = fake.requests[0]
    assert url == "http://letta:8283/v1/agents/agent-1/archival-memory"
    assert body == {"text": "fact one"} and headers.get("Authorization") == "Bearer pw"

    fake = http([{"results": [{"id": "p1", "content": "fact one"}], "count": 1}])
    hits = a.recall("fact", 5)
    method, url, _, _ = fake.requests[0]
    assert "/archival-memory/search?" in url and "top_k=5" in url and "query=fact" in url
    assert hits == [{"text": "fact one", "score": None}]


def test_letta_requires_agent_id(monkeypatch):
    monkeypatch.delenv("LETTA_AGENT_ID", raising=False)
    from memeval.adapters.letta import LettaAdapter
    with pytest.raises(RuntimeError, match="LETTA_AGENT_ID"):
        LettaAdapter().retain([{"content": "x"}])


def test_zep_prepare_retain_recall(http, monkeypatch):
    monkeypatch.setenv("ZEP_URL", "http://zep:8000")
    monkeypatch.setenv("ZEP_API_KEY", "sec")
    monkeypatch.setenv("ZEP_USER", "u1")
    monkeypatch.setenv("ZEP_SESSION", "s1")
    from memeval.adapters.zep import ZepAdapter
    fake = http([{}])
    a = ZepAdapter()
    a.prepare()
    urls = [r[1] for r in fake.requests]
    assert "http://zep:8000/api/v2/users" in urls
    assert "http://zep:8000/api/v2/sessions" in urls
    assert fake.requests[0][3].get("Authorization") == "Api-Key sec"

    fake = http([{}])
    a.retain([{"content": "fact one"}])
    method, url, body, _ = fake.requests[0]
    assert url == "http://zep:8000/api/v2/sessions/s1/memory"
    assert body["messages"][0]["content"] == "fact one"
    assert body["messages"][0]["role_type"] == "user"

    fake = http([{"results": [{"fact": {"uuid": "f1", "fact": "extracted fact"}}]}])
    hits = a.recall("fact", 5)
    method, url, body, _ = fake.requests[0]
    assert url == "http://zep:8000/api/v2/sessions/search?limit=5"
    assert body == {"text": "fact", "user_id": "u1"}
    assert hits == [{"text": "extracted fact", "score": None}]


def test_registry_and_statuses():
    from memeval.adapters import REGISTRY
    assert set(REGISTRY) == {"hindsight", "mem0", "letta", "zep", "example-rest"}
    assert REGISTRY["mem0"].status == "tested"
    assert REGISTRY["letta"].status == "template"
    assert REGISTRY["zep"].status == "template"


def test_zep_prepare_propagates_connectivity_errors(monkeypatch):
    import urllib.error
    monkeypatch.setenv("ZEP_URL", "http://zep:8000")
    from memeval.adapters.zep import ZepAdapter

    def refuse(req, timeout=None, context=None):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(urllib.error.URLError):
        ZepAdapter().prepare()


def test_hindsight_retain_batches(http, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_URL", "http://hs:8888")
    monkeypatch.setenv("HINDSIGHT_TOKEN", "t")
    monkeypatch.setenv("HINDSIGHT_BANK", "scratch")
    monkeypatch.setenv("HINDSIGHT_BATCH", "20")
    from memeval.adapters.hindsight import HindsightAdapter
    fake = http([{}])
    a = HindsightAdapter()
    a.retain([{"content": f"fact {i}"} for i in range(45)])
    posts = [r for r in fake.requests if r[1].endswith("/memories")]
    assert len(posts) == 3                       # 20 + 20 + 5
    assert len(posts[0][2]["items"]) == 20 and len(posts[2][2]["items"]) == 5


@pytest.mark.parametrize("name", ["hindsight", "mem0", "letta", "zep", "example-rest"])
def test_http_error_body_surfaces_for_every_adapter(monkeypatch, name):
    # one shared helper: a 4xx from any store says why, not just the status line
    import urllib.error

    from memeval.adapters import REGISTRY
    monkeypatch.setenv("LETTA_AGENT_ID", "agent-1")

    def bad_request(req, timeout=None, context=None):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                     io.BytesIO(b'{"detail": "user_id required"}'))
    monkeypatch.setattr(urllib.request, "urlopen", bad_request)
    with pytest.raises(urllib.error.HTTPError, match="user_id required"):
        REGISTRY[name]().recall("q", 5)


def test_example_rest_retain_and_recall(http, monkeypatch):
    monkeypatch.setenv("MEMEVAL_REST_URL", "http://rest:8000/")
    monkeypatch.setenv("MEMEVAL_REST_TOKEN", "tok")
    monkeypatch.setenv("MEMEVAL_REST_USER", "u9")
    from memeval.adapters.example_rest import ExampleRestAdapter
    fake = http([{}])
    a = ExampleRestAdapter()
    a.retain([{"content": "fact one"}])
    method, url, body, headers = fake.requests[0]
    assert method == "POST" and url == "http://rest:8000/memories"
    assert body == {"text": "fact one", "user_id": "u9"}
    assert headers.get("Authorization") == "Bearer tok"
    assert headers.get("Content-type") == "application/json"

    fake = http([{"results": [{"memory": "fact one", "score": 0.5}, {"text": "alt"}]}])
    hits = a.recall("fact", 1)
    assert fake.requests[0][2] == {"query": "fact", "user_id": "u9"}
    assert hits == [{"text": "fact one", "score": 0.5}]  # k applied client-side


def test_request_json_defaults_to_get_without_body(http):
    from memeval.adapters._http import request_json
    fake = http([{"ok": True}])
    assert request_json("http://x/ping") == {"ok": True}
    assert fake.requests[0][0] == "GET" and fake.requests[0][2] is None
    fake = http([{}])
    request_json("http://x/put", body={"a": 1}, method="PUT")
    assert fake.requests[0][0] == "PUT" and fake.requests[0][2] == {"a": 1}


def test_hindsight_batch_floor(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_BATCH", "0")
    from memeval.adapters.hindsight import HindsightAdapter
    assert HindsightAdapter().batch == 1  # used to raise: range() arg 3 must not be zero


def test_hindsight_consolidate_failure_reaches_the_run_log(monkeypatch, tmp_path):
    import urllib.error

    from memeval.adapters.hindsight import HindsightAdapter
    from memeval.events import RunLog, read_events
    from memeval.runner import Runner

    def refuse(req, timeout=None, context=None):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(urllib.error.URLError):
        HindsightAdapter().consolidate()  # adapter no longer swallows it...
    log = RunLog(tmp_path / "run.jsonl")
    assert Runner(HindsightAdapter(), log).consolidate() is False  # ...the Runner guards it
    log.close()
    ev = [e for e in read_events(log.path) if e["ev"] == "consolidate"][0]
    assert ev["ok"] is False and "refused" in ev["error"]  # and the log tells the truth
