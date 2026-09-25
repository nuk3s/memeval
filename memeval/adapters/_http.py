"""Shared urllib plumbing for the bundled REST adapters. Stdlib only.

Every adapter used to carry its own copy of this, and only some of them folded the
server's error body into the exception. Keeping it here means a 4xx in the run log
says *why* for all of them.
"""
import json
import urllib.error
import urllib.request

ERROR_BODY_LIMIT = 200


def request_json(url, body=None, method=None, headers=None, timeout=120, context=None):
    """Send `body` as JSON (POST by default when a body is given, GET otherwise) and
    decode the JSON reply; an empty reply decodes to {}.

    An HTTPError is re-raised with the first ERROR_BODY_LIMIT chars of the response
    body in its message, so the error row reads "400 Bad Request — {detail...}"
    instead of just the status line.
    """
    data = json.dumps(body).encode() if body is not None else None
    if method is None:
        method = "POST" if data is not None else "GET"
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=context) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:ERROR_BODY_LIMIT]
        except Exception:
            pass
        raise urllib.error.HTTPError(
            e.url, e.code, f"{e.reason} — {detail}" if detail else e.reason,
            e.headers, None) from None
