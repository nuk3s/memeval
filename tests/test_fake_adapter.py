import time

from conftest import FakeAdapter


def test_fake_scores_by_overlap():
    a = FakeAdapter()
    a.retain([{"content": "the checkout service runs on port 8080"},
              {"content": "the batch job starts at 02:00 UTC"}])
    hits = a.recall("what port does the checkout service use?", 5)
    assert hits and "8080" in hits[0]["text"]


def test_fake_write_drop():
    a = FakeAdapter(drop_ids={"d2"})
    a.retain([{"id": "d1", "content": "alpha beta gamma"},
              {"id": "d2", "content": "delta epsilon zeta"}])
    assert not a.recall("delta epsilon", 5)


def test_second_retain_does_not_rehide_docs():
    a = FakeAdapter(index_delay=0.15)
    a.retain([{"content": "alpha beta gamma delta"}])
    time.sleep(0.2)
    assert a.recall("alpha beta gamma", 5)
    a.retain([{"content": "epsilon zeta eta theta"}])
    assert a.recall("alpha beta gamma", 5), "old docs must stay visible after a second retain"


def test_fail_queries_word_boundary_and_case():
    a = FakeAdapter(fail_queries=("port",))
    a.retain([{"content": "career opportunities abound here"}])
    assert a.recall("what opportunities exist?", 5)  # substring 'port' must NOT trip it
    import pytest
    with pytest.raises(ConnectionError):
        a.recall("which PORT is used?", 5)  # case-insensitive whole word must trip
