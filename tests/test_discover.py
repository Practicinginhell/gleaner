"""Tests for "Find websites": Muse Spark web search for feeds, checked before they are offered."""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gleaner import discover
from gleaner.db import Database
from gleaner.models import Question, SourceRow
from gleaner.server import create_app

ATOM = b"""<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Evals blog</title>
<entry><title>How we evaluate agents</title><link href="https://evals.example/post"/><id>1</id><updated>2026-11-01T00:00:00Z</updated>
<summary>Notes.</summary></entry></feed>"""


def test_parse_candidates_reads_fenced_json_and_drops_bad_urls():
    text = 'Here you go:\n```json\n[{"title": "Evals", "feed_url": "https://evals.example/feed", "why": "Weekly evals notes."},\n' \
           '{"title": "Dup", "feed_url": "https://evals.example/feed"}, {"title": "Bad", "feed_url": "javascript:alert(1)"},\n' \
           '{"title": "Plain", "url": "http://plain.example/rss"}]\n```'
    assert discover.parse_candidates(text) == [
        {"title": "Evals", "url": "https://evals.example/feed", "why": "Weekly evals notes."},
        {"title": "Plain", "url": "http://plain.example/rss", "why": ""},
    ]
    assert discover.parse_candidates("no json here") == []
    assert discover.parse_candidates("[not json") == []


def test_search_feeds_asks_muse_spark_with_web_search():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.read())
        reply = '[{"title": "Evals", "feed_url": "https://evals.example/feed", "why": "Weekly."}]'
        return httpx.Response(200, json={"output": [
            {"type": "web_search_call", "id": "ws1", "status": "completed"},
            {"type": "message", "content": [{"type": "output_text", "text": reply, "annotations": []}]},
        ]})

    feeds = discover.search_feeds("How do people evaluate agents?", "k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert feeds == [{"title": "Evals", "url": "https://evals.example/feed", "why": "Weekly."}]
    assert seen["url"] == "https://api.meta.ai/v1/responses" and seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "muse-spark-1.3"
    assert seen["body"]["tools"] == [{"type": "web_search"}]
    assert "How do people evaluate agents?" in seen["body"]["input"]


@pytest.mark.parametrize(("status", "message"), [(401, "rejected MODEL_API_KEY"), (500, "HTTP 500")])
def test_search_feeds_explains_failures(status, message):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status, json={})))
    with pytest.raises(discover.DiscoverError, match=message):
        discover.search_feeds("q", "k", client=client)


def test_check_feeds_keeps_only_working_feeds(monkeypatch):
    """A model can name feeds that don't exist: only ones that parse with items are offered."""
    import gleaner.sources.rss

    monkeypatch.setattr(gleaner.sources.rss, "validate_safe_url", lambda url: url)  # offline: no DNS
    monkeypatch.setattr(discover, "is_safe_external_url", lambda url: "private" not in url)
    pages = {"https://ok.example/feed": httpx.Response(200, content=ATOM),
             "https://gone.example/feed": httpx.Response(404),
             "https://html.example/": httpx.Response(200, content=b"<html><body>hi</body></html>"),
             "http://moved.example/feed": httpx.Response(301, headers={"Location": "https://www.moved.example/feed"}),
             "https://www.moved.example/feed": httpx.Response(200, content=ATOM),
             "https://private.example/feed": httpx.Response(200, content=ATOM),
             "https://to-private.example/feed": httpx.Response(302, headers={"Location": "https://private.example/feed"})}
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pages[str(r.url)]))
    urls = ["https://ok.example/feed", "https://gone.example/feed", "https://html.example/", "http://moved.example/feed",
            "https://private.example/feed", "https://to-private.example/feed"]
    cands = [{"title": u.split("/")[2], "url": u, "why": ""} for u in urls]

    found = asyncio.run(discover.check_feeds(cands, client))

    # a redirect is followed once and the feed is offered under its final address, which a collect can read
    assert found == [
        {"title": "ok.example", "url": "https://ok.example/feed", "why": "", "latest": "How we evaluate agents"},
        {"title": "moved.example", "url": "https://www.moved.example/feed", "why": "", "latest": "How we evaluate agents"},
    ]


def test_parse_candidates_ignores_brackets_around_the_array():
    """Web-search replies carry citations like [1]: the array is still found."""
    arr = '[{"title": "Evals", "feed_url": "https://evals.example/feed", "why": "Weekly."}]'
    for text in (f"I searched [web] and found:\n{arr}", f"{arr}\nSources: [1] https://x [2] https://y"):
        assert [c["url"] for c in discover.parse_candidates(text)] == ["https://evals.example/feed"]


@pytest.mark.parametrize("reply", [b"not json", b"[]", b'{"output": null}', b'{"output": [{"type": "message", "content": null}]}'])
def test_search_feeds_turns_a_malformed_reply_into_a_clear_error(reply):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=reply)))
    try:
        assert discover.search_feeds("q", "k", client=client) == []
    except discover.DiscoverError as exc:
        assert "unexpected" in str(exc)


def _app_with_question(tmp_path):
    db = Database(tmp_path / "d.db")
    db.upsert_question(Question(id="q-ev", text="How do people evaluate agents?", keywords=["evaluate", "agents"]))
    return db, TestClient(create_app(db))


def test_find_feeds_endpoint_needs_the_key_and_a_real_question(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    _, client = _app_with_question(tmp_path)
    res = client.post("/api/questions/q-ev/find-feeds")
    assert res.status_code == 503 and "MODEL_API_KEY" in res.json()["detail"]
    monkeypatch.setenv("MODEL_API_KEY", "k")
    assert client.post("/api/questions/nope/find-feeds").status_code == 404


def test_find_feeds_endpoint_offers_checked_feeds_not_already_added(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "k")
    db, client = _app_with_question(tmp_path)
    db.upsert_source(SourceRow(id="have", type="rss", params={"url": "https://have.example/feed"}, cadence_s=3600))
    asked = {}

    def fake_search(question, api_key):
        asked["question"] = question
        return [{"title": "Have", "url": "https://have.example/feed", "why": ""},
                {"title": "New", "url": "https://new.example/feed", "why": "Weekly."}]

    async def fake_check(cands, client):
        return [{**c, "latest": "Post"} for c in cands]

    monkeypatch.setattr(discover, "search_feeds", fake_search)
    monkeypatch.setattr(discover, "check_feeds", fake_check)

    body = client.post("/api/questions/q-ev/find-feeds").json()

    assert asked["question"] == "How do people evaluate agents?"
    assert body == {"feeds": [{"title": "New", "url": "https://new.example/feed", "why": "Weekly.", "latest": "Post"}],
                    "searched_by": "muse-spark-1.3"}


def test_find_feeds_endpoint_reports_meta_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "k")
    _, client = _app_with_question(tmp_path)

    def fail(question, api_key):
        raise discover.DiscoverError("Meta rejected MODEL_API_KEY.")

    monkeypatch.setattr(discover, "search_feeds", fail)
    res = client.post("/api/questions/q-ev/find-feeds")
    assert res.status_code == 502 and res.json()["detail"] == "Meta rejected MODEL_API_KEY."


def test_every_question_card_offers_find_websites(tmp_path):
    _, client = _app_with_question(tmp_path)
    page = client.get("/").text
    card = page.split('id="question-q-ev"')[1].split("</article>")[0]
    assert "findFeeds(" in card and "Find websites" in card
    assert 'id="feeds-modal"' in page
