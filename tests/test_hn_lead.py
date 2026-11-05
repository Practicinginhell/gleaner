"""HN lead: items gleaner's sources published before they were posted to Hacker News."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from gleaner.db import Database
from gleaner.hn_lead import check_hn_sightings, hn_leads
from gleaner.models import Item, SourceRow

NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)
DAY0 = NOW - timedelta(days=5)


def _db(tmp_path) -> Database:
    db = Database(tmp_path / "hn.db")
    for src, kind in (("arxiv-x", "arxiv"), ("blog", "rss"), ("hn-frontpage", "hn")):
        db.upsert_source(SourceRow(id=src, type=kind, params={}))
    db.upsert_item(Item(id="paper", source_id="arxiv-x", title="A paper", text="t", status="scored",
                        url="https://arxiv.org/abs/2610.12345v1", published_at=DAY0))
    db.upsert_item(Item(id="post", source_id="blog", title="A post", text="t", status="scored",
                        url="https://www.example.com/post/", published_at=DAY0))
    db.upsert_item(Item(id="story", source_id="hn-frontpage", title="On HN", text="t", status="scored",
                        url="https://example.org/x", published_at=DAY0))
    return db


def _hit(url: str, posted: datetime, points: int, object_id: str = "1") -> dict:
    return {"objectID": object_id, "url": url, "points": points, "created_at": posted.isoformat().replace("+00:00", "Z")}


@pytest.mark.asyncio
async def test_items_posted_to_hn_after_the_source_published_them_are_leads(tmp_path):
    db = _db(tmp_path)
    queries = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(request.url.params["query"])
        assert request.url.params["restrictSearchableAttributes"] == "url"
        if request.url.params["query"] == "2610.12345":
            return httpx.Response(200, json={"hits": [
                _hit("https://arxiv.org/abs/2610.12345", DAY0 + timedelta(days=2), 120, "42"),
                _hit("https://arxiv.org/abs/2610.12345v2", DAY0 + timedelta(days=3), 30, "43"),
                _hit("https://example.com/unrelated", DAY0 + timedelta(days=1), 900, "44"),  # search noise
            ]})
        # posted to HN before the blog published it: not a lead
        return httpx.Response(200, json={"hits": [_hit("https://example.com/post", DAY0 - timedelta(hours=3), 50)]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await check_hn_sightings(db, client, now=NOW) == 2
        assert await check_hn_sightings(db, client, now=NOW + timedelta(hours=1)) == 0  # checked today already

    assert sorted(queries) == ["2610.12345", "example.com/post"]  # HN's own stories aren't checked
    leads = hn_leads(db)
    assert [(lead.item.id, lead.points, lead.lead) for lead in leads] == [("paper", 120, timedelta(days=2))]
    assert leads[0].hn_url == "https://news.ycombinator.com/item?id=42"


@pytest.mark.asyncio
async def test_a_failed_hn_search_skips_the_item_and_keeps_going(tmp_path):
    db = _db(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["query"] == "2610.12345":
            return httpx.Response(503)
        return httpx.Response(200, json={"hits": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await check_hn_sightings(db, client, now=NOW) == 1

    assert hn_leads(db) == []


def test_a_lead_shorter_than_the_source_polling_interval_is_not_shown(tmp_path):
    db = _db(tmp_path)
    db.upsert_source(SourceRow(id="blog", type="rss", params={}, cadence_s=3 * 3600))
    db.upsert_hn_sighting("post", hn_id="7", posted_at=DAY0 + timedelta(minutes=8), points=40, checked_at=NOW)
    db.upsert_hn_sighting("paper", hn_id="8", posted_at=DAY0 + timedelta(days=1), points=5, checked_at=NOW)

    # gleaner reads the blog every 3 hours: 8 minutes ahead of HN is not ahead for the user
    assert [lead.item.id for lead in hn_leads(db)] == ["paper"]


@pytest.mark.asyncio
async def test_an_hn_post_of_a_longer_url_is_not_the_item(tmp_path):
    db = _db(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": [
            _hit("https://arxiv.org/abs/2610.123456", DAY0 + timedelta(days=1), 80),  # a different paper
            _hit("https://example.com/post/2", DAY0 + timedelta(days=1), 80),  # a different post
        ]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await check_hn_sightings(db, client, now=NOW)
    assert hn_leads(db) == []


@pytest.mark.asyncio
async def test_homepage_links_do_not_crowd_out_items_that_can_be_checked(tmp_path):
    db = _db(tmp_path)
    for n in range(5):  # newer than the paper and the post, and never checkable
        db.upsert_item(Item(id=f"home{n}", source_id="blog", title="Home", text="t", status="scored",
                            url=f"https://site{n}.example/", published_at=DAY0 + timedelta(hours=n + 1)))

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"hits": []}))) as client:
        assert await check_hn_sightings(db, client, now=NOW, limit=2) == 2


def test_one_hn_post_reached_through_two_websites_is_listed_once(tmp_path):
    """The same paper from two arXiv feeds showed up twice under "Ahead of Hacker News"."""
    db = _db(tmp_path)
    db.upsert_source(SourceRow(id="arxiv-y", type="arxiv", params={}))
    db.upsert_item(Item(id="paper-again", source_id="arxiv-y", title="A paper", text="t", status="scored",
                        url="https://arxiv.org/abs/2610.12345v2", published_at=DAY0 + timedelta(hours=6)))
    db.upsert_hn_sighting("paper", hn_id="8", posted_at=DAY0 + timedelta(days=1), points=5, checked_at=NOW)
    db.upsert_hn_sighting("paper-again", hn_id="8", posted_at=DAY0 + timedelta(days=1), points=5, checked_at=NOW)

    # kept once, through the website that had it first
    assert [lead.item.id for lead in hn_leads(db)] == ["paper"]
