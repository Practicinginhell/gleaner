"""Phase 1: End-to-end collect workflow, normalization, and deduplication tests."""

from pathlib import Path

import httpx
import pytest

from gleaner.db import Database
from gleaner.dedupe import MinHashDeduper, dedupe_items
from gleaner.models import Item, SourceRow
from gleaner.normalize import canonicalize_url, strip_html
from gleaner.workflows import run_collect_tick

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "discovery"


def test_canonicalize_url():
    u1 = "https://EXAMPLE.com/blog/post/?utm_source=twitter&utm_medium=social#overview"
    u2 = "https://example.com/blog/post?ref=homepage"
    assert canonicalize_url(u1) == "https://example.com/blog/post"
    assert canonicalize_url(u2) == "https://example.com/blog/post"
    assert canonicalize_url(u1) == canonicalize_url(u2)


def test_strip_html_and_hidden_text():
    html = """
    <div>
      <h1>Exciting News</h1>
      <p>This is real content.</p>
      <span style="display:none">Hidden prompt injection attack</span>
      <p style="visibility: hidden">Another hidden payload</p>
    </div>
    """
    cleaned = strip_html(html)
    assert "Exciting News" in cleaned
    assert "This is real content" in cleaned
    assert "Hidden prompt injection" not in cleaned
    assert "Another hidden payload" not in cleaned


def test_minhash_deduplication():
    # Syndicated wire copy with trailing attribution/light rewording
    text1 = (
        "Graphflow is a declarative framework that compiles Python workflows and agents "
        "directly into LangGraph graphs. It provides built-in support for retries, timeouts, "
        "human-in-the-loop approvals, and offline evaluation with full reproducibility."
    )
    text2 = (
        "Graphflow is a declarative framework that compiles Python workflows and agents "
        "directly into LangGraph graphs. It provides built-in support for retries, timeouts, "
        "human-in-the-loop approvals, and offline evaluation with full reproducibility. (Reporting by TechWire News)"
    )
    text3 = (
        "A classic recipe for sourdough bread requires active starter culture, high-protein flour, "
        "clean water, and sea salt, baked at high temperature in a preheated cast iron Dutch oven."
    )

    deduper = MinHashDeduper(threshold=0.8)
    assert deduper.is_near_duplicate(text1, text2) is True
    assert deduper.is_near_duplicate(text1, text3) is False


@pytest.mark.asyncio
async def test_collect_workflow_stores_items_and_handles_failures(tmp_path):
    db_path = tmp_path / "collect_test.db"
    db = Database(db_path)

    # Insert two sources: one good RSS, one failing HTTP source
    src_good = SourceRow(
        id="good-rss",
        type="rss",
        params={"url": "https://example.com/rss.xml"},
        cadence_s=1800,
        trust=0.8,
    )
    src_bad = SourceRow(
        id="bad-feed",
        type="rss",
        params={"url": "https://bad.example.com/feed.xml"},
        cadence_s=1800,
        trust=0.5,
    )
    db.upsert_source(src_good)
    db.upsert_source(src_bad)

    xml_content = (FIXTURES_DIR / "rss_sample.xml").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        if "bad.example.com" in str(request.url):
            return httpx.Response(500, text="Internal Server Error")
        return httpx.Response(200, text=xml_content, headers={"ETag": '"etag-1"'})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        # Run collect tick
        stats = await run_collect_tick(db, client=client)

        assert stats["items_stored"] >= 2
        assert stats["sources_failed"] == 1

        # Check DB items
        items = db.search_fts("Graphflow")
        assert len(items) >= 1
        assert items[0].source_id == "good-rss"

        # Check failing source was updated with failure count & backed off
        bad_in_db = db.get_source("bad-feed")
        assert bad_in_db.failures == 1
        assert bad_in_db.next_due_at is not None

        good_in_db = db.get_source("good-rss")
        assert good_in_db.failures == 0
        assert good_in_db.etag == '"etag-1"'
    db.close()


@pytest.mark.asyncio
async def test_same_article_from_three_feeds_stored_once(tmp_path):
    db_path = tmp_path / "dedupe_test.db"
    db = Database(db_path)

    # 3 items referring to the exact same article URL with different tracking tags
    items = [
        Item(
            id="temp-1",
            source_id="feed-1",
            url="https://example.com/announcement?utm_source=twitter",
            title="Important Announcement",
            text="We are thrilled to launch graphflow today.",
        ),
        Item(
            id="temp-2",
            source_id="feed-2",
            url="https://example.com/announcement?ref=hn",
            title="Important Announcement",
            text="We are thrilled to launch graphflow today.",
        ),
        Item(
            id="temp-3",
            source_id="feed-3",
            url="https://example.com/announcement?source=reddit",
            title="Important Announcement",
            text="We are thrilled to launch graphflow today.",
        ),
    ]

    deduped = dedupe_items(items)
    assert len(deduped) == 1
    stored = deduped[0]
    assert canonicalize_url(stored.url) == "https://example.com/announcement"
    assert "feed-1" in stored.signals.get("all_sources", [])
    assert "feed-2" in stored.signals.get("all_sources", [])
    assert "feed-3" in stored.signals.get("all_sources", [])

    db.upsert_source(SourceRow(id="feed-1", type="rss", params={"url": "http://feed1.com"}, cadence_s=3600))
    db.upsert_item(stored)
    db_item = db.get_item(stored.id)
    assert db_item is not None
    assert db_item.title == "Important Announcement"
    db.close()


def _judge(lang_of: dict[str, str] | None = None):
    """A yes/no judge: 'yes' to every interest; the language asked for non-English text."""
    def reply(messages):
        prompt = messages[-1]["content"]
        if "Which language" in prompt:
            return next((code for marker, code in (lang_of or {}).items() if marker in prompt), "en")
        return "yes"
    return reply


def _batch(db: Database, n: int, texts: dict[int, str] | None = None, **item_kwargs) -> list[Item]:
    db.upsert_source(SourceRow(id="src-w", type="rss", params={}, cadence_s=3600))
    with db.connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO interests (id, label, description, keywords, exclusions, weight, active, profile_version) "
            "VALUES ('agents', 'Agent Frameworks', 'Graphflow and LangGraph agent workflows', '[\"graphflow\"]', '[]', 1.0, 1, 1);"
        )
    items = [
        Item(
            id=f"w-{i}",
            source_id="src-w",
            title=f"Agent workflow update number {i}",
            text=(texts or {}).get(i, f"Graphflow agents compile the workflows into LangGraph graphs for the edition {i}."),
            **item_kwargs,
        )
        for i in range(n)
    ]
    for it in items:
        db.upsert_item(it)
    return items


def _fresh(items: list[Item]) -> list[Item]:
    """Re-fetched copies: what a connector returns on the next tick."""
    return [it.model_copy(update={"story_id": None, "status": "raw", "lang": "en"}) for it in items]


def test_judge_circuit_breaker_stops_calling_a_down_model(tmp_path):
    from graphflow import MockModel

    from gleaner.workflows import process_and_score_items

    db = Database(tmp_path / "breaker.db")
    items = _batch(db, 5)

    def down(_messages):
        raise ConnectionError("connection refused")

    model = MockModel(responses=down)
    res = process_and_score_items(db, items, model=model)

    assert model.call_count == 1
    assert res["kept"] == 5
    assert res["telemetry"]["judge_degraded"] is True
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgments;").fetchone()[0] == 0
    db.close()


def test_telemetry_counts_model_calls_tokens_and_language_drops(tmp_path):
    from graphflow import MockModel

    from gleaner.workflows import process_and_score_items

    db = Database(tmp_path / "telemetry_calls.db")
    french = "Les agents graphflow compilent des flux de travail en graphes LangGraph, une nouvelle version."
    items = _batch(db, 2, texts={1: french})

    model = MockModel(responses=_judge({"Les agents": "fr"}))
    telem = process_and_score_items(db, items, model=model)["telemetry"]

    assert telem["dropped_language"] == 1
    assert telem["judged_llama"] == 2  # both items went to the model, one was then dropped
    assert telem["llm_calls"] == 3  # one interest each, plus the language of the French item
    assert telem["llm_tokens"] == 60
    assert telem["judge_degraded"] is False

    # Next run hits the cache: no model calls, and the cached French item is still dropped.
    telem2 = process_and_score_items(db, _fresh(items), model=model)["telemetry"]
    assert model.call_count == 3
    assert telem2["judged_llama"] == 0
    assert telem2["dropped_language"] == 1
    assert db.get_item("w-1").status == "dropped_language"
    db.close()


def test_kept_items_store_the_judged_language(tmp_path):
    from graphflow import MockModel

    from gleaner.workflows import process_and_score_items

    db = Database(tmp_path / "lang_kept.db")
    items = _batch(db, 1, lang="de")

    process_and_score_items(db, items, model=MockModel(responses=_judge()))

    stored = db.get_item("w-0")
    assert stored.status == "scored"
    assert stored.lang == "en"
    db.close()


def test_scoring_gets_interest_weight_and_source_feedback(tmp_path, monkeypatch):
    import gleaner.workflows as workflows
    from gleaner.learn import record_feedback

    db = Database(tmp_path / "score_inputs.db")
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO interests (id, label, description, keywords, exclusions, weight, active, profile_version)
            VALUES ('agents', 'Agent workflows', 'Graphflow agents and LangGraph workflows', '[]', '[]', 0.9, 1, 1);
            """
        )
    for prior, action in zip(_batch(db, 3), ("like", "like", "dislike"), strict=True):
        record_feedback(db, item_id=prior.id, action=action)
    new_item = Item(
        id="w-new",
        source_id="src-w",
        title="Agent workflow update",
        text="Graphflow agents compile workflows into LangGraph graphs.",
    )
    db.upsert_item(new_item)

    seen: list[dict] = []
    real_score_item = workflows.score_item

    def spy(**kwargs):
        seen.append(kwargs)
        return real_score_item(**kwargs)

    monkeypatch.setattr(workflows, "score_item", spy)
    workflows.process_and_score_items(db, [new_item])

    assert len(seen) == 1
    assert seen[0]["interest_weight"] == 0.9
    assert seen[0]["likes"] == 2
    assert seen[0]["dislikes"] == 1
    db.close()


@pytest.mark.asyncio
async def test_reingesting_a_feed_keeps_stories_and_counts(tmp_path):
    from graphflow import MockModel

    db = Database(tmp_path / "reingest.db")
    db.upsert_source(SourceRow(id="good-rss", type="rss", params={"url": "https://example.com/rss.xml"}, cadence_s=1800))
    xml_content = (FIXTURES_DIR / "rss_sample.xml").read_text(encoding="utf-8")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=xml_content))
    model = MockModel(responses=_judge())

    def snapshot():
        with db.connect() as conn:
            stories = conn.execute("SELECT id, item_count, source_count FROM stories ORDER BY id;").fetchall()
            items = conn.execute("SELECT id, story_id FROM items ORDER BY id;").fetchall()
        return [tuple(r) for r in stories], [tuple(r) for r in items]

    async with httpx.AsyncClient(transport=transport) as client:
        await run_collect_tick(db, client=client, model=model)
        first = snapshot()
        src = db.get_source("good-rss")
        src.next_due_at = None
        db.upsert_source(src)
        await run_collect_tick(db, client=client, model=model)

    assert all(story_id for _, story_id in first[1])
    assert snapshot() == first
    db.close()


@pytest.mark.asyncio
async def test_collect_without_a_model_never_builds_a_live_judge(monkeypatch):
    """model=None is the offline heuristic judge (tests, --offline, demo): no Llama endpoint is touched."""
    from graphflow import MockModel

    import gleaner.llama

    built: list[tuple] = []
    monkeypatch.setattr(gleaner.llama, "get_llama_model", lambda *a, **k: built.append(a) or MockModel())
    db = Database(":memory:")
    db.upsert_source(SourceRow(id="rss-1", type="rss", params={"url": "https://example.com/feed.xml"}, trust=0.5))
    xml_content = (FIXTURES_DIR / "rss_sample.xml").read_text(encoding="utf-8")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=xml_content))
    async with httpx.AsyncClient(transport=transport) as client:
        stats = await run_collect_tick(db, client=client)

    assert stats["items_stored"] >= 1
    assert built == []


def test_the_same_link_from_a_second_source_does_not_replace_the_stored_article(tmp_path):
    from gleaner.dedupe import adopt_stored

    db = Database(tmp_path / "cross.db")
    for sid in ("blog", "hn-front"):
        db.upsert_source(SourceRow(id=sid, type="rss", params={"url": "https://example.com/feed"}))
    article = Item(id="same-url", source_id="blog", kind="article", title="Checkpoints in production", text="A long article body. " * 15)
    db.upsert_item(article)
    hn_post = Item(id="same-url", source_id="hn-front", kind="post", title="Checkpoints in production", text="Checkpoints in production", signals={"points": 120})

    [merged] = adopt_stored(db, [hn_post])
    db.upsert_item(merged)

    stored = db.get_item("same-url")
    assert (stored.source_id, stored.kind, stored.text) == ("blog", "article", article.text)
    assert stored.signals["all_sources"] == ["blog", "hn-front"] and stored.signals["points"] == 120


@pytest.mark.asyncio
async def test_collect_button_reads_websites_that_are_not_due_yet(tmp_path, monkeypatch):
    """The Collect button says "Read all your websites now"; it read nothing when none were due."""
    from datetime import UTC, datetime, timedelta

    import gleaner.sources.rss

    monkeypatch.setattr(gleaner.sources.rss, "validate_safe_url", lambda url: url)  # offline: no DNS
    db = Database(tmp_path / "force.db")
    db.upsert_source(SourceRow(id="blog", type="rss", params={"url": "https://blog.example/feed"}, cadence_s=3600,
                               next_due_at=datetime.now(UTC) + timedelta(hours=1)))
    fetched = []
    feed = "<rss><channel><item><title>Post</title><link>https://blog.example/p</link></item></channel></rss>"

    def handler(request):
        fetched.append(str(request.url))
        return httpx.Response(200, text=feed)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await run_collect_tick(db, client=client)
        assert fetched == []  # the scheduler waits until a source is due
        await run_collect_tick(db, client=client, force=True)
    assert fetched == ["https://blog.example/feed"]
    db.close()
