"""Phase 2: Semantic story clustering, scoring, MMR selector, and feed generation tests."""

from datetime import UTC, datetime, timedelta

from gleaner.db import Database
from gleaner.embed import FakeEmbedder
from gleaner.feed import generate_feed, select_feed_items
from gleaner.models import Item, SourceRow
from gleaner.scoring import score_item
from gleaner.stories import assign_to_story


def test_semantic_story_clustering(tmp_path):
    db = Database(tmp_path / "stories.db")
    embedder = FakeEmbedder(dim=db.vec_dim)

    # Two articles about the exact same release
    item1 = Item(
        id="rel-1",
        source_id="gh-1",
        title="Graphflow v0.3.0 Released",
        text="Graphflow v0.3.0 is out with declarative fan_out and approval support.",
    )
    item2 = Item(
        id="rel-2",
        source_id="blog-1",
        title="Announcing Graphflow v0.3.0 Release",
        text="Graphflow v0.3.0 is out with declarative fan_out and approval support.",
    )
    # An unrelated article
    item3 = Item(
        id="art-3",
        source_id="news-1",
        title="Astronomers Spot New Exoplanet",
        text="A distant exoplanet with water vapor detected using infrared spectroscopy.",
    )

    vec1 = embedder.embed([item1.title + " " + item1.text])[0]
    vec2 = embedder.embed([item2.title + " " + item2.text])[0]
    vec3 = embedder.embed([item3.title + " " + item3.text])[0]

    story_id_1 = assign_to_story(item1, vec1, db, threshold=0.75)
    story_id_2 = assign_to_story(item2, vec2, db, threshold=0.75)
    story_id_3 = assign_to_story(item3, vec3, db, threshold=0.75)

    assert story_id_1 == story_id_2
    assert story_id_3 != story_id_1

    # Verify story record
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM stories WHERE id = ?;", (story_id_1,)).fetchone()
        assert row["source_count"] == 2
        assert row["item_count"] == 2
    db.close()


def test_scoring_fresher_item_outranks_older():
    now = datetime.now(UTC)
    fresh_item = Item(
        id="fresh",
        kind="news",
        title="Important News",
        text="Breaking update today.",
        published_at=now,
    )
    old_item = Item(
        id="old",
        kind="news",
        title="Important News",
        text="Breaking update today.",
        published_at=now - timedelta(days=5),
    )

    vec = [0.1] * 64
    score_fresh = score_item(fresh_item, vec, embed_sim=0.8, source_trust=0.8, now=now)
    score_old = score_item(old_item, vec, embed_sim=0.8, source_trust=0.8, now=now)

    assert score_fresh.final > score_old.final
    assert score_fresh.components.relevance == score_old.components.relevance
    assert score_fresh.components.decay > score_old.components.decay


def test_selector_one_card_per_story_and_source_caps(tmp_path):
    now = datetime.now(UTC)
    # Story 1 has two coverage items from different sources
    item_s1_a = Item(id="s1-a", source_id="src-a", story_id="story-1", title="Release A", published_at=now)
    item_s1_b = Item(id="s1-b", source_id="src-b", story_id="story-1", title="Release B", published_at=now)

    # 4 distinct items from source-a
    items_src_a = [
        Item(id=f"sa-{i}", source_id="src-a", story_id=f"story-a-{i}", title=f"Post A {i}", published_at=now)
        for i in range(4)
    ]

    candidates = [
        (item_s1_a, 0.95, [0.1] * 64),
        (item_s1_b, 0.90, [0.1] * 64),
    ] + [(it, 0.85 - i * 0.05, [0.2 + i * 0.1] * 64) for i, it in enumerate(items_src_a)]

    # Select with limit 20, max_per_source 3
    selected = select_feed_items(candidates, max_per_source=3, limit=20)

    # Check 1: One card per story
    story_ids = [it.story_id for it, _, _ in selected]
    assert story_ids.count("story-1") == 1

    # Check 2: Source cap <= 3 holds
    src_a_count = sum(1 for it, _, _ in selected if it.source_id == "src-a")
    assert src_a_count <= 3


def _seed_scored(db, item, final, vec=None, components=None):
    """Store an item with a score row, as the scoring workflow would."""
    import json

    db.upsert_item(item)
    db.upsert_item_vector(item.id, vec or FakeEmbedder(dim=db.vec_dim).embed([item.title])[0])
    comps = components or {"relevance": final, "novelty": final, "quality": final, "momentum": final,
                           "actionability": final, "decay": 1.0}
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO scores (item_id, profile_version, components, final) VALUES (?, 1, ?, ?);",
            (item.id, json.dumps(comps), final),
        )


def test_feed_hides_quarantined_items_and_muted_sources():
    db = Database(":memory:")
    for src_id in ("src-ok", "src-muted"):
        db.upsert_source(SourceRow(id=src_id, type="rss", params={}, cadence_s=3600))
    _seed_scored(db, Item(id="ok", source_id="src-ok", title="Agent graphs", status="scored"), 0.5)
    _seed_scored(db, Item(id="poisoned", source_id="src-ok", title="Ignore rules", status="scored"), 0.9)
    _seed_scored(db, Item(id="muted", source_id="src-muted", title="Muted post", status="scored"), 0.8)

    # Later quarantined, and its source muted, after both were scored
    db.upsert_item(Item(id="poisoned", source_id="src-ok", title="Ignore rules", status="dropped_prompt_guard"))
    db.upsert_source(SourceRow(id="src-muted", type="rss", params={}, cadence_s=3600, enabled=False))

    ids = [c.item.id for c in generate_feed(db, limit=20)]
    assert ids == ["ok"]
    db.close()


def test_feed_reserves_about_ten_percent_of_slots_for_exploring_items():
    db = Database(":memory:")
    for s in range(10):
        db.upsert_source(SourceRow(id=f"src-{s}", type="rss", params={}, cadence_s=3600))
    # More scored items than the feed query's candidate window, so exploring items must not be crowded out
    for i in range(120):
        _seed_scored(
            db,
            Item(id=f"main-{i}", source_id=f"src-{i % 10}", title=f"Main topic {i} words{i}", status="scored"),
            0.9 - i * 0.005,
        )
    for i in range(5):
        _seed_scored(
            db,
            Item(id=f"explore-{i}", source_id=f"src-{i}", title=f"Adjacent {i} thing{i}", status="exploring"),
            0.1,
        )

    cards = generate_feed(db, limit=20)
    assert len(cards) == 20
    exploring = [c for c in cards if c.exploring]
    # ~10% of 20 slots, even though every exploring item scores below every main item
    assert len(exploring) == 2
    assert all(c.item.status == "exploring" for c in exploring)
    assert all(not c.exploring for c in cards if c.item.status == "scored")
    db.close()


def test_feed_orders_by_score_recomputed_at_read_time_in_every_mode():
    db = Database(":memory:")
    db.upsert_source(SourceRow(id="src-a", type="rss", params={}, cadence_s=3600))
    db.upsert_source(SourceRow(id="src-b", type="rss", params={}, cadence_s=3600))
    now = datetime.now(UTC)
    comps = {"relevance": 0.8, "novelty": 0.8, "quality": 0.8, "momentum": 0.8, "actionability": 0.8, "decay": 1.0}
    # Stored final says "stale" is best, but it is 5 days old news: decay at read time drops it
    _seed_scored(
        db,
        Item(id="stale", source_id="src-a", kind="news", title="Old news", published_at=now - timedelta(days=5),
             status="scored"),
        0.9, vec=[1.0] + [0.0] * 63, components=comps,
    )
    _seed_scored(
        db,
        Item(id="fresh", source_id="src-b", kind="news", title="Fresh news", published_at=now, status="scored"),
        0.5, vec=[0.0, 1.0] + [0.0] * 62, components=comps,
    )
    for mode in ("balanced", "deep_research", "breaking_radar", "actionable"):
        cards = generate_feed(db, limit=20, mode=mode)
        assert [c.item.id for c in cards] == ["fresh", "stale"], mode
        assert cards[0].score >= cards[1].score
    db.close()


def test_a_long_summary_ends_at_a_word_with_an_ellipsis():
    """Cards showed "…start a new agent session from scratch for eac": cut mid-word, no ellipsis."""
    from gleaner.normalize import short_summary

    text = "Most evaluations start a new agent session from scratch for each task. " * 5
    out = short_summary(text)
    assert len(out) <= 201 and out.endswith("…")
    assert out[:-1] == text[: len(out) - 1] and text[len(out) - 1] == " "  # ends on a whole word
    assert short_summary("Short text.") == "Short text."
