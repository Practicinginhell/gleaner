"""Phase 0: Database migrations, schema, and storage tests."""

import sqlite3

import pytest

from gleaner.db import Database
from gleaner.models import Item, SourceRow


def test_fresh_db_migrates(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    version = db.get_schema_version()
    assert version >= 1

    # Check key tables exist
    tables = db.list_tables()
    assert "sources" in tables
    assert "items" in tables
    assert "stories" in tables
    assert "interests" in tables
    assert "judgments" in tables
    assert "scores" in tables
    assert "schema_version" in tables
    db.close()


def test_migrating_twice_is_noop(tmp_path):
    db_path = tmp_path / "test.db"
    db1 = Database(db_path)
    v1 = db1.get_schema_version()
    db1.close()

    # Re-opening runs migrations again
    db2 = Database(db_path)
    v2 = db2.get_schema_version()
    assert v1 == v2
    db2.close()


def test_wal_and_foreign_keys(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    with db.connect() as conn:
        cursor = conn.cursor()
        mode = cursor.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"
        fk = cursor.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert fk == 1
    db.close()


def test_sqlite_vec_loads_or_falls_back(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    assert db.has_vec is True or db.has_vec is False

    # Insert a vector and query
    item_id = "test-item-1"
    vec = [0.1] * db.vec_dim
    db.upsert_item_vector(item_id, vec)

    similar = db.search_similar_items(vec, limit=5)
    assert len(similar) >= 1
    assert similar[0][0] == item_id
    db.close()


def test_crud_and_fts5(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)

    # Insert a source
    src = SourceRow(
        id="test-src",
        type="rss",
        params={"url": "http://example.com/feed.xml"},
        cadence_s=3600,
        trust=0.8,
    )
    db.upsert_source(src)
    fetched_src = db.get_source("test-src")
    assert fetched_src is not None
    assert fetched_src.id == "test-src"

    # Insert items
    item1 = Item(
        id="item-1",
        source_id="test-src",
        kind="article",
        url="http://example.com/art1",
        title="Agentic Workflows with Graphflow",
        text="A comprehensive guide to compiling graphs into LangGraph.",
    )
    item2 = Item(
        id="item-2",
        source_id="test-src",
        kind="article",
        url="http://example.com/art2",
        title="Baking Sourdough Bread",
        text="How to maintain a sourdough starter in cold weather.",
    )
    db.upsert_item(item1)
    db.upsert_item(item2)

    # Full-text search
    results = db.search_fts("Graphflow")
    assert len(results) == 1
    assert results[0].id == "item-1"

    # Re-inserting item with same id is an idempotent upsert
    db.upsert_item(item1)
    assert len(db.search_fts("Graphflow")) == 1
    db.close()


def test_in_memory_database():
    mem_db = Database(":memory:")
    assert mem_db.get_schema_version() == 1
    assert "items" in mem_db.list_tables()
    item = Item(
        id="mem-1",
        title="In-Memory Test",
        text="Testing in-memory sqlite connection persistence",
    )
    mem_db.upsert_item(item)
    assert mem_db.get_item("mem-1") is not None
    assert len(mem_db.search_fts("Testing")) == 1
    mem_db.close()



def test_old_judgments_table_gains_lang_column(tmp_path):
    db_path = tmp_path / "old.db"
    Database(db_path).close()

    # Simulate a database created before judgments had a lang column.
    conn = sqlite3.connect(db_path)
    if "lang" in [r[1] for r in conn.execute("PRAGMA table_info(judgments);")]:
        conn.execute("ALTER TABLE judgments DROP COLUMN lang;")
    conn.commit()
    conn.close()

    db = Database(db_path)
    with db.connect() as conn:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(judgments);").fetchall()]
    assert "lang" in cols
    db.close()


def test_failed_vec_load_leaves_extension_loading_disabled(tmp_path, monkeypatch):
    import gleaner.db as db_module

    db = Database(tmp_path / "ext.db")
    db.has_vec = True

    class _BrokenVec:
        @staticmethod
        def load(conn):
            raise RuntimeError("load failed")

    monkeypatch.setattr(db_module, "sqlite_vec", _BrokenVec, raising=False)
    with db.connect() as conn, pytest.raises(sqlite3.OperationalError, match="not authorized"):
        conn.execute("SELECT load_extension('/nonexistent/ext');")


def _similarity_db(tmp_path, monkeypatch, use_vec: bool) -> Database:
    import gleaner.db as db_module

    if use_vec and not db_module.SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    if not use_vec:
        monkeypatch.setattr(db_module, "SQLITE_VEC_AVAILABLE", False)
    db = Database(tmp_path / f"sim_{use_vec}.db", vec_dim=4)
    assert db.has_vec is use_vec
    return db



@pytest.mark.parametrize("use_vec", [True, False], ids=["sqlite-vec", "fallback"])
def test_search_similar_items_returns_descending_similarity(tmp_path, monkeypatch, use_vec):
    db = _similarity_db(tmp_path, monkeypatch, use_vec)
    db.upsert_item_vector("exact", [1.0, 0.0, 0.0, 0.0])
    db.upsert_item_vector("near", [0.8, 0.6, 0.0, 0.0])
    db.upsert_item_vector("far", [0.0, 0.0, 1.0, 0.0])

    results = db.search_similar_items([1.0, 0.0, 0.0, 0.0], limit=3)

    assert [r[0] for r in results] == ["exact", "near", "far"]
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)
    assert scores[1] == pytest.approx(0.8, abs=1e-5)
    assert scores[2] == pytest.approx(0.0, abs=1e-5)
    db.close()


def test_reupserting_item_without_story_keeps_its_story(tmp_path):
    db = Database(tmp_path / "story_keep.db")
    item = Item(id="it-s", title="Story item", text="text", story_id="story-1")
    db.upsert_item(item)

    db.upsert_item(item.model_copy(update={"story_id": None}))

    assert db.get_item("it-s").story_id == "story-1"
    db.close()
