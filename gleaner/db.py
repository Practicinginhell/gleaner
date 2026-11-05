"""Database management, migrations, and storage for discovery feed."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import struct
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gleaner.embed import cosine_similarity
from gleaner.models import (
    Answer,
    Claim,
    Item,
    Question,
    Sentence,
    SourceRow,
    TrackRecord,
)

try:
    import sqlite_vec

    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

logger = logging.getLogger(__name__)


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec into ``conn``.

    Extension loading is switched off again even if loading fails.
    """
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def _json_default(val: Any) -> Any:
    """Serialize a value that ``json.dumps`` can't.

    Datetimes become ISO 8601; anything else becomes ``str``.
    """
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


# Sources only the dashboard's demo buttons create (sample deadlines, the injection demo): their
# items are not the user's reading, so measured totals leave them out
DEMO_SOURCE_IDS = ("opportunities-feed", "untrusted-feed")


class Database:
    """Manages SQLite connection, schema migrations, and relational + vector storage.

    Attributes:
        path: Database file path, or ':memory:'.
        vec_dim: Embedding dimension of the sqlite-vec tables.
        has_vec: Whether sqlite-vec is loaded; when False, vectors go to the ``*_vec_fallback`` tables and
            similarity is computed in Python.
    """

    def __init__(self, path: str | Path = ":memory:", vec_dim: int = 64) -> None:
        """Open the database and apply migrations.

        An in-memory database keeps one shared connection for its lifetime.

        Args:
            path: Database file path, or ':memory:'.
            vec_dim: Embedding dimension of the sqlite-vec tables.
        """
        self.path = str(path)
        self.vec_dim = vec_dim
        self.has_vec = False
        self._mem_conn: sqlite3.Connection | None = None
        self._check_sqlite_vec()
        if self.path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
            self._mem_conn.execute("PRAGMA foreign_keys = ON;")
            if self.has_vec:
                try:
                    _load_sqlite_vec(self._mem_conn)
                except Exception as e:  # noqa: BLE001 - optional sqlite-vec extension: logged, cosine fallback
                    logger.warning("sqlite-vec failed to load: %s", e)
        self.migrate()

    def _check_sqlite_vec(self) -> None:
        """Set ``has_vec`` to whether sqlite-vec can be loaded.

        A failure is logged, not raised.
        """
        if not SQLITE_VEC_AVAILABLE:
            self.has_vec = False
            return
        try:
            test_conn = sqlite3.connect(":memory:")
            try:
                _load_sqlite_vec(test_conn)
            finally:
                test_conn.close()
            self.has_vec = True
        except Exception as e:  # noqa: BLE001 - optional sqlite-vec extension: logged, cosine fallback
            logger.info("sqlite-vec unavailable, using the cosine fallback: %s", e)
            self.has_vec = False

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection]:
        """Open a connection that commits on success and rolls back on an exception.

        A file database gets a new connection (WAL mode, foreign keys on, sqlite-vec loaded when
        available) that is closed afterwards; an in-memory database reuses its shared connection.

        Yields:
            The connection.
        """
        if self._mem_conn is not None:
            try:
                yield self._mem_conn
                self._mem_conn.commit()
            except Exception:
                self._mem_conn.rollback()
                raise
            return

        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if self.path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")

        if self.has_vec:
            try:
                _load_sqlite_vec(conn)
            except Exception as e:  # noqa: BLE001 - optional sqlite-vec extension: logged, cosine fallback
                logger.warning("sqlite-vec failed to load: %s", e)

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def close(self) -> None:
        """Close database connection."""
        if self._mem_conn is not None:
            self._mem_conn.close()
            self._mem_conn = None

    def get_schema_version(self) -> int:
        """Return the highest applied schema version, or 0 for a new database.

        Creates the ``schema_version`` table if it doesn't exist.
        """
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TIMESTAMP NOT NULL);"
            )
            row = conn.execute("SELECT MAX(version) FROM schema_version;").fetchone()
            return row[0] if (row and row[0] is not None) else 0

    def list_tables(self) -> list[str]:
        """Return the names of all tables, including virtual tables."""
        with self.connect() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [row["name"] for row in cursor.fetchall()]

    def migrate(self) -> None:
        """Bring the schema up to date.

        A new database gets the full schema (migration 1). An existing one gets the tables and
        columns added since, each created only if missing.
        """
        current_version = self.get_schema_version()
        if current_version < 1:
            self._apply_migration_1()
        else:
            with self.connect() as conn:
                tables = [r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
                if "sources" in tables:
                    source_cols = [r["name"] for r in conn.execute("PRAGMA table_info(sources);").fetchall()]
                    if "enabled" not in source_cols:
                        conn.execute("ALTER TABLE sources ADD COLUMN enabled INTEGER DEFAULT 1;")
                if "judgments" in tables:
                    judgment_cols = [r["name"] for r in conn.execute("PRAGMA table_info(judgments);").fetchall()]
                    if "lang" not in judgment_cols:
                        conn.execute("ALTER TABLE judgments ADD COLUMN lang TEXT DEFAULT 'en';")
                conn.execute(
                    """
                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    linked_interests TEXT,
                    watch TEXT,
                    profile_version INTEGER NOT NULL DEFAULT 1,
                    created_at TIMESTAMP
                );
                """
                )
                conn.execute(
                    """
                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY,
                    question_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    stance TEXT NOT NULL DEFAULT 'supports',
                    quote TEXT NOT NULL,
                    origin_key TEXT NOT NULL,
                    model TEXT NOT NULL DEFAULT 'default',
                    created_at TIMESTAMP
                );
                """
                )
                conn.execute(
                    """
                CREATE TABLE IF NOT EXISTS answers (
                    question_id TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    sentences TEXT NOT NULL,
                    confidence TEXT NOT NULL DEFAULT 'insufficient',
                    counterpoint_claim_id TEXT,
                    what_changed TEXT,
                    what_changed_by TEXT,
                    synthesized_by TEXT,
                    created_at TIMESTAMP,
                    PRIMARY KEY(question_id, version)
                );
                """
                )
                conn.execute(
                    """
                CREATE TABLE IF NOT EXISTS track_record (
                    id TEXT PRIMARY KEY,
                    question_id TEXT NOT NULL,
                    answer_version INTEGER NOT NULL,
                    verdict TEXT,
                    hn_lead_days REAL,
                    created_at TIMESTAMP
                );
                """
                )
                if "questions" in tables:
                    q_cols = [r["name"] for r in conn.execute("PRAGMA table_info(questions);").fetchall()]
                    if "watch" not in q_cols:
                        conn.execute("ALTER TABLE questions ADD COLUMN watch TEXT;")
                conn.execute(
                    """
                CREATE TABLE IF NOT EXISTS hn_sightings (
                    item_id TEXT PRIMARY KEY,
                    hn_id TEXT,
                    posted_at TIMESTAMP,
                    points INTEGER,
                    checked_at TIMESTAMP NOT NULL
                );
                """
                )
                answer_cols = [r["name"] for r in conn.execute("PRAGMA table_info(answers);").fetchall()]
                if "what_changed_by" not in answer_cols:
                    conn.execute("ALTER TABLE answers ADD COLUMN what_changed_by TEXT;")
                if "synthesized_by" not in answer_cols:
                    conn.execute("ALTER TABLE answers ADD COLUMN synthesized_by TEXT;")
                conn.execute(
                    """
                CREATE TABLE IF NOT EXISTS telemetry (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at TIMESTAMP
                );
                """
                )
                conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")

    def _apply_migration_1(self) -> None:
        """Create the full schema and record version 1.

        The vector tables are sqlite-vec virtual tables when ``has_vec`` is True, else plain
        ``*_vec_fallback`` tables holding JSON.
        """
        with self.connect() as conn:
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                params TEXT NOT NULL,
                cadence_s INTEGER NOT NULL,
                next_due_at TIMESTAMP,
                etag TEXT,
                last_modified TEXT,
                failures INTEGER DEFAULT 0,
                trust REAL DEFAULT 0.5,
                enabled INTEGER DEFAULT 1
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY,
                source_id TEXT REFERENCES sources(id),
                kind TEXT NOT NULL,
                url TEXT,
                title TEXT NOT NULL,
                author TEXT,
                published_at TIMESTAMP,
                fetched_at TIMESTAMP,
                lang TEXT,
                text TEXT,
                content_hash TEXT,
                signals TEXT,
                story_id TEXT,
                status TEXT
            );
            """
            )
            conn.execute(
                """
            CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
                id UNINDEXED,
                title,
                text
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS stories (
                id TEXT PRIMARY KEY,
                centroid TEXT,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                item_count INTEGER DEFAULT 1,
                source_count INTEGER DEFAULT 1
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS interests (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                description TEXT NOT NULL,
                keywords TEXT,
                exclusions TEXT,
                weight REAL DEFAULT 1.0,
                kind TEXT DEFAULT 'topic',
                active INTEGER DEFAULT 1,
                profile_version INTEGER DEFAULT 1
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS judgments (
                item_id TEXT PRIMARY KEY REFERENCES items(id),
                profile_version INTEGER,
                model TEXT,
                relevance TEXT,
                novelty REAL,
                actionability REAL,
                type TEXT,
                why TEXT,
                summary TEXT,
                judged_at TIMESTAMP,
                lang TEXT DEFAULT 'en'
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS scores (
                item_id TEXT PRIMARY KEY REFERENCES items(id),
                profile_version INTEGER,
                components TEXT,
                final REAL,
                computed_at TIMESTAMP
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS impressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT REFERENCES items(id),
                ts TIMESTAMP,
                action TEXT,
                dwell_s REAL
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT REFERENCES items(id),
                ts TIMESTAMP,
                action TEXT,
                notes TEXT
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS opportunities (
                id TEXT PRIMARY KEY,
                item_id TEXT REFERENCES items(id),
                type TEXT,
                deadline TIMESTAMP,
                eligibility TEXT,
                link TEXT,
                amount TEXT,
                status TEXT
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS trends (
                id TEXT PRIMARY KEY,
                label TEXT,
                story_ids TEXT,
                z_score REAL,
                window TEXT,
                detected_at TIMESTAMP
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS digests (
                id TEXT PRIMARY KEY,
                date TEXT,
                content TEXT,
                rendered_html TEXT,
                created_at TIMESTAMP
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                workflow TEXT,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                status TEXT,
                stats TEXT,
                errors TEXT
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS fetch_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT REFERENCES sources(id),
                fetched_at TIMESTAMP,
                status_code INTEGER,
                item_count INTEGER,
                error TEXT
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                linked_interests TEXT,
                watch TEXT,
                profile_version INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                text TEXT NOT NULL,
                stance TEXT NOT NULL,
                quote TEXT NOT NULL,
                origin_key TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TIMESTAMP
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS answers (
                question_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                sentences TEXT NOT NULL,
                confidence TEXT NOT NULL,
                counterpoint_claim_id TEXT,
                what_changed TEXT,
                what_changed_by TEXT,
                synthesized_by TEXT,
                created_at TIMESTAMP,
                PRIMARY KEY (question_id, version)
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS track_record (
                id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL,
                answer_version INTEGER NOT NULL,
                verdict TEXT,
                hn_lead_days REAL,
                created_at TIMESTAMP
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS hn_sightings (
                item_id TEXT PRIMARY KEY,
                hn_id TEXT,
                posted_at TIMESTAMP,
                points INTEGER,
                checked_at TIMESTAMP NOT NULL
            );
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS telemetry (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TIMESTAMP
            );
            """
            )

            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")

            if self.has_vec:
                conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS item_vec USING vec0(item_id text primary key, embedding float[{self.vec_dim}]);"
                )
                conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS interest_vec USING vec0(interest_id text primary key, embedding float[{self.vec_dim}]);"
                )
            else:
                conn.execute("CREATE TABLE IF NOT EXISTS item_vec_fallback (item_id TEXT PRIMARY KEY, embedding TEXT);")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS interest_vec_fallback (interest_id TEXT PRIMARY KEY, embedding TEXT);"
                )

            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (1, ?);",
                (datetime.now(UTC).isoformat(),),
            )

    # Sources CRUD
    def upsert_source(self, src: SourceRow) -> None:
        """Insert a source, or replace every field of the existing one with this id.

        Args:
            src: The source to store.
        """
        with self.connect() as conn:
            conn.execute(
                """
            INSERT INTO sources (id, type, params, cadence_s, next_due_at, etag, last_modified, failures, trust, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                type = excluded.type,
                params = excluded.params,
                cadence_s = excluded.cadence_s,
                next_due_at = excluded.next_due_at,
                etag = excluded.etag,
                last_modified = excluded.last_modified,
                failures = excluded.failures,
                trust = excluded.trust,
                enabled = excluded.enabled;
            """,
                (
                    src.id,
                    src.type,
                    json.dumps(src.params),
                    src.cadence_s,
                    src.next_due_at.isoformat() if src.next_due_at else None,
                    src.etag,
                    src.last_modified,
                    src.failures,
                    src.trust,
                    1 if src.enabled else 0,
                ),
            )

    def get_source(self, source_id: str) -> SourceRow | None:
        """Return the source with this id, or None."""
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM sources WHERE id = ?;", (source_id,)).fetchone()
            if not row:
                return None
            return SourceRow(
                id=row["id"],
                type=row["type"],
                params=json.loads(row["params"]),
                cadence_s=row["cadence_s"],
                next_due_at=datetime.fromisoformat(row["next_due_at"]) if row["next_due_at"] else None,
                etag=row["etag"],
                last_modified=row["last_modified"],
                failures=row["failures"],
                trust=row["trust"],
                enabled=bool(row["enabled"]) if "enabled" in row.keys() else True,
            )

    def list_sources(self) -> list[SourceRow]:
        """Return every source, enabled or not."""
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM sources;").fetchall()
            return [
                SourceRow(
                    id=r["id"],
                    type=r["type"],
                    params=json.loads(r["params"]),
                    cadence_s=r["cadence_s"],
                    next_due_at=datetime.fromisoformat(r["next_due_at"]) if r["next_due_at"] else None,
                    etag=r["etag"],
                    last_modified=r["last_modified"],
                    failures=r["failures"],
                    trust=r["trust"],
                    enabled=bool(r["enabled"]) if "enabled" in r.keys() else True,
                )
                for r in rows
            ]

    def get_fetch_logs(self, source_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent fetch log entries for a source.

        Args:
            source_id: Id of the source.
            limit: Maximum number of entries to return.

        Returns:
            Entries newest first, each with id, source_id, fetched_at, status_code, item_count and error.
        """
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, source_id, fetched_at, status_code, item_count, error FROM fetch_log WHERE source_id = ? ORDER BY id DESC LIMIT ?;",
                (source_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # Items CRUD and FTS
    def upsert_item(self, item: Item) -> None:
        """Insert or update an item by id, and sync its full-text index.

        A re-fetched item arrives without its story, so an existing ``story_id`` is kept when the
        new one is None.

        Args:
            item: The item to store.
        """
        with self.connect() as conn:
            conn.execute(
                """
            INSERT INTO items (
                id, source_id, kind, url, title, author, published_at, fetched_at,
                lang, text, content_hash, signals, story_id, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_id = excluded.source_id,
                kind = excluded.kind,
                url = excluded.url,
                title = excluded.title,
                author = excluded.author,
                published_at = excluded.published_at,
                fetched_at = excluded.fetched_at,
                lang = excluded.lang,
                text = excluded.text,
                content_hash = excluded.content_hash,
                signals = excluded.signals,
                -- a re-fetched item arrives without its story; keep the one it already has
                story_id = COALESCE(excluded.story_id, items.story_id),
                status = excluded.status;
            """,
                (
                    item.id,
                    item.source_id,
                    item.kind,
                    item.url,
                    item.title,
                    item.author,
                    item.published_at.isoformat() if item.published_at else None,
                    item.fetched_at.isoformat() if item.fetched_at else None,
                    item.lang,
                    item.text,
                    item.content_hash,
                    json.dumps(item.signals),
                    item.story_id,
                    item.status,
                ),
            )
            # Sync to FTS5
            conn.execute("DELETE FROM items_fts WHERE id = ?;", (item.id,))
            conn.execute("INSERT INTO items_fts (id, title, text) VALUES (?, ?, ?);", (item.id, item.title, item.text))

    def get_item(self, item_id: str) -> Item | None:
        """Return the item with this id, or None."""
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM items WHERE id = ?;", (item_id,)).fetchone()
            if not row:
                return None
            return self._row_to_item(row)

    def _row_to_item(self, row: sqlite3.Row) -> Item:
        """Build an Item from an ``items`` row."""
        return Item(
            id=row["id"],
            source_id=row["source_id"],
            kind=row["kind"],
            url=row["url"],
            title=row["title"],
            author=row["author"],
            published_at=datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
            fetched_at=datetime.fromisoformat(row["fetched_at"]) if row["fetched_at"] else None,
            lang=row["lang"] or "en",
            text=row["text"] or "",
            content_hash=row["content_hash"],
            signals=json.loads(row["signals"]) if row["signals"] else {},
            story_id=row["story_id"],
            status=row["status"] or "raw",
        )

    def search_fts(self, query: str, limit: int = 20) -> list[Item]:
        """Full-text search items by title and text, best match first.

        The query is split into words and any word may match, so FTS5 syntax in ``query`` is
        never interpreted.

        Args:
            query: Free text to search for.
            limit: Maximum number of items to return.

        Returns:
            The matching items; empty when ``query`` has no words.
        """
        terms = re.findall(r"\b\w+\b", query)
        if not terms:
            return []
        safe_query = " OR ".join(f'"{t}"' for t in terms)
        with self.connect() as conn:
            rows = conn.execute(
                """
            SELECT items.* FROM items_fts
            JOIN items ON items.id = items_fts.id
            WHERE items_fts MATCH ?
            ORDER BY rank
            LIMIT ?;
            """,
                (safe_query, limit),
            ).fetchall()
            return [self._row_to_item(r) for r in rows]

    # Vectors
    def upsert_item_vector(self, item_id: str, vec: list[float]) -> None:
        """Store or replace an item's embedding.

        Args:
            item_id: Id of the item.
            vec: The embedding.
        """
        with self.connect() as conn:
            if self.has_vec:
                raw = sqlite_vec.serialize_float32(vec)
                conn.execute("DELETE FROM item_vec WHERE item_id = ?;", (item_id,))
                conn.execute("INSERT INTO item_vec (item_id, embedding) VALUES (?, ?);", (item_id, raw))
            else:
                conn.execute(
                    "INSERT INTO item_vec_fallback (item_id, embedding) VALUES (?, ?) ON CONFLICT(item_id) DO UPDATE SET embedding = excluded.embedding;",
                    (item_id, json.dumps(vec)),
                )

    def get_item_vector(self, item_id: str) -> list[float] | None:
        """Return the item's stored embedding, or None."""
        with self.connect() as conn:
            if self.has_vec:
                row = conn.execute("SELECT embedding FROM item_vec WHERE item_id = ?;", (item_id,)).fetchone()
                if row and row["embedding"]:
                    blob = row["embedding"]
                    return list(struct.unpack(f"{len(blob)//4}f", blob))
            else:
                row = conn.execute("SELECT embedding FROM item_vec_fallback WHERE item_id = ?;", (item_id,)).fetchone()
                if row and row["embedding"]:
                    return json.loads(row["embedding"])
        return None

    def upsert_interest_vector(self, interest_id: str, vec: list[float]) -> None:
        """Store or replace an interest's embedding.

        Args:
            interest_id: Id of the interest.
            vec: The embedding.
        """
        with self.connect() as conn:
            if self.has_vec:
                raw = sqlite_vec.serialize_float32(vec)
                conn.execute("DELETE FROM interest_vec WHERE interest_id = ?;", (interest_id,))
                conn.execute("INSERT INTO interest_vec (interest_id, embedding) VALUES (?, ?);", (interest_id, raw))
            else:
                conn.execute(
                    "INSERT INTO interest_vec_fallback (interest_id, embedding) VALUES (?, ?) ON CONFLICT(interest_id) DO UPDATE SET embedding = excluded.embedding;",
                    (interest_id, json.dumps(vec)),
                )

    def get_interest_vector(self, interest_id: str) -> list[float] | None:
        """Return the interest's stored embedding, or None."""
        with self.connect() as conn:
            if self.has_vec:
                row = conn.execute("SELECT embedding FROM interest_vec WHERE interest_id = ?;", (interest_id,)).fetchone()
                if row and row["embedding"]:
                    blob = row["embedding"]
                    return list(struct.unpack(f"{len(blob)//4}f", blob))
            else:
                row = conn.execute("SELECT embedding FROM interest_vec_fallback WHERE interest_id = ?;", (interest_id,)).fetchone()
                if row and row["embedding"]:
                    return json.loads(row["embedding"])
        return None

    def search_similar_items(self, query_vec: list[float], limit: int = 20) -> list[tuple[str, float]]:
        """Return up to ``limit`` (item_id, cosine similarity) pairs, most similar first.

        Both storage paths return the same measure: sqlite-vec finds the nearest
        neighbours by L2 distance, then their cosine similarity is computed here.

        Args:
            query_vec: The embedding to compare against.
            limit: Maximum number of pairs to return.

        Returns:
            (item_id, cosine similarity) pairs, most similar first.
        """
        with self.connect() as conn:
            if self.has_vec:
                q = sqlite_vec.serialize_float32(query_vec)
                rows = conn.execute(
                    "SELECT item_id, embedding FROM item_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?;",
                    (q, limit),
                ).fetchall()
                knn = [
                    (r["item_id"], cosine_similarity(query_vec, list(struct.unpack(f"{len(r['embedding']) // 4}f", r["embedding"]))))
                    for r in rows
                ]
                knn.sort(key=lambda x: x[1], reverse=True)
                return knn
            else:
                rows = conn.execute("SELECT item_id, embedding FROM item_vec_fallback;").fetchall()
                scored: list[tuple[str, float]] = []
                for r in rows:
                    stored_vec = json.loads(r["embedding"])
                    sim = cosine_similarity(query_vec, stored_vec)
                    scored.append((r["item_id"], sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[:limit]

    def sync_embedder(self, model_id: str) -> bool:
        """Record which embedding model made the stored vectors (ADR-0003).

        When ``model_id`` differs from the recorded one, the stored item and interest
        vectors belong to another vector space: they are deleted so they get re-embedded,
        and True is returned. The first model seen is recorded as-is.

        Args:
            model_id: Id of the embedding model in use.

        Returns:
            True when stored vectors were deleted, else False.
        """
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'embedder_model_id';").fetchone()
            if row is not None and row["value"] == model_id:
                return False
            stale = row is not None
            if stale:
                for table in ("item_vec", "interest_vec") if self.has_vec else ("item_vec_fallback", "interest_vec_fallback"):
                    conn.execute(f"DELETE FROM {table};")  # noqa: S608 - table name comes from the fixed tuple above
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('embedder_model_id', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value;",
                (model_id,),
            )
            return stale

    def set_telemetry(self, key: str, data: dict[str, Any]) -> None:
        """Store or replace a telemetry record, stamped with the current time.

        Args:
            key: Record key, e.g. 'latest'.
            data: JSON-serializable telemetry data.
        """
        now = datetime.now(UTC).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
            INSERT INTO telemetry (key, data, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                data = excluded.data,
                updated_at = excluded.updated_at;
            """,
                (key, json.dumps(data), now),
            )

    def get_telemetry(self, key: str = "latest") -> dict[str, Any]:
        """Return a telemetry record's data.

        Args:
            key: Record key.

        Returns:
            The stored data; an empty dict when the record is missing or not valid JSON (logged).
        """
        with self.connect() as conn:
            row = conn.execute("SELECT data FROM telemetry WHERE key = ?;", (key,)).fetchone()
            if row and row["data"]:
                try:
                    return json.loads(row["data"])
                except json.JSONDecodeError:
                    logger.warning("Telemetry %r is not valid JSON; ignoring it", key)
                    return {}
        return {}

    def upsert_question(self, q: Question) -> None:
        """Insert or update a question by id.

        An update keeps the stored ``created_at``.

        Args:
            q: The question to store.
        """
        with self.connect() as conn:
            conn.execute(
                """
            INSERT INTO questions (id, text, active, linked_interests, watch, profile_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                text = excluded.text,
                active = excluded.active,
                linked_interests = excluded.linked_interests,
                watch = excluded.watch,
                profile_version = excluded.profile_version;
            """,
                (
                    q.id,
                    q.text,
                    int(q.active),
                    json.dumps(q.linked_interests),
                    q.watch,
                    q.profile_version,
                    q.created_at.isoformat(),
                ),
            )

    def get_question(self, question_id: str) -> Question | None:
        """Return the question with this id, or None."""
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM questions WHERE id = ?;", (question_id,)).fetchone()
            if not row:
                return None
            return Question(
                id=row["id"],
                text=row["text"],
                active=bool(row["active"]),
                linked_interests=json.loads(row["linked_interests"] or "[]"),
                watch=row["watch"] if "watch" in row.keys() else None,
                profile_version=row["profile_version"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(UTC),
            )

    def list_questions(self, active_only: bool = True) -> list[Question]:
        """Return questions in the order they were first added.

        Args:
            active_only: Leave out inactive questions.

        Returns:
            The questions.
        """
        with self.connect() as conn:
            q = "SELECT * FROM questions WHERE active = 1 ORDER BY rowid ASC;" if active_only else "SELECT * FROM questions ORDER BY rowid ASC;"
            rows = conn.execute(q).fetchall()
            return [
                Question(
                    id=r["id"],
                    text=r["text"],
                    active=bool(r["active"]),
                    linked_interests=json.loads(r["linked_interests"] or "[]"),
                    watch=r["watch"] if "watch" in r.keys() else None,
                    profile_version=r["profile_version"],
                    created_at=datetime.fromisoformat(r["created_at"]) if r["created_at"] else datetime.now(UTC),
                )
                for r in rows
            ]

    def upsert_claim(self, claim: Claim) -> None:
        """Insert or update a claim by id.

        An update keeps its question, item and ``created_at``.

        Args:
            claim: The claim to store.
        """
        with self.connect() as conn:
            conn.execute(
                """
            INSERT INTO claims (id, question_id, item_id, text, stance, quote, origin_key, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                text = excluded.text,
                stance = excluded.stance,
                quote = excluded.quote,
                origin_key = excluded.origin_key,
                model = excluded.model;
            """,
                (
                    claim.id,
                    claim.question_id,
                    claim.item_id,
                    claim.text,
                    claim.stance,
                    claim.quote,
                    claim.origin_key,
                    claim.model,
                    claim.created_at.isoformat(),
                ),
            )

    def list_claims_for_question(self, question_id: str) -> list[Claim]:
        """Return the question's claims, oldest first."""
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM claims WHERE question_id = ? ORDER BY created_at ASC;", (question_id,)).fetchall()
            return [
                Claim(
                    id=r["id"],
                    question_id=r["question_id"],
                    item_id=r["item_id"],
                    text=r["text"],
                    stance=r["stance"],
                    quote=r["quote"],
                    origin_key=r["origin_key"],
                    model=r["model"],
                    created_at=datetime.fromisoformat(r["created_at"]) if r["created_at"] else datetime.now(UTC),
                )
                for r in rows
            ]

    def delete_claim(self, claim_id: str) -> None:
        """Delete the claim with this id; an unknown id is ignored."""
        with self.connect() as conn:
            conn.execute("DELETE FROM claims WHERE id = ?;", (claim_id,))

    def get_claim(self, claim_id: str) -> Claim | None:
        """Return the claim with this id, or None."""
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM claims WHERE id = ?;", (claim_id,)).fetchone()
            if not row:
                return None
            return Claim(
                id=row["id"],
                question_id=row["question_id"],
                item_id=row["item_id"],
                text=row["text"],
                stance=row["stance"],
                quote=row["quote"],
                origin_key=row["origin_key"],
                model=row["model"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(UTC),
            )

    def upsert_answer(self, ans: Answer) -> None:
        """Insert or update an answer version.

        An existing row for this question and version is updated.

        The existing version keeps its ``created_at``.

        Args:
            ans: The answer to store.
        """
        with self.connect() as conn:
            conn.execute(
                """
            INSERT INTO answers (question_id, version, sentences, confidence, counterpoint_claim_id, what_changed, what_changed_by, synthesized_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(question_id, version) DO UPDATE SET
                sentences = excluded.sentences,
                confidence = excluded.confidence,
                counterpoint_claim_id = excluded.counterpoint_claim_id,
                what_changed = excluded.what_changed,
                what_changed_by = excluded.what_changed_by,
                synthesized_by = excluded.synthesized_by;
            """,
                (
                    ans.question_id,
                    ans.version,
                    json.dumps([s.model_dump() for s in ans.sentences]),
                    ans.confidence,
                    ans.counterpoint_claim_id,
                    ans.what_changed,
                    ans.what_changed_by,
                    ans.synthesized_by,
                    ans.created_at.isoformat(),
                ),
            )

    def get_latest_answer(self, question_id: str) -> Answer | None:
        """Return the question's highest answer version, or None."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM answers WHERE question_id = ? ORDER BY version DESC LIMIT 1;",
                (question_id,),
            ).fetchone()
            if not row:
                return None
            sentences_raw = json.loads(row["sentences"] or "[]")
            sentences = [Sentence(**s) for s in sentences_raw]
            keys = row.keys()
            return Answer(
                question_id=row["question_id"],
                version=row["version"],
                sentences=sentences,
                confidence=row["confidence"],
                counterpoint_claim_id=row["counterpoint_claim_id"],
                what_changed=row["what_changed"],
                what_changed_by=row["what_changed_by"],
                synthesized_by=row["synthesized_by"] if "synthesized_by" in keys else None,
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(UTC),
            )

    def list_answers(self, question_id: str) -> list[Answer]:
        """Return every version of the question's answer, oldest first."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM answers WHERE question_id = ? ORDER BY version ASC;",
                (question_id,),
            ).fetchall()
            out = []
            for r in rows:
                sentences_raw = json.loads(r["sentences"] or "[]")
                sentences = [Sentence(**s) for s in sentences_raw]
                keys = r.keys()
                out.append(
                    Answer(
                        question_id=r["question_id"],
                        version=r["version"],
                        sentences=sentences,
                        confidence=r["confidence"],
                        counterpoint_claim_id=r["counterpoint_claim_id"],
                        what_changed=r["what_changed"],
                        what_changed_by=r["what_changed_by"],
                        synthesized_by=r["synthesized_by"] if "synthesized_by" in keys else None,
                        created_at=datetime.fromisoformat(r["created_at"]) if r["created_at"] else datetime.now(UTC),
                    )
                )
            return out

    def impact_counts(self) -> dict[str, Any]:
        """Totals since the first fetch, for the dashboard's impact card.

        They are measured, never estimated.

        Items from the OSV and demo sources are left out; quotes count only on active questions other
        than 'q-stack', and never from demo items.

        Returns:
            'read', 'sources', 'since' (date of the first fetch, or None), 'kept', 'filtered',
            'quotes', 'checked' (quotes extracted by a model rather than heuristics) and
            'checked_by' (those models' names, sorted).
        """
        with self.connect() as conn:
            items = conn.execute(
                """
            SELECT COUNT(*) AS read, COUNT(DISTINCT source_id) AS sources, MIN(fetched_at) AS since,
                   COALESCE(SUM(status IN ('scored', 'exploring')), 0) AS kept,
                   COALESCE(SUM(status IN ('dropped_stage_a', 'dropped_stage_b')), 0) AS filtered
            FROM items WHERE COALESCE(source_id, '') NOT IN (?, ?, ?);
            """,
                ("osv", *DEMO_SOURCE_IDS),
            ).fetchone()
            # quotes on the questions the Brief shows, not on hidden ones or demo items
            claims = conn.execute(
                """
            SELECT claims.model, COUNT(*) AS n FROM claims
            JOIN questions ON questions.id = claims.question_id
            JOIN items ON items.id = claims.item_id
            WHERE questions.active = 1 AND claims.question_id != 'q-stack'
              AND COALESCE(items.source_id, '') NOT IN (?, ?)
            GROUP BY claims.model;
            """,
                DEMO_SOURCE_IDS,
            ).fetchall()
        checked = {r["model"]: r["n"] for r in claims if r["model"] not in ("heuristic", "osv", "default")}
        return {
            "read": items["read"],
            "sources": items["sources"],
            "since": items["since"][:10] if items["since"] else None,
            "kept": items["kept"],
            "filtered": items["filtered"],
            "quotes": sum(r["n"] for r in claims),
            "checked": sum(checked.values()),
            "checked_by": sorted(checked),
        }

    def scored_item_ids(self) -> list[str]:
        """Ids of every kept item, newest first.

        A newly added question is matched against these.
        """
        with self.connect() as conn:
            rows = conn.execute("SELECT id FROM items WHERE status = 'scored' ORDER BY published_at DESC;").fetchall()
            return [r["id"] for r in rows]

    def hn_check_candidates(self, checked_before: datetime) -> list[Item]:
        """Kept items with a URL to check on Hacker News.

        Items from HN itself, and items checked on HN since `checked_before`, are left out.
        """
        with self.connect() as conn:
            rows = conn.execute(
                """
            SELECT items.* FROM items
            JOIN sources ON sources.id = items.source_id
            LEFT JOIN hn_sightings h ON h.item_id = items.id
            WHERE items.status = 'scored' AND items.url IS NOT NULL AND sources.type != 'hn'
              AND (h.checked_at IS NULL OR h.checked_at < ?)
            ORDER BY items.published_at DESC;
            """,
                (checked_before.isoformat(),),
            ).fetchall()
            return [self._row_to_item(r) for r in rows]

    def upsert_hn_sighting(
        self, item_id: str, hn_id: str | None, posted_at: datetime | None, points: int | None, checked_at: datetime
    ) -> None:
        """Record an item's Hacker News check, replacing any earlier one.

        Args:
            item_id: Id of the checked item.
            hn_id: The HN story id, or None when the item wasn't found.
            posted_at: When it was posted on HN, if found.
            points: Its HN points, if found.
            checked_at: When the check ran.
        """
        with self.connect() as conn:
            conn.execute(
                """
            INSERT INTO hn_sightings (item_id, hn_id, posted_at, points, checked_at) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                hn_id = excluded.hn_id, posted_at = excluded.posted_at,
                points = excluded.points, checked_at = excluded.checked_at;
            """,
                (item_id, hn_id, posted_at.isoformat() if posted_at else None, points, checked_at.isoformat()),
            )

    def list_hn_sightings(self) -> list[dict[str, Any]]:
        """Items found on HN: item_id, hn_id, posted_at, points."""
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM hn_sightings WHERE hn_id IS NOT NULL;").fetchall()
            return [{"item_id": r["item_id"], "hn_id": r["hn_id"], "points": r["points"] or 0,
                     "posted_at": datetime.fromisoformat(r["posted_at"])} for r in rows]

    def record_track_record(self, tr: TrackRecord) -> None:
        """Insert a track record, or update the one with this id.

        An update changes only the verdict and the HN lead.

        Args:
            tr: The track record to store.
        """
        with self.connect() as conn:
            conn.execute(
                """
            INSERT INTO track_record (id, question_id, answer_version, verdict, hn_lead_days, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                verdict = excluded.verdict,
                hn_lead_days = excluded.hn_lead_days;
            """,
                (
                    tr.id,
                    tr.question_id,
                    tr.answer_version,
                    tr.verdict,
                    tr.hn_lead_days,
                    tr.created_at.isoformat(),
                ),
            )

    def get_track_record_hit_rate(self, question_id: str | None = None) -> dict[str, Any]:
        """Count the user's verdicts on answer changes and the share judged useful.

        Args:
            question_id: Count only this question's records; None (or empty) counts all.

        Returns:
            Counts of 'useful', 'wrong' and 'already_knew', 'total' and 'total_reviewed' (both the
            sum of the three), 'hit_rate' (useful / (useful + wrong), 0.0 when both are zero) and
            'hit_rate_pct' (the rate as a percentage, one decimal).
        """
        with self.connect() as conn:
            if question_id:
                rows = conn.execute(
                    "SELECT verdict, COUNT(*) as cnt FROM track_record WHERE verdict IS NOT NULL AND question_id = ? GROUP BY verdict;",
                    (question_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT verdict, COUNT(*) as cnt FROM track_record WHERE verdict IS NOT NULL GROUP BY verdict;"
                ).fetchall()
            counts = {r["verdict"]: r["cnt"] for r in rows}
            useful = counts.get("useful", 0)
            wrong = counts.get("wrong", 0)
            already_knew = counts.get("already_knew", 0)
            denom = useful + wrong
            hit_rate = (useful / denom) if denom > 0 else 0.0
            return {
                "useful": useful,
                "wrong": wrong,
                "already_knew": already_knew,
                "total": useful + wrong + already_knew,
                "total_reviewed": useful + wrong + already_knew,
                "hit_rate": hit_rate,
                "hit_rate_pct": round(hit_rate * 100.0, 1),
            }
