"""Phase 4: Feedback and Rocchio learning loop tests."""

import json

from gleaner.db import Database
from gleaner.embed import FakeEmbedder, cosine_similarity
from gleaner.learn import (
    record_feedback,
    rescore_scored_items,
    source_feedback_counts,
    update_interests_from_feedback,
)
from gleaner.models import Item, SourceRow
from gleaner.scoring import compute_source_quality


def _add_interest(db, embedder, int_id, label, description, keywords):
    """Insert an approved interest with its seed centroid, as the profile workflow does."""
    with db.connect() as conn:
        conn.execute(
            """
        INSERT INTO interests (id, label, description, keywords, exclusions, weight, kind, active, profile_version)
        VALUES (?, ?, ?, ?, '[]', 1.0, 'topic', 1, 1);
        """,
            (int_id, label, description, json.dumps(keywords)),
        )
    seed = embedder.embed([f"search_query: {label} {description} {' '.join(keywords)}"])[0]
    db.upsert_interest_vector(int_id, seed)
    return seed


def _interest_state(db, int_id):
    with db.connect() as conn:
        row = conn.execute("SELECT weight, profile_version FROM interests WHERE id = ?;", (int_id,)).fetchone()
    return float(row["weight"]), int(row["profile_version"]), db.get_interest_vector(int_id)


def _mean(vectors):
    return [sum(col) / len(vectors) for col in zip(*vectors, strict=True)]


X_ITEMS = [
    ("x-1", "LangGraph agent orchestration patterns"),
    ("x-2", "Graphflow multi-agent orchestration release"),
    ("x-3", "Agent frameworks compared LangGraph graphflow"),
    ("x-4", "Building multi-agent systems with LangGraph"),
    ("x-5", "Orchestration of agent workflows in production"),
]


def _seed_two_interests_and_x_likes(db, embedder):
    seed_x = _add_interest(
        db, embedder, "agent-frameworks", "Agent Frameworks", "Orchestration frameworks for agents",
        ["langgraph", "graphflow", "multi-agent", "orchestration"],
    )
    seed_y = _add_interest(
        db, embedder, "sourdough", "Sourdough Baking", "Bread baking with wild yeast",
        ["sourdough", "starter", "bread", "flour"],
    )
    db.upsert_source(SourceRow(id="src-1", type="rss", params={"url": "http://feed.com"}, cadence_s=3600))
    liked_vecs = []
    for item_id, title in X_ITEMS:
        db.upsert_item(Item(id=item_id, source_id="src-1", title=title, text=title, status="scored"))
        vec = embedder.embed([f"search_document: {title} {title}"])[0]
        db.upsert_item_vector(item_id, vec)
        liked_vecs.append(vec)
        record_feedback(db, item_id=item_id, action="like")
    return seed_x, seed_y, liked_vecs


def test_five_likes_on_x_move_x_centroid_toward_x_and_leave_y_unchanged():
    db = Database(":memory:")
    embedder = FakeEmbedder(dim=db.vec_dim)
    seed_x, seed_y, liked_vecs = _seed_two_interests_and_x_likes(db, embedder)
    liked_mean = _mean(liked_vecs)

    update_interests_from_feedback(db, embedder=embedder)

    weight_x, version_x, centroid_x = _interest_state(db, "agent-frameworks")
    assert weight_x > 1.0
    assert version_x == 2
    assert cosine_similarity(centroid_x, liked_mean) > cosine_similarity(seed_x, liked_mean) + 1e-3
    assert cosine_similarity(centroid_x, seed_x) >= 0.8

    weight_y, version_y, centroid_y = _interest_state(db, "sourdough")
    assert weight_y == 1.0
    assert version_y == 1
    assert cosine_similarity(centroid_y, seed_y) > 1 - 1e-6
    db.close()


def test_learning_update_twice_gives_identical_weights_and_version():
    db = Database(":memory:")
    embedder = FakeEmbedder(dim=db.vec_dim)
    _seed_two_interests_and_x_likes(db, embedder)

    update_interests_from_feedback(db, embedder=embedder)
    first = {i: _interest_state(db, i) for i in ("agent-frameworks", "sourdough")}
    update_interests_from_feedback(db, embedder=embedder)
    second = {i: _interest_state(db, i) for i in ("agent-frameworks", "sourdough")}

    for int_id in first:
        assert first[int_id][0] == second[int_id][0], int_id
        assert first[int_id][1] == second[int_id][1], int_id
        assert cosine_similarity(first[int_id][2], second[int_id][2]) > 1 - 1e-6, int_id
    db.close()


def test_learning_drift_is_bounded_even_with_a_large_step():
    db = Database(":memory:")
    embedder = FakeEmbedder(dim=db.vec_dim)
    seed_x, _, _ = _seed_two_interests_and_x_likes(db, embedder)

    update_interests_from_feedback(db, embedder=embedder, alpha=50.0)

    _, _, centroid_x = _interest_state(db, "agent-frameworks")
    assert cosine_similarity(centroid_x, seed_x) >= 0.8 - 1e-6
    db.close()


def test_learning_rescores_stored_items_after_feedback():
    db = Database(":memory:")
    embedder = FakeEmbedder(dim=db.vec_dim)
    _seed_two_interests_and_x_likes(db, embedder)
    # An unrated item about X, scored before any feedback was learned
    title = "LangGraph orchestration for multi-agent apps"
    db.upsert_item(Item(id="x-new", source_id="src-1", title=title, text=title, status="scored"))
    db.upsert_item_vector("x-new", embedder.embed([f"search_document: {title} {title}"])[0])
    comps = {"relevance": 0.1, "novelty": 0.5, "quality": 0.5, "momentum": 0.0, "actionability": 0.3, "decay": 1.0}
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO scores (item_id, profile_version, components, final) VALUES ('x-new', 1, ?, 0.1);",
            (json.dumps(comps),),
        )

    def stored(item_id="x-new"):
        with db.connect() as conn:
            return conn.execute(
                "SELECT components, final, profile_version FROM scores WHERE item_id = ?;", (item_id,)
            ).fetchone()

    # Rescoring against the unlearned profile gives the baseline relevance
    assert rescore_scored_items(db, embedder=embedder) == 1
    baseline_relevance = json.loads(stored()["components"])["relevance"]

    update_interests_from_feedback(db, embedder=embedder)

    row = stored()
    new_comps = json.loads(row["components"])
    # X's weight went up and its centroid moved toward the liked items: the unrated X item gains
    assert new_comps["relevance"] > baseline_relevance
    # Five likes on src-1 raise its Beta quality above the 0.5-trust prior
    assert new_comps["quality"] > compute_source_quality(trust=0.5)
    assert row["final"] > 0.1
    assert row["profile_version"] == 2
    db.close()


def test_feedback_updates_source_quality():
    db = Database(":memory:")
    src = SourceRow(id="src-trusted", type="rss", params={"url": "http://feed.com"}, cadence_s=3600, trust=0.6)
    db.upsert_source(src)
    for n in range(6):
        db.upsert_item(Item(id=f"item-{n}", source_id="src-trusted", title="Quality Article", text="Good information."))

    prior = compute_source_quality(0.6, *source_feedback_counts(db, "src-trusted"))

    record_feedback(db, item_id="item-0", action="like")
    record_feedback(db, item_id="item-1", action="like")
    assert source_feedback_counts(db, "src-trusted") == (2, 0)
    updated = compute_source_quality(0.6, *source_feedback_counts(db, "src-trusted"))
    assert updated > prior

    for n in range(2, 6):
        record_feedback(db, item_id=f"item-{n}", action="dislike")
    assert source_feedback_counts(db, "src-trusted") == (2, 4)
    worse = compute_source_quality(0.6, *source_feedback_counts(db, "src-trusted"))
    assert worse < prior
    db.close()


def test_rescoring_without_new_feedback_keeps_the_ingest_relevance():
    from gleaner.workflows import process_and_score_items

    db = Database(":memory:")
    embedder = FakeEmbedder(dim=db.vec_dim)
    _seed_two_interests_and_x_likes(db, embedder)
    with db.connect() as conn:  # a learned weight: ingest and rescore must apply it the same way
        conn.execute("UPDATE interests SET weight = 1.5 WHERE id = 'agent-frameworks';")
        conn.execute("DELETE FROM feedback;")
    text = "LangGraph and graphflow orchestration patterns for multi-agent systems in production, with checkpoints."
    item = Item(id="x-ingest", source_id="src-1", title="Multi-agent orchestration with LangGraph", text=text * 3)
    db.upsert_item(item)
    process_and_score_items(db, [item], embedder=embedder, model=None)

    def relevance():
        with db.connect() as conn:
            row = conn.execute("SELECT components FROM scores WHERE item_id = 'x-ingest';").fetchone()
        return json.loads(row["components"])["relevance"]

    at_ingest = relevance()
    rescore_scored_items(db, embedder=embedder)
    assert relevance() == at_ingest
    db.close()


def test_source_quality_counts_each_items_latest_verdict_once():
    from gleaner.workflows import _source_feedback_counts

    db = Database(":memory:")
    db.upsert_source(SourceRow(id="src-1", type="rss", params={"url": "http://feed.com"}))
    for item_id in ("a", "b"):
        db.upsert_item(Item(id=item_id, source_id="src-1", title=item_id, text=item_id))
    for action in ("like", "like", "like"):  # three clicks on one item
        record_feedback(db, "a", action)
    record_feedback(db, "b", "like")
    record_feedback(db, "b", "dislike")  # changed their mind: the dislike stands

    assert source_feedback_counts(db, "src-1") == (1, 1)
    assert _source_feedback_counts(db, "src-1") == (1, 1)
    db.close()
