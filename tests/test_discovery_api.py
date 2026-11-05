"""Phase 4: FastAPI endpoints and SSE streaming tests."""


from fastapi.testclient import TestClient
from graphflow import MockModel, ModelResponse

import gleaner.llama
from gleaner.db import Database
from gleaner.models import Item, ScoreBreakdown, SourceRow
from gleaner.server import create_app


def _seed_scored_item(db: Database, item: Item, final: float = 0.8) -> None:
    db.upsert_item(item)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO scores (item_id, profile_version, components, final, computed_at) VALUES (?, 1, ?, ?, ?);",
            (item.id, ScoreBreakdown(relevance=0.9).model_dump_json(), final, "2026-10-27T00:00:00+00:00"),
        )


def _offline_model(monkeypatch, content: str) -> None:
    """Make the ask endpoint use a scripted MockModel instead of a live Llama endpoint."""
    monkeypatch.setattr(
        gleaner.llama, "get_llama_model", lambda *a, **k: MockModel(responses=[ModelResponse(content=content)])
    )


def test_api_feed_and_feedback(tmp_path, monkeypatch):
    _offline_model(monkeypatch, "Nothing notable.")
    db = Database(tmp_path / "api.db")
    src = SourceRow(id="src-test", type="rss", params={"url": "http://feed.com"}, cadence_s=3600)
    db.upsert_source(src)
    item = Item(id="item-test-1", source_id="src-test", title="API Test Post", text="Content for test.", status="scored")
    _seed_scored_item(db, item)

    app = create_app(db)
    client = TestClient(app)

    # 1. GET /feed returns the seeded, scored item
    res_feed = client.get("/api/feed")
    assert res_feed.status_code == 200
    cards = res_feed.json()["cards"]
    assert [c["item"]["id"] for c in cards] == ["item-test-1"]
    assert cards[0]["item"]["title"] == "API Test Post"
    assert cards[0]["score"] > 0

    # 2. POST /api/feedback
    res_fb = client.post("/api/feedback", json={"item_id": "item-test-1", "action": "like"})
    assert res_fb.status_code == 200
    assert res_fb.json().get("status") == "recorded"

    # Verify feedback was stored
    with db.connect() as conn:
        fb_rows = conn.execute("SELECT * FROM feedback WHERE item_id = 'item-test-1';").fetchall()
    assert len(fb_rows) == 1
    assert fb_rows[0]["action"] == "like"

    # 3. GET /api/sources/{id} for unknown source -> 404
    res_unknown = client.get("/api/sources/unknown-source-999")
    assert res_unknown.status_code == 404

    # 4. POST /api/ask returns an answer
    res_ask = client.post("/api/ask", json={"query": "What agent frameworks were released?"})
    assert res_ask.status_code == 200
    assert res_ask.json()["answer"] == "Nothing notable."

    # 5. GET / returns HTML dashboard
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "text/html" in res_index.headers.get("content-type", "")
    assert "gleaner" in res_index.text
    assert "API Test Post" in res_index.text
    db.close()


def test_api_ask_success_path_extracts_citations(monkeypatch):
    _offline_model(monkeypatch, "See [item-abc-123] and again [item-abc-123], plus [item-def-456].")
    db = Database(":memory:")
    client = TestClient(create_app(db))

    res = client.post("/api/ask", json={"query": "anything new?"})

    assert res.status_code == 200
    body = res.json()
    assert body["answer"].startswith("See [item-abc-123]")
    assert body["citations"] == ["item-abc-123", "item-def-456"]


def test_api_patch_source_validates_and_persists(tmp_path):
    db = Database(tmp_path / "patch.db")
    db.upsert_source(SourceRow(id="src-p", type="rss", params={"url": "http://feed.com"}, cadence_s=3600, failures=2))
    client = TestClient(create_app(db))

    for bad in ({"trust": 1.5}, {"trust": -0.1}, {"cadence_s": 10}, {"enabled": "maybe"}, {"bogus": 1}):
        res = client.patch("/api/sources/src-p", json=bad)
        assert res.status_code == 422, bad
    assert db.get_source("src-p").trust == 0.5

    res = client.patch("/api/sources/src-p", json={"enabled": False, "trust": 0.9, "cadence_s": 600})
    assert res.status_code == 200
    stored = db.get_source("src-p")
    assert (stored.enabled, stored.trust, stored.cadence_s) == (False, 0.9, 600)

    res = client.patch("/api/sources/src-p", json={"enabled": True})
    assert res.status_code == 200
    stored = db.get_source("src-p")
    assert stored.enabled is True
    assert stored.failures == 0
    assert stored.trust == 0.9

    assert client.patch("/api/sources/missing", json={"enabled": True}).status_code == 404
    db.close()


def test_api_ask_with_unusable_model_config_degrades_to_retrieval(monkeypatch):
    """A bad model config (e.g. a non-Meta model name) must not turn /api/ask into a 500."""

    def _bad_config(*args, **kwargs):
        raise ValueError("Model 'qwen' is not a Meta Llama model")

    monkeypatch.setattr(gleaner.llama, "get_llama_model", _bad_config)
    db = Database(":memory:")
    client = TestClient(create_app(db))

    resp = client.post("/api/ask", json={"query": "graphflow"})

    assert resp.status_code == 200
    assert "a plain search found nothing" in resp.json()["answer"]


def test_api_standing_questions(tmp_path):
    """GET /api/questions and POST /api/questions/{id}/feedback endpoints."""
    from gleaner.models import Answer, Question, Sentence

    db = Database(tmp_path / "questions_api.db")
    q = Question(id="q-agents", text="Is multi-agent coordination production ready?", linked_interests=["agents"])
    db.upsert_question(q)

    ans = Answer(
        question_id="q-agents",
        version=1,
        sentences=[Sentence(text="Single agent graphs are currently more reliable.", claim_ids=["c1"])],
        confidence="medium",
        what_changed="Initial answer synthesized from benchmark evidence.",
        counterpoint_claim_id=None,
    )
    db.upsert_answer(ans)

    client = TestClient(create_app(db))

    # 1. GET /api/questions
    res = client.get("/api/questions")
    assert res.status_code == 200
    data = res.json()
    assert len(data["questions"]) == 1
    q_data = data["questions"][0]
    assert q_data["id"] == "q-agents"
    assert q_data["text"] == "Is multi-agent coordination production ready?"
    assert q_data["latest_answer"]["confidence"] == "medium"
    assert "Single agent graphs" in q_data["latest_answer"]["sentences"][0]["text"]

    # 2. POST /api/questions/{id}/feedback
    fb_res = client.post("/api/questions/q-agents/feedback", json={"rating": "useful"})
    assert fb_res.status_code == 200
    fb_data = fb_res.json()
    assert fb_data["status"] == "recorded"
    assert fb_data["hit_rate"]["useful"] == 1
    assert fb_data["hit_rate"]["hit_rate_pct"] == 100.0


def test_user_findings_validation(tmp_path):
    """Verify fixes for user findings:
    - 404 for unknown item in feedback
    - 400 for invalid action in feedback
    - Idempotent duplicate feedback
    - 400 for invalid mode in /api/feed
    - 400 for invalid date in /api/digest/{date}
    - Meaningful reply for empty ask query
    - Offline header says 'not run yet' instead of claiming Llama 3.2 ran
    """
    db = Database(tmp_path / "findings.db")
    src = SourceRow(id="src-1", type="rss", params={"url": "http://example.com"}, cadence_s=3600)
    db.upsert_source(src)
    item = Item(id="item-real-1", source_id="src-1", title="Valid Item", text="Content here", status="scored")
    db.upsert_item(item)

    client = TestClient(create_app(db))

    # Finding 4: Feedback for non-existent item returns 404, not 500
    res_unknown = client.post("/api/feedback", json={"item_id": "nonexistent-item", "action": "like"})
    assert res_unknown.status_code == 404

    # Finding 5: Feedback with invalid action returns 400
    res_bad_action = client.post("/api/feedback", json={"item_id": "item-real-1", "action": "explode"})
    assert res_bad_action.status_code == 400

    # Finding 6: Duplicate like does not create duplicate rows
    res_like1 = client.post("/api/feedback", json={"item_id": "item-real-1", "action": "like"})
    assert res_like1.status_code == 200
    res_like2 = client.post("/api/feedback", json={"item_id": "item-real-1", "action": "like"})
    assert res_like2.status_code == 200
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM feedback WHERE item_id = 'item-real-1';").fetchall()
    assert len(rows) == 1

    # Finding 8: Invalid mode returns 400
    res_bad_mode = client.get("/api/feed?mode=bogus")
    assert res_bad_mode.status_code == 400

    # Finding 8: Invalid date returns 400
    res_bad_date = client.get("/api/digest/not-a-date")
    assert res_bad_date.status_code == 400

    # Finding 9: Empty ask query returns helpful message, not arbitrary items
    res_empty_ask = client.post("/api/ask", json={"query": "   "})
    assert res_empty_ask.status_code == 200
    assert "Please enter a question" in res_empty_ask.json()["answer"]
    assert res_empty_ask.json()["citations"] == []

    # Finding 3: Header chip says 'not run yet', brand badge is 'Meta Models'
    res_page = client.get("/")
    assert res_page.status_code == 200
    assert "Not checked by a model yet" in res_page.text
    assert "Checked by Llama 3.2 3B" not in res_page.text

    db.close()


