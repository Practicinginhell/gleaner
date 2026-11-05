"""Tests for interactive user features:
- In-app '+ Add Source' endpoint (POST /api/sources)
- In-app '+ Track Question' endpoint (POST /api/questions)
- Seed sample opportunities endpoint (POST /api/opportunities/seed-sample)
- Dashboard UI controls for adding sources, tracking questions, and score inspector
"""

from fastapi.testclient import TestClient

from gleaner.db import Database
from gleaner.server import create_app


def test_create_source_endpoint(tmp_path):
    """POST /api/sources validates inputs and adds a source to the database."""
    db = Database(tmp_path / "sources_test.db")
    client = TestClient(create_app(db))

    # 1. Valid RSS source
    res_rss = client.post(
        "/api/sources",
        json={
            "id": "my-ai-blog",
            "type": "rss",
            "params": {"url": "https://example.com/rss.xml"},
            "cadence_s": 7200,
            "trust": 0.85,
        },
    )
    assert res_rss.status_code == 201
    data = res_rss.json()
    assert data["id"] == "my-ai-blog"
    assert data["type"] == "rss"
    assert data["params"]["url"] == "https://example.com/rss.xml"
    assert db.get_source("my-ai-blog") is not None

    # 2. Valid GitHub source
    res_gh = client.post(
        "/api/sources",
        json={
            "id": "gh-fastapi",
            "type": "github",
            "params": {"repo": "fastapi/fastapi"},
            "cadence_s": 3600,
        },
    )
    assert res_gh.status_code == 201
    assert db.get_source("gh-fastapi") is not None

    # 3. Duplicate ID returns 409 Conflict
    res_dup = client.post(
        "/api/sources",
        json={
            "id": "my-ai-blog",
            "type": "rss",
            "params": {"url": "https://example.com/other.xml"},
        },
    )
    assert res_dup.status_code == 409

    # 4. SSRF unsafe URL returns 400 Bad Request
    res_ssrf = client.post(
        "/api/sources",
        json={
            "id": "bad-meta",
            "type": "rss",
            "params": {"url": "http://169.254.169.254/latest/meta-data/"},
        },
    )
    assert res_ssrf.status_code == 400

    # 5. Invalid source type returns 400
    res_bad_type = client.post(
        "/api/sources",
        json={
            "id": "bad-type",
            "type": "unsupported_type",
            "params": {},
        },
    )
    assert res_bad_type.status_code == 400

    db.close()


def test_create_question_endpoint(tmp_path):
    """POST /api/questions creates standing questions in the database."""
    db = Database(tmp_path / "questions_test.db")
    client = TestClient(create_app(db))

    res = client.post(
        "/api/questions",
        json={
            "text": "What are the common failure modes of LangGraph in production?",
            "linked_interests": ["langgraph", "multi-agent"],
        },
    )
    assert res.status_code == 201
    data = res.json()
    assert "id" in data
    assert data["text"] == "What are the common failure modes of LangGraph in production?"
    assert "langgraph" in data["linked_interests"]

    # Verify stored in DB
    questions = db.list_questions()
    assert any(q.text == "What are the common failure modes of LangGraph in production?" for q in questions)

    # Short invalid question rejected
    res_short = client.post("/api/questions", json={"text": "Hi"})
    assert res_short.status_code == 422 or res_short.status_code == 400

    db.close()


def test_seed_sample_opportunities_endpoint(tmp_path):
    """POST /api/opportunities/seed-sample creates sample research deadlines."""
    db = Database(tmp_path / "opps_test.db")
    client = TestClient(create_app(db))

    res = client.post("/api/opportunities/seed-sample")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "seeded"
    assert data["count"] >= 1

    # Check that GET /api/opportunities now returns the seeded items
    res_list = client.get("/api/opportunities")
    assert res_list.status_code == 200
    opps = res_list.json()
    assert len(opps) >= 1
    assert any(o.get("deadline") for o in opps)

    db.close()


def test_demo_data_is_marked_as_samples_and_its_sources_are_never_fetched(tmp_path):
    db = Database(tmp_path / "demo.db")
    client = TestClient(create_app(db))
    client.post("/api/opportunities/seed-sample")
    client.post("/api/security/simulate-poison")

    for src in db.list_sources():
        assert src.enabled is False, src.id  # the scheduler would fetch these placeholder URLs
    opps = client.get("/api/opportunities").json()
    assert opps and all(db.get_item(o["item_id"]).title.startswith("Sample:") for o in opps)
    assert all(o["link"] is None for o in opps)  # no real-looking programs or domains, and no dead example.com link
    db.close()


def test_other_websites_cannot_trigger_actions(tmp_path):
    db = Database(tmp_path / "csrf.db")
    client = TestClient(create_app(db))

    # a page on another site can POST a form here; the browser marks it with that site's Origin
    res = client.post("/api/opportunities/seed-sample", headers={"Origin": "https://evil.example"})
    assert res.status_code == 403
    assert db.list_sources() == []
    # the dashboard itself (same origin) and clients that send no Origin (CLI, curl) still work
    assert client.post("/api/opportunities/seed-sample", headers={"Origin": "http://testserver"}).status_code == 200
    assert client.post("/api/opportunities/seed-sample").status_code == 200
    db.close()


def test_a_question_already_tracked_is_not_added_twice(tmp_path):
    db = Database(tmp_path / "dup_q.db")
    client = TestClient(create_app(db))
    text = "What are the trade-offs between Monte Carlo Tree Search and beam search in test-time scaling?"

    assert client.post("/api/questions", json={"text": text}).status_code == 201
    res = client.post("/api/questions", json={"text": "  what are the trade-offs between monte carlo tree search and beam search in test-time scaling "})
    assert res.status_code == 409
    assert len(db.list_questions()) == 1
    db.close()


def test_a_new_question_never_takes_over_a_hidden_questions_history(tmp_path):
    from gleaner.models import Claim

    db = Database(tmp_path / "hidden_q.db")
    client = TestClient(create_app(db))
    old = client.post("/api/questions", json={"text": "What are the trade-offs between MCTS and beam search?"}).json()["id"]
    db.upsert_claim(Claim(id="c1", question_id=old, item_id="i1", text="t", quote="Beam search is cheaper per step.", stance="context", origin_key="o"))
    with db.connect() as conn:
        conn.execute("UPDATE questions SET active = 0 WHERE id = ?;", (old,))

    # same 32-character slug, different question
    new = client.post("/api/questions", json={"text": "What are the trade-offs between LoRA and full fine-tuning?"}).json()["id"]

    assert new != old and db.list_claims_for_question(new) == []
    db.close()


def test_impact_counts_leave_out_demo_data_and_hidden_questions(tmp_path):
    from gleaner.models import Claim

    db = Database(tmp_path / "impact_demo.db")
    client = TestClient(create_app(db))
    client.post("/api/opportunities/seed-sample")
    client.post("/api/security/simulate-poison")
    hidden = client.post("/api/questions", json={"text": "Is this question hidden from the brief?"}).json()["id"]
    db.upsert_claim(Claim(id="c1", question_id=hidden, item_id="opp-nsf-agents", text="t", quote="A quote on a hidden question.", stance="context", origin_key="o"))
    with db.connect() as conn:
        conn.execute("UPDATE questions SET active = 0 WHERE id = ?;", (hidden,))

    impact = db.impact_counts()
    assert (impact["read"], impact["quotes"]) == (0, 0)  # nothing here was read from the user's sources
    db.close()


def test_a_new_question_gets_evidence_from_recent_items_right_away(tmp_path, monkeypatch):
    # It used to wait for the next collect, showing "No evidence yet" over matches already kept
    from graphflow import MockModel

    import gleaner.llama
    from gleaner.models import Item, Question, SourceRow

    db = Database(tmp_path / "backfill.db")
    db.upsert_source(SourceRow(id="s", type="rss", params={}, cadence_s=3600))
    db.upsert_question(Question(id="q-old", text="What are the common failure modes of LangGraph in production?"))
    db.upsert_item(Item(id="kept", source_id="s", title="Postmortem", url="https://example.com/a", status="scored", text=(
        "LangGraph checkpointer writes were lost for long-running agents after a resume in production.")))
    db.upsert_item(Item(id="dropped", source_id="s", title="Other", url="https://example.com/b", status="dropped", text=(
        "LangGraph checkpointer state grew without bound for long-running agents.")))
    judge = MockModel(responses=lambda messages: "yes")
    monkeypatch.setattr(gleaner.llama, "get_llama_model", lambda role, **kw: judge if role == "judge" else None)
    client = TestClient(create_app(db))

    res = client.post("/api/questions", json={"text": "Is LangGraph's checkpointer stable enough for long-running agents?"})

    assert res.status_code == 201
    assert res.json()["evidence"]["claims_added"] == 1
    claims = db.list_claims_for_question(res.json()["id"])
    assert [(c.item_id, c.model) for c in claims] == [("kept", "mock")]  # Llama confirmed it; dropped items never count
    assert db.list_claims_for_question("q-old") == []  # only the new question is searched
    assert all("Is LangGraph's checkpointer" in m[-1]["content"] for m in judge.history)
    assert db.get_latest_answer(res.json()["id"]) is not None
    db.close()


def _deadline_item(db, item_id="grant-1", title="Research grant for multi-agent systems", days=12):
    from datetime import UTC, datetime, timedelta

    from gleaner.models import Item, SourceRow
    from gleaner.opportunities import extract_opportunity

    db.upsert_source(SourceRow(id="grants", type="rss", params={}, cadence_s=3600))
    item = Item(id=item_id, source_id="grants", kind="event", title=title, text="Apply by the deadline.", url="https://example.com/grant")
    db.upsert_item(item)
    return extract_opportunity(item, db, deadline=datetime.now(UTC) + timedelta(days=days), opp_type="grant", link=item.url)


def test_ask_without_a_model_says_so_and_lists_titles_not_codes(tmp_path, monkeypatch):
    # A walkthrough got "[c9d36335] Scaling-up BERT…" with no word that Llama wasn't running
    import gleaner.llama
    from gleaner.models import Item, SourceRow

    db = Database(tmp_path / "ask_off.db")
    db.upsert_source(SourceRow(id="blog", type="rss", params={}, cadence_s=3600))
    db.upsert_item(Item(id="c9d36335aaaa", source_id="blog", title="LangGraph 1.2 adds durable timers", text="LangGraph release notes.",
                        url="https://example.com/lg", status="scored"))

    def no_model(role, **kw):
        raise ConnectionError("Ollama is not running")

    monkeypatch.setattr(gleaner.llama, "get_llama_model", no_model)
    body = TestClient(create_app(db)).post("/api/ask", json={"query": "What changed in LangGraph?"}).json()

    assert body["answer"].startswith("Llama isn't running, so this is a plain search")
    assert "[c9d36335]" not in body["answer"] and "intelligence feed" not in body["answer"]
    assert body["sources"] == [{"id": "c9d36335aaaa", "title": "LangGraph 1.2 adds durable timers", "url": "https://example.com/lg", "source": "blog"}]
    db.close()


def test_asking_about_deadlines_lists_the_deadlines(tmp_path, monkeypatch):
    # "Which deadlines are coming up?" is one of the page's own suggestions; it returned five blog posts
    import gleaner.llama

    db = Database(tmp_path / "ask_deadlines.db")
    _deadline_item(db)
    monkeypatch.setattr(gleaner.llama, "get_llama_model", lambda role, **kw: (_ for _ in ()).throw(ConnectionError("off")))

    body = TestClient(create_app(db)).post("/api/ask", json={"query": "Which deadlines are coming up?"}).json()

    assert "Research grant for multi-agent systems" in body["answer"]
    assert "days left" in body["answer"]
    db.close()


def test_the_ask_agent_can_look_up_deadlines(tmp_path):
    from graphflow import MockModel, ModelResponse, ToolCall

    from gleaner.ask import build_ask_agent

    db = Database(tmp_path / "ask_tool.db")
    _deadline_item(db)
    model = MockModel(responses=[ModelResponse(tool_calls=[ToolCall(id="c1", name="list_deadlines", args={})]), ModelResponse(content="ok")])

    build_ask_agent(db, model=model).run({"input": "deadlines?"}, thread_id="t-deadlines")

    seen = [m["content"] for m in model.history[1] if isinstance(m, dict) and m.get("role") == "tool"][-1]
    assert "Research grant for multi-agent systems" in seen and "grant" in seen
    # deadline titles come from fetched items: the model sees them fenced as untrusted data
    assert seen.startswith("--- BEGIN RETRIEVED UNTRUSTED CONTENT ---") and seen.rstrip().endswith("--- END RETRIEVED UNTRUSTED CONTENT ---")
    db.close()


def test_a_question_can_be_removed_from_the_brief(tmp_path):
    db = Database(tmp_path / "remove_q.db")
    client = TestClient(create_app(db))
    qid = client.post("/api/questions", json={"text": "can small AI models run on my phone without internet?"}).json()["id"]

    assert client.delete(f"/api/questions/{qid}").status_code == 200
    assert db.list_questions() == []  # hidden, its history kept
    assert db.get_question(qid) is not None
    assert client.delete("/api/questions/no-such-question").status_code == 404
    assert "can small AI models run on my phone" not in client.get("/").text
    db.close()


def test_status_check_counts_like_the_page_and_names_the_real_models(tmp_path):
    # System status said "Degraded" over a paused demo source, counted demo items, and "Not configured"
    from gleaner.models import SourceRow

    db = Database(tmp_path / "status.db")
    db.upsert_source(SourceRow(id="paused-demo", type="rss", params={}, cadence_s=3600, enabled=False, failures=3))
    db.upsert_source(SourceRow(id="ok", type="rss", params={}, cadence_s=3600))

    d = TestClient(create_app(db)).get("/api/env/check").json()

    assert d["status"] == "healthy" and d["failing_sources"] == 0 and d["active_sources"] == 1
    assert d["models"]["judge"] == "llama3.2:3b" and d["models"]["answer"] == "muse-spark-1.3"
    db.close()


def test_write_answer_says_why_no_answer_was_written(tmp_path, monkeypatch):
    from gleaner.models import Question

    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    db = Database(tmp_path / "why.db")
    db.upsert_question(Question(id="q", text="Is LangGraph's checkpointer stable for long-running agents?"))

    body = TestClient(create_app(db)).post("/api/questions/q/synthesize?force=true").json()

    assert body["synthesized_by"] is None
    assert "MODEL_API_KEY isn't set" in body["reason"]
    db.close()


def test_ask_never_shows_items_from_paused_websites(tmp_path):
    from gleaner.ask import hybrid_search
    from gleaner.models import Item, SourceRow

    db = Database(tmp_path / "paused.db")
    db.upsert_source(SourceRow(id="live", type="rss", params={}, cadence_s=3600))
    db.upsert_source(SourceRow(id="tech-blog-sample", type="rss", params={}, cadence_s=3600, enabled=False))
    db.upsert_item(Item(id="live-1", source_id="live", title="Graphflow workflows in practice", text="Graphflow declarative workflows."))
    db.upsert_item(Item(id="demo-1", source_id="tech-blog-sample", title="Building Declarative Workflows with Graphflow", text="Graphflow declarative workflows."))

    assert [it.id for it in hybrid_search(db, "Graphflow workflows")] == ["live-1"]
    db.close()


def test_sample_deadlines_link_nowhere_fake(tmp_path):
    db = Database(tmp_path / "samples.db")
    client = TestClient(create_app(db))
    client.post("/api/opportunities/seed-sample")
    with db.connect() as conn:
        links = [r[0] for r in conn.execute("SELECT link FROM opportunities;")]
    assert links and all(link is None for link in links)
    db.close()


class _Named:
    def __init__(self, name):
        self.model_name = name


def _fake_ask(monkeypatch, failing=()):
    """Stub the model factory and the agent; return the model names tried, in order."""
    import gleaner.ask
    import gleaner.llama

    tried = []
    monkeypatch.setattr(gleaner.llama, "get_llama_model", lambda role, model_name=None, **kw: _Named(model_name or "llama3.2:3b"))

    class Agent:
        def __init__(self, model):
            self.model = model

        def run(self, state, thread_id):
            tried.append(self.model.model_name)
            if self.model.model_name in failing:
                raise ConnectionError("down")
            return {"output": f"answer from {self.model.model_name}"}

    monkeypatch.setattr(gleaner.ask, "build_ask_agent", lambda db, model: Agent(model))
    return tried


def test_ask_uses_muse_spark_when_the_key_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "k")
    monkeypatch.delenv("LLAMA_ASK_MODEL", raising=False)
    tried = _fake_ask(monkeypatch)
    body = TestClient(create_app(Database(tmp_path / "a.db"))).post("/api/ask", json={"query": "What changed?"}).json()
    assert tried == ["muse-spark-1.3"]
    assert body["answer"] == "answer from muse-spark-1.3" and body["answered_by"] == "muse-spark-1.3"


def test_ask_falls_back_to_llama_when_muse_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "k")
    monkeypatch.delenv("LLAMA_ASK_MODEL", raising=False)
    tried = _fake_ask(monkeypatch, failing={"muse-spark-1.3"})
    body = TestClient(create_app(Database(tmp_path / "a.db"))).post("/api/ask", json={"query": "What changed?"}).json()
    assert tried == ["muse-spark-1.3", "llama3.2:3b"]
    assert body["answered_by"] == "llama3.2:3b"


def test_ask_without_the_key_or_with_an_override_uses_one_model(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    tried = _fake_ask(monkeypatch)
    client = TestClient(create_app(Database(tmp_path / "a.db")))
    client.post("/api/ask", json={"query": "What changed?"})
    monkeypatch.setenv("MODEL_API_KEY", "k")
    monkeypatch.setenv("LLAMA_ASK_MODEL", "llama3.2:3b")
    client.post("/api/ask", json={"query": "What changed?"})
    assert tried == ["llama3.2:3b", "llama3.2:3b"]


def test_status_names_the_model_ask_really_uses(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "k")
    monkeypatch.delenv("LLAMA_ASK_MODEL", raising=False)
    monkeypatch.delenv("GLEANER_MODEL", raising=False)
    body = TestClient(create_app(Database(tmp_path / "a.db"))).get("/api/env/check").json()
    assert body["models"]["ask"] == "muse-spark-1.3"


def test_ask_fallback_names_the_models_that_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "k")
    monkeypatch.delenv("LLAMA_ASK_MODEL", raising=False)
    _fake_ask(monkeypatch, failing={"muse-spark-1.3", "llama3.2:3b"})
    body = TestClient(create_app(Database(tmp_path / "a.db"))).post("/api/ask", json={"query": "What changed?"}).json()
    assert body["answer"].startswith("Muse Spark and Llama didn't answer")


def test_write_answer_with_a_non_meta_model_says_so(tmp_path, monkeypatch):
    from gleaner.models import Question

    monkeypatch.setenv("LLAMA_ANSWER_MODEL", "gpt-4o")
    db = Database(tmp_path / "s.db")
    db.upsert_question(Question(id="q1", text="What changed?", keywords=["changed"]))
    res = TestClient(create_app(db)).post("/api/questions/q1/synthesize")
    assert res.status_code == 400
    assert "not a Meta model" in res.json()["detail"]
