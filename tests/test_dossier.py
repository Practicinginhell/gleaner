"""Tests for research dossier generation and export (Obsidian/PKM compatible Markdown)."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from gleaner.db import Database
from gleaner.dossier import generate_dossier
from gleaner.models import Answer, Claim, Item, Question, Sentence, SourceRow
from gleaner.server import create_app


def test_generate_dossier_empty_db(tmp_path):
    db = Database(tmp_path / "empty.db")
    md = generate_dossier(db)
    assert "# Gleaner Intelligence Dossier" in md
    assert "## Standing Questions & Living Answers" in md
    assert "No standing questions tracked yet." in md
    db.close()


def test_generate_dossier_with_questions_and_stories(tmp_path):
    db = Database(tmp_path / "dossier.db")
    now = datetime.now(UTC)

    # 1. Seed standing question
    q = Question(
        id="q-agents",
        text="Are small open models reliable enough for production tool use?",
        linked_interests=["agents"],
    )
    db.upsert_question(q)

    # 2. Seed source and item
    src = SourceRow(id="arxiv-feed", type="arxiv", params={}, enabled=True)
    db.upsert_source(src)
    item = Item(
        id="item-bench-1",
        source_id="arxiv-feed",
        title="Scaling Laws for Autonomous Tool Calling in 3B Models",
        text="Small 3B models achieve 82% accuracy on single function calls with JSON schema constraints.",
        url="https://arxiv.org/abs/2610.12345",
        status="scored",
        published_at=now,
    )
    db.upsert_item(item)

    # 3. Seed claims
    c1 = Claim(
        id="c-1",
        question_id=q.id,
        item_id=item.id,
        text="Small 3B models achieve 82% accuracy on single function calls.",
        quote="Small 3B models achieve 82% accuracy on single function calls with JSON schema constraints.",
        stance="supports",
        origin_key="arxiv:2610.12345",
        created_at=now,
    )
    c2 = Claim(
        id="c-2",
        question_id=q.id,
        item_id=item.id,
        text="Recursive loops exhibit 65% failure without checkpoint recovery.",
        quote="Recursive loops exhibit 65% failure without checkpoint recovery.",
        stance="contradicts",
        origin_key="github:langchain-ai/langgraph",
        created_at=now,
    )
    db.upsert_claim(c1)
    db.upsert_claim(c2)

    # 4. Seed answer
    ans = Answer(
        question_id=q.id,
        version=2,
        sentences=[
            Sentence(text="Small 3B models handle single function calls with high reliability [1].", claim_ids=[c1.id]),
            Sentence(text="However, multi-step recursion requires external checkpointer recovery [2].", claim_ids=[c2.id]),
        ],
        confidence="medium",
        counterpoint_claim_id=c2.id,
        what_changed="Added empirical benchmark evidence and counterpoint for recursion limits.",
        what_changed_by="llama3.2:3b",
        synthesized_by="muse-spark-1.1",
        created_at=now,
    )
    db.upsert_answer(ans)

    # Generate dossier
    md = generate_dossier(db)
    assert "# Gleaner Intelligence Dossier" in md
    assert "### Are small open models reliable enough for production tool use?" in md
    assert "**Version**: 2 • **Confidence**: MEDIUM • **Synthesized by**: muse-spark-1.1" in md
    assert "**What Changed**: Added empirical benchmark evidence" in md
    assert "Small 3B models handle single function calls" in md
    assert "**Strongest Counterpoint**: Recursive loops exhibit 65% failure" in md
    assert "https://arxiv.org/abs/2610.12345" in md
    assert "Small 3B models achieve 82% accuracy" in md
    db.close()


def test_api_export_dossier_endpoint(tmp_path):
    db = Database(tmp_path / "api_dossier.db")
    app = create_app(db, enable_scheduler=False)
    client = TestClient(app)

    res = client.get("/api/export/dossier")
    assert res.status_code == 200
    assert "text/markdown" in res.headers["content-type"]
    assert 'attachment; filename="gleaner-dossier-' in res.headers["content-disposition"]
    assert "# Gleaner Intelligence Dossier" in res.text
    db.close()


def test_dossier_lists_detected_trends(tmp_path, monkeypatch):
    import gleaner.dossier
    from gleaner.models import Trend

    db = Database(tmp_path / "trends.db")
    monkeypatch.setattr(
        gleaner.dossier, "detect_trends", lambda db: [Trend(id="t1", label="Agent checkpoints", story_ids=["a", "b"], z_score=3.1)]
    )
    md = generate_dossier(db)
    assert "Agent checkpoints" in md and "2 stories" in md


def test_dossier_shows_no_confidence_for_unjudged_quotes_and_escapes_feed_links(tmp_path):
    db = Database(tmp_path / "plain.db")
    q = Question(id="q-1", text="Does LangGraph checkpointing scale?")
    db.upsert_question(q)
    db.upsert_source(SourceRow(id="blog", type="rss", params={}))
    db.upsert_item(Item(id="i-1", source_id="blog", title="Bad ](http://evil.example) title", text="t",
                        url="https://ex.com/a b)c", status="scored"))
    db.upsert_claim(Claim(id="c-1", question_id=q.id, item_id="i-1", text="Checkpoints grew.",
                          quote="The checkpoint store grew without bound.", stance="context", origin_key="blog:ex"))
    db.upsert_answer(Answer(question_id=q.id, version=1, sentences=[], confidence="insufficient"))

    md = generate_dossier(db)

    # quotes only, no model judged them: a confidence level would be invented (the dashboard hides it too)
    assert "Confidence" not in md
    # a feed title or URL can't break out of its Markdown link
    assert "[Bad \\](http://evil.example) title](https://ex.com/a%20b%29c)" in md
    # claims include cryptographic tamper-evident receipts
    assert "`receipt: #" in md
    db.close()
