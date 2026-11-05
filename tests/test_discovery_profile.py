"""Phase 2: Interest profile bootstrap, approval gate, and embedding tests."""

from pathlib import Path

from graphflow import ApprovalDecision, HumanApprovalRequired

from gleaner.db import Database
from gleaner.embed import FakeEmbedder
from gleaner.profile import build_profile_workflow

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "discovery"


def test_profile_bootstrap_pauses_for_approval_and_saves_on_approval(tmp_path):
    db_path = tmp_path / "profile_test.db"
    db = Database(db_path)
    embedder = FakeEmbedder(dim=db.vec_dim)

    app = build_profile_workflow(db, embedder=embedder)
    thread_id = "bootstrap-thread-1"

    interests_yaml = (FIXTURES_DIR / "interests.yaml").read_text(encoding="utf-8")

    # Step 1: Run bootstrap workflow until interrupt
    events = list(app.stream({"raw_yaml": interests_yaml}, thread_id=thread_id))
    interrupts = [e for e in events if isinstance(e, HumanApprovalRequired)]
    assert len(interrupts) == 1
    assert "Review interest profile" in interrupts[0].prompt

    # Database has no interests yet before approval
    with db.connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM interests;").fetchone()[0]
        assert count == 0

    # Step 2: Approve the draft
    decision = ApprovalDecision(approved=True)
    res = app.resume(thread_id=thread_id, decision=decision)
    assert res.get("status") == "approved"

    # Database now contains the approved interests and centroid vectors
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM interests;").fetchall()
        assert len(rows) == 2
        labels = [r["label"] for r in rows]
        assert "Agent Frameworks" in labels
        assert "Test-Time Compute" in labels
    db.close()


def test_profile_bootstrap_saves_nothing_on_rejection(tmp_path):
    db_path = tmp_path / "profile_reject.db"
    db = Database(db_path)
    embedder = FakeEmbedder(dim=db.vec_dim)

    app = build_profile_workflow(db, embedder=embedder)
    thread_id = "reject-thread-1"

    interests_yaml = (FIXTURES_DIR / "interests.yaml").read_text(encoding="utf-8")

    list(app.stream({"raw_yaml": interests_yaml}, thread_id=thread_id))

    # Reject
    decision = ApprovalDecision(approved=False, feedback="Not ready to approve yet")
    res = app.resume(thread_id=thread_id, decision=decision)
    assert res.get("status") == "rejected"

    with db.connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM interests;").fetchone()[0]
        assert count == 0
    db.close()


def _interests_yaml(weight=1.0, kind="topic"):
    return f"""interests:
  - id: "agents"
    label: "Agent Frameworks"
    description: "Agent orchestration frameworks."
    keywords: ["agent"]
    weight: {weight}
    kind: "{kind}"
  - id: "ttc"
    label: "Test-Time Compute"
    description: "Inference-time scaling."
"""


def _bootstrap(app, raw_yaml, thread_id):
    res = app.run({"raw_yaml": raw_yaml}, thread_id=thread_id)
    assert res.status == "interrupted"
    return app.resume(thread_id, ApprovalDecision(approved=True))


def test_profile_draft_is_validated_like_the_config_file(tmp_path):
    import pytest
    from graphflow import ExecutionError

    from gleaner.config import DiscoveryConfigError

    db = Database(tmp_path / "profile_invalid.db")
    app = build_profile_workflow(db, embedder=FakeEmbedder(dim=db.vec_dim))
    with pytest.raises((DiscoveryConfigError, ExecutionError), match="kind"):
        app.run({"raw_yaml": _interests_yaml(kind="hobby")}, thread_id="invalid")
    db.close()


def test_profile_version_bumps_only_when_an_interest_changes(tmp_path):
    db = Database(tmp_path / "profile_version.db")
    app = build_profile_workflow(db, embedder=FakeEmbedder(dim=db.vec_dim))

    def versions():
        with db.connect() as conn:
            return dict(conn.execute("SELECT id, profile_version FROM interests;").fetchall())

    _bootstrap(app, _interests_yaml(), "v1")
    assert versions() == {"agents": 1, "ttc": 1}
    _bootstrap(app, _interests_yaml(), "v1-again")
    assert versions() == {"agents": 1, "ttc": 1}
    _bootstrap(app, _interests_yaml(weight=0.5), "v2")
    assert versions() == {"agents": 2, "ttc": 1}
    db.close()


def test_pending_profile_approval_survives_a_restart(tmp_path):
    import pytest

    pytest.importorskip("langgraph.checkpoint.sqlite", reason="needs graphflow[sqlite] (langgraph-checkpoint-sqlite)")
    db_path = tmp_path / "profile_restart.db"
    db = Database(db_path)
    first = build_profile_workflow(db, embedder=FakeEmbedder(dim=db.vec_dim))
    assert first.run({"raw_yaml": _interests_yaml()}, thread_id="restart").status == "interrupted"

    # A new process builds a new app over the same database and answers the pending approval.
    second = build_profile_workflow(Database(db_path), embedder=FakeEmbedder(dim=db.vec_dim))
    res = second.resume("restart", ApprovalDecision(approved=True))
    assert res.get("status") == "approved"
    db.close()


def test_file_db_keeps_profile_checkpoints_in_sqlite_beside_it(tmp_path, monkeypatch):
    from graphflow import InMemoryCheckpointStore

    import gleaner.profile

    opened = []

    def fake_sqlite_store(db_path):
        opened.append(db_path)
        return InMemoryCheckpointStore()

    monkeypatch.setattr(gleaner.profile, "SqliteCheckpointStore", fake_sqlite_store)
    build_profile_workflow(Database(tmp_path / "beside.db"))
    assert opened == [str(tmp_path / "beside.checkpoints.db")]

    build_profile_workflow(Database(":memory:"))  # in-memory DBs keep in-memory checkpoints
    assert len(opened) == 1


def test_profile_checkpoints_fall_back_to_memory_without_the_sqlite_extra(tmp_path, monkeypatch, caplog):
    import gleaner.profile

    def missing_extra(db_path):
        raise ImportError("SqliteCheckpointStore requires `pip install graphflow[sqlite]`")

    monkeypatch.setattr(gleaner.profile, "SqliteCheckpointStore", missing_extra)
    db = Database(tmp_path / "fallback.db")
    app = build_profile_workflow(db, embedder=FakeEmbedder(dim=db.vec_dim))
    assert _bootstrap(app, _interests_yaml(), "fallback").get("status") == "approved"
    assert "restart" in caplog.text
    db.close()
