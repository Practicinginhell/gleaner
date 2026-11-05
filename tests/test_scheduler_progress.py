"""Tests for autonomous cadence scheduler, streaming progress broadcaster, and living answer synthesis endpoints."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from graphflow import MockModel, ModelResponse

import gleaner.llama
from gleaner.db import Database
from gleaner.models import Claim, Item, Question, SourceRow
from gleaner.server import CollectProgressBroadcaster, cadence_scheduler_loop, create_app
from gleaner.workflows import acquire_run_lock, release_run_lock


def test_collect_progress_broadcaster():
    broadcaster = CollectProgressBroadcaster()
    assert broadcaster.active is False
    assert broadcaster.status_dict()["active"] is False

    q = broadcaster.add_subscriber()
    broadcaster.start_run("run-123")
    assert broadcaster.active is True
    assert broadcaster.run_id == "run-123"

    ev1 = q.get_nowait()
    assert ev1["stage"] == "start"
    assert ev1["payload"]["run_id"] == "run-123"

    broadcaster.emit("fetching", "Fetching sources", {"count": 3})
    ev2 = q.get_nowait()
    assert ev2["stage"] == "fetching"
    assert ev2["payload"]["count"] == 3

    broadcaster.finish_run({"items_stored": 5})
    assert broadcaster.active is False
    assert broadcaster.stats == {"items_stored": 5}

    ev3 = q.get_nowait()
    assert ev3["stage"] == "complete"
    assert ev3["payload"]["stats"] == {"items_stored": 5}

    # Termination signal
    ev_term = q.get_nowait()
    assert ev_term is None

    broadcaster.remove_subscriber(q)
    assert q not in broadcaster.subscribers


@pytest.mark.asyncio
async def test_cadence_scheduler_loop_triggers_due_sources(tmp_path, monkeypatch):
    db = Database(tmp_path / "sched.db")
    now = datetime.now(UTC)
    src = SourceRow(
        id="src-sched",
        type="rss",
        params={"url": "https://example.com/rss"},
        cadence_s=3600,
        next_due_at=now - timedelta(minutes=5),
        enabled=True,
    )
    db.upsert_source(src)

    broadcaster = CollectProgressBroadcaster()
    tick_called = asyncio.Event()

    async def fake_tick(database, **kwargs):
        progress_cb = kwargs.get("progress_callback")
        if progress_cb:
            progress_cb("testing", "Progress emitted")
        tick_called.set()
        return {"sources_run": 1, "items_stored": 2, "sources_failed": 0}

    import gleaner.workflows
    monkeypatch.setattr(gleaner.workflows, "run_collect_tick", fake_tick)

    stop_event = asyncio.Event()

    async def run_sched():
        await cadence_scheduler_loop(db, broadcaster, interval_s=0.01, stop_event=stop_event)

    sched_task = asyncio.create_task(run_sched())
    try:
        await asyncio.wait_for(tick_called.wait(), timeout=3.0)
        assert broadcaster.stats == {"sources_run": 1, "items_stored": 2, "sources_failed": 0}
    finally:
        stop_event.set()
        await sched_task
    db.close()


@pytest.mark.asyncio
async def test_cadence_scheduler_loop_respects_run_lock(tmp_path):
    db = Database(tmp_path / "lock_sched.db")
    now = datetime.now(UTC)
    src = SourceRow(
        id="src-due",
        type="rss",
        params={"url": "https://example.com/rss"},
        cadence_s=3600,
        next_due_at=now - timedelta(minutes=5),
        enabled=True,
    )
    db.upsert_source(src)

    broadcaster = CollectProgressBroadcaster()
    # Externally acquire lock
    assert acquire_run_lock(db, run_id="external-holder") is True

    stop_event = asyncio.Event()

    async def run_one_shot():
        # Stop almost immediately
        await asyncio.sleep(0.02)
        stop_event.set()

    stopper = asyncio.create_task(run_one_shot())
    await cadence_scheduler_loop(db, broadcaster, interval_s=0.01, stop_event=stop_event)
    await stopper

    # Broadcaster should never have started because lock was held
    assert broadcaster.active is False
    release_run_lock(db, run_id="external-holder")
    db.close()


def test_api_collect_status_and_sse_events(tmp_path):
    db = Database(tmp_path / "api_sse.db")
    app = create_app(db, enable_scheduler=False)
    client = TestClient(app)

    res_status = client.get("/api/collect/status")
    assert res_status.status_code == 200
    data = res_status.json()
    assert data["active"] is False
    assert data["stage"] is None

    # Test SSE stream returns initial event
    with client.stream("GET", "/api/collect/events") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        line = next(response.iter_lines())
        while not line or line.startswith(":"):
            line = next(response.iter_lines())
        assert line.startswith("data: ")
        assert "Broadcaster ready" in line

    db.close()


def test_api_collect_sync_and_background(tmp_path, monkeypatch):
    db = Database(tmp_path / "api_collect.db")
    app = create_app(db, enable_scheduler=False)
    client = TestClient(app)

    async def fake_tick(database, **kwargs):
        return {"sources_run": 2, "items_stored": 4, "sources_failed": 0}

    import gleaner.workflows
    monkeypatch.setattr(gleaner.workflows, "run_collect_tick", fake_tick)

    # 1. Synchronous collect
    res_sync = client.post("/api/collect")
    assert res_sync.status_code == 200
    assert res_sync.json()["status"] == "success"
    assert res_sync.json()["stats"]["items_stored"] == 4

    # 2. Background collect
    res_bg = client.post("/api/collect?background=true")
    assert res_bg.status_code == 200
    assert res_bg.json()["status"] == "started"
    assert "run_id" in res_bg.json()

    # 3. Conflict when lock already acquired
    run_id = "manual-lock"
    acquire_run_lock(db, run_id=run_id)
    res_conflict = client.post("/api/collect")
    assert res_conflict.status_code == 409
    release_run_lock(db, run_id=run_id)

    db.close()


def test_api_synthesize_question_answer(tmp_path, monkeypatch):
    db = Database(tmp_path / "api_synth.db")
    app = create_app(db, enable_scheduler=False)
    client = TestClient(app)

    q = Question(
        id="q-synth-test",
        text="Can small language models handle complex tool use?",
        linked_interests=["agents"],
    )
    db.upsert_question(q)

    # Unknown question returns 404
    res_404 = client.post("/api/questions/q-nonexistent/synthesize")
    assert res_404.status_code == 404

    # Seed an item and claim
    item = Item(
        id="it-1",
        title="Tool Calling Benchmark",
        text="Small 3B models achieve 80% on standard tool calls.",
        status="scored",
    )
    db.upsert_item(item)
    claim = Claim(
        id="c-1",
        question_id=q.id,
        item_id=item.id,
        text="Small 3B models achieve 80% on standard tool calls.",
        stance="supports",
        quote="Small 3B models achieve 80% on standard tool calls.",
        origin_key="benchmarks:tool-use",
    )
    db.upsert_claim(claim)

    # Mock answer model response
    mock_answer_json = (
        '{"sentences": [{"text": "Small 3B models achieve 80% on standard tool calls.", "claim_ids": ["c-1"]}], '
        '"what_changed": "Synthesized living answer from benchmark evidence."}'
    )
    monkeypatch.setattr(
        gleaner.llama,
        "get_llama_model",
        lambda role, **kw: MockModel(responses=[ModelResponse(content=mock_answer_json)]),
    )

    res = client.post(f"/api/questions/{q.id}/synthesize")
    assert res.status_code == 200
    data = res.json()
    assert data["question_id"] == q.id
    assert len(data["sentences"]) == 1
    assert data["sentences"][0]["claim_ids"] == ["c-1"]
    assert data["what_changed"] == "Synthesized living answer from benchmark evidence."
    assert data["synthesized_by"] == "mock"

    # Verify saved in DB
    saved = db.get_latest_answer(q.id)
    assert saved is not None
    assert saved.synthesized_by == "mock"
    assert saved.what_changed == "Synthesized living answer from benchmark evidence."

    db.close()


def test_cli_serve_scheduler_args(monkeypatch):
    import gleaner.cli as cli

    served_args: dict[str, Any] = {}

    def fake_create_app(db, enable_scheduler=False, scheduler_interval_s=30.0):
        served_args["enable_scheduler"] = enable_scheduler
        served_args["scheduler_interval_s"] = scheduler_interval_s
        return None

    monkeypatch.setattr("gleaner.server.create_app", fake_create_app)
    monkeypatch.setattr("uvicorn.run", lambda app, host, port: None)
    monkeypatch.setattr(cli, "get_default_db", lambda: Database(":memory:"))

    # Default serve has scheduler enabled
    cli.main(["serve", "--port", "9999"])
    assert served_args["enable_scheduler"] is True
    assert served_args["scheduler_interval_s"] == 30.0

    # With --no-scheduler
    cli.main(["serve", "--no-scheduler", "--scheduler-interval", "15.0"])
    assert served_args["enable_scheduler"] is False
    assert served_args["scheduler_interval_s"] == 15.0
