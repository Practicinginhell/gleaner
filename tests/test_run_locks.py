"""The collect run lock: one holder at a time, released on every way out, and a tick that
leaves the server responsive."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

import gleaner.llama
import gleaner.workflows
from gleaner import cli
from gleaner.db import Database
from gleaner.models import SourceRow
from gleaner.server import CollectProgressBroadcaster, cadence_scheduler_loop, create_app
from gleaner.workflows import acquire_run_lock, release_run_lock, run_collect_tick


def _running(db: Database) -> list[str]:
    with db.connect() as conn:
        return [r["id"] for r in conn.execute("SELECT id FROM runs WHERE status = 'running';")]


def _due_source(db: Database) -> None:
    db.upsert_source(
        SourceRow(
            id="src-due",
            type="rss",
            params={"url": "https://example.com/rss"},
            cadence_s=3600,
            next_due_at=datetime.now(UTC) - timedelta(minutes=5),
        )
    )


def test_two_processes_never_hold_the_lock_at_once(tmp_path):
    db = Database(tmp_path / "race.db")
    for trial in range(60):
        barrier = threading.Barrier(2)
        won: list[bool] = []

        def take(n: int, barrier=barrier, won=won, trial=trial) -> None:
            barrier.wait()
            won.append(acquire_run_lock(db, run_id=f"t{trial}-{n}"))

        threads = [threading.Thread(target=take, args=(n,)) for n in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(won) == [False, True], f"trial {trial}: {won}"
        for run_id in _running(db):
            release_run_lock(db, run_id=run_id)
    db.close()


@pytest.mark.asyncio
async def test_shutting_down_mid_tick_releases_the_lock(tmp_path, monkeypatch):
    db = Database(tmp_path / "cancel.db")
    _due_source(db)
    started = asyncio.Event()

    async def slow_tick(database, **kwargs):
        started.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(gleaner.workflows, "run_collect_tick", slow_tick)
    monkeypatch.setattr(gleaner.llama, "get_llama_model", lambda role: None)
    broadcaster = CollectProgressBroadcaster()
    task = asyncio.create_task(cadence_scheduler_loop(db, broadcaster, interval_s=0.01))
    await asyncio.wait_for(started.wait(), timeout=3)
    task.cancel()  # what the server's lifespan does on shutdown
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _running(db) == []
    assert broadcaster.active is False
    assert acquire_run_lock(db, run_id="next") is True
    db.close()


def test_a_model_config_error_does_not_lock_collect(tmp_path, monkeypatch):
    db = Database(tmp_path / "config_error.db")

    def bad_model(role):
        raise ValueError("'gpt-4o' is not a Meta model")

    monkeypatch.setattr(gleaner.llama, "get_llama_model", bad_model)
    client = TestClient(create_app(db), raise_server_exceptions=False)
    assert client.post("/api/collect").status_code == 500
    assert _running(db) == []
    db.close()


def test_server_shutdown_cancels_a_background_collect_and_frees_the_lock(tmp_path, monkeypatch):
    db = Database(tmp_path / "bg.db")

    async def endless_tick(database, **kwargs):
        await asyncio.sleep(60)

    monkeypatch.setattr(gleaner.workflows, "run_collect_tick", endless_tick)
    monkeypatch.setattr(gleaner.llama, "get_llama_model", lambda role: None)
    with TestClient(create_app(db)) as client:
        assert client.post("/api/collect?background=true").json()["status"] == "started"
        assert len(_running(db)) == 1
    assert _running(db) == []
    db.close()


def test_ctrl_c_during_cli_collect_releases_the_lock(tmp_path, monkeypatch):
    db_path = tmp_path / "ctrl_c.db"
    monkeypatch.setattr(cli, "get_default_db", lambda: Database(db_path))

    async def interrupted(database, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(gleaner.workflows, "run_collect_tick", interrupted)
    monkeypatch.setattr(gleaner.llama, "get_llama_model", lambda role: None)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["collect"])
    assert _running(Database(db_path)) == []


def test_sse_does_not_replay_the_previous_runs_end(tmp_path):
    db = Database(tmp_path / "sse.db")
    app = create_app(db)
    app.state.broadcaster.start_run("old-run")
    app.state.broadcaster.finish_run({"items_stored": 1})
    client = TestClient(app)
    with client.stream("GET", "/api/collect/events") as response:
        line = next(line for line in response.iter_lines() if line.startswith("data: "))
    # the dashboard closes the stream on "complete": the next run's events would be lost
    assert '"stage": "idle"' in line
    db.close()


@pytest.mark.asyncio
async def test_slow_judging_does_not_freeze_the_server(tmp_path, monkeypatch):
    db = Database(tmp_path / "loop.db")
    _due_source(db)

    def slow_answers(*args, **kwargs):
        time.sleep(0.5)  # a Llama call: synchronous, seconds long
        return {"claims_added": 0, "answers_changed": 0}

    monkeypatch.setattr(gleaner.workflows, "update_standing_answers", slow_answers)
    stalls: list[float] = []

    async def heartbeat() -> None:
        while True:
            t = time.monotonic()
            await asyncio.sleep(0.02)
            stalls.append(time.monotonic() - t)

    beat = asyncio.create_task(heartbeat())
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404))) as client:
        await run_collect_tick(db, client=client)
    beat.cancel()
    assert max(stalls, default=float("inf")) < 0.25  # empty: the heartbeat never got to run
    db.close()
