"""Phase 4: Run lock and scheduler cadence tests."""

from datetime import UTC, datetime, timedelta

from gleaner.db import Database
from gleaner.models import SourceRow
from gleaner.workflows import acquire_run_lock, is_source_due, release_run_lock


def test_collect_run_lock_prevents_overlap(tmp_path):
    db = Database(tmp_path / "lock.db")
    run_id_1 = "run-001"
    run_id_2 = "run-002"

    # First process acquires run lock
    acquired = acquire_run_lock(db, run_id=run_id_1, workflow="collect")
    assert acquired is True

    # Concurrent attempt fails to acquire lock
    concurrent_attempt = acquire_run_lock(db, run_id=run_id_2, workflow="collect")
    assert concurrent_attempt is False

    # First process releases lock
    release_run_lock(db, run_id=run_id_1)

    # Now second attempt succeeds
    second_attempt = acquire_run_lock(db, run_id=run_id_2, workflow="collect")
    assert second_attempt is True
    release_run_lock(db, run_id=run_id_2)
    db.close()


def test_scheduler_cadence_and_due_checks():
    now = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
    src_30m = SourceRow(
        id="src-30",
        type="rss",
        cadence_s=1800,
        next_due_at=now,
    )
    # At t=0, next_due_at == now -> due
    assert is_source_due(src_30m, current_time=now) is True

    # At t=10m, next_due_at is in future -> not due
    future = now + timedelta(minutes=10)
    src_30m.next_due_at = now + timedelta(minutes=30)
    assert is_source_due(src_30m, current_time=future) is False

    # At t=30m -> due
    due_time = now + timedelta(minutes=30)
    assert is_source_due(src_30m, current_time=due_time) is True


def test_collect_tick_handles_naive_due_times_and_disabled_sources(tmp_path):
    import asyncio

    import httpx
    from graphflow import MockModel

    from gleaner.workflows import run_collect_tick

    db = Database(tmp_path / "due.db")
    past_naive = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=5)
    future_naive = past_naive + timedelta(hours=2)
    db.upsert_source(SourceRow(id="due", type="rss", params={"url": "https://a.example.com/rss"}, next_due_at=past_naive))
    db.upsert_source(SourceRow(id="later", type="rss", params={"url": "https://b.example.com/rss"}, next_due_at=future_naive))
    db.upsert_source(SourceRow(id="off", type="rss", params={"url": "https://c.example.com/rss"}, enabled=False))
    empty_feed = "<rss><channel><title>t</title></channel></rss>"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=empty_feed))

    async def tick():
        async with httpx.AsyncClient(transport=transport) as client:
            return await run_collect_tick(db, client=client, model=MockModel())

    stats = asyncio.run(tick())

    assert stats["sources_run"] == 1
    assert db.get_source("due").next_due_at.replace(tzinfo=UTC) > datetime.now(UTC)
    db.close()
