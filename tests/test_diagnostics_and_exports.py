"""Tests for diagnostics, exports and per-source Fetch now:
- Environment & engine diagnostics (/api/env/check)
- 1-Click Feed export (/api/export/feed?format=json, csv)
- 1-Click Digest export (/api/export/digest?format=markdown, json)
- Per-source 'Fetch Now' on-demand endpoint (/api/sources/{id}/fetch)
- Per-source fetch log drawer endpoint (/api/sources/{id}/logs)
- UI buttons for export, fetch-now, and diagnostics
"""

from fastapi.testclient import TestClient

from gleaner.db import Database
from gleaner.models import Item, SourceRow
from gleaner.server import create_app


def test_env_check_endpoint(tmp_path):
    """GET /api/env/check returns system health and diagnostics."""
    db = Database(tmp_path / "env_check.db")
    src = SourceRow(id="src-1", type="rss", params={"url": "http://test.com"}, cadence_s=3600)
    db.upsert_source(src)
    item = Item(id="it-1", source_id="src-1", title="Test Item", text="Body text", status="scored")
    db.upsert_item(item)

    client = TestClient(create_app(db))
    res = client.get("/api/env/check")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "python_version" in data
    assert "sqlite_vec" in data
    assert data["total_sources"] == 1
    assert data["items_read"] == 1
    db.close()


def test_export_feed_endpoints(tmp_path):
    """GET /api/export/feed supports JSON and CSV attachments."""
    db = Database(tmp_path / "export_feed.db")
    src = SourceRow(id="src-rss", type="rss", params={"url": "http://feed.com"}, cadence_s=3600)
    db.upsert_source(src)
    item = Item(id="it-exp-1", source_id="src-rss", title="Exportable Post", text="Post summary", status="scored")
    db.upsert_item(item)
    # Add a score so it appears in feed
    with db.connect() as conn:
        conn.execute("INSERT INTO scores (item_id, final, components) VALUES ('it-exp-1', 0.85, '{}');")

    client = TestClient(create_app(db))

    # 1. JSON Export
    res_json = client.get("/api/export/feed?format=json")
    assert res_json.status_code == 200
    assert "application/json" in res_json.headers["content-type"]
    assert "attachment" in res_json.headers.get("content-disposition", "")
    data = res_json.json()
    assert len(data) >= 1
    assert data[0]["item"]["title"] == "Exportable Post"

    # 2. CSV Export
    res_csv = client.get("/api/export/feed?format=csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
    assert "attachment" in res_csv.headers.get("content-disposition", "")
    lines = res_csv.text.strip().splitlines()
    assert len(lines) >= 2  # header + row
    assert "Exportable Post" in lines[1]
    assert "id,score,relevance" in lines[0]
    assert "source_id,kind,title,url" in lines[0]

    # 3. Invalid format
    res_bad = client.get("/api/export/feed?format=unsupported")
    assert res_bad.status_code == 400

    db.close()


def test_export_digest_endpoints(tmp_path):
    """GET /api/export/digest supports Markdown and JSON attachments."""
    db = Database(tmp_path / "export_digest.db")
    client = TestClient(create_app(db))

    # 1. Markdown Export
    res_md = client.get("/api/export/digest?format=markdown")
    assert res_md.status_code == 200
    assert "text/markdown" in res_md.headers["content-type"]
    assert "attachment" in res_md.headers.get("content-disposition", "")
    assert "Daily Briefing" in res_md.text

    # 2. JSON Export
    res_json = client.get("/api/export/digest?format=json")
    assert res_json.status_code == 200
    assert "application/json" in res_json.headers["content-type"]
    assert "Daily Briefing" in res_json.json()["title"]

    db.close()


async def test_single_source_fetch_and_logs(tmp_path):
    """POST /api/sources/{id}/fetch runs immediate fetch and updates fetch_log."""
    import gleaner.workflows
    from gleaner.sources.base import FetchResult

    db = Database(tmp_path / "single_fetch.db")
    src = SourceRow(id="src-custom", type="rss", params={"url": "http://single.feed"}, cadence_s=3600)
    db.upsert_source(src)

    class MockSingleConnector:
        async def fetch(self, *a, **k):
            item = Item(id="single-it-1", source_id="src-custom", title="Single Item", text="Fetched on demand")
            return FetchResult(items=[item], status_code=200)

    from unittest.mock import patch
    with patch.object(gleaner.workflows, "get_connector", return_value=MockSingleConnector()):
        client = TestClient(create_app(db))

        # 1. Fetch single source
        res = client.post("/api/sources/src-custom/fetch")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["source_id"] == "src-custom"
        assert data["items_fetched"] == 1

        # 2. Unknown source returns 404
        res_unknown = client.post("/api/sources/unknown-source/fetch")
        assert res_unknown.status_code == 404

        # 3. View fetch logs
        res_logs = client.get("/api/sources/src-custom/logs")
        assert res_logs.status_code == 200
        logs = res_logs.json()
        assert len(logs) >= 1
        assert logs[0]["status_code"] == 200
        assert logs[0]["item_count"] == 1

    db.close()
