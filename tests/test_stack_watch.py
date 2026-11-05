"""Tests for M17 dependency watch: pyproject.toml parsing and OSV advisories."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from gleaner.db import Database
from gleaner.models import Question
from gleaner.questions import extract_claims_for_item
from gleaner.stack_watch import (
    check_stack_dependencies,
    fetch_osv_advisories,
    parse_pyproject_dependencies,
)


def test_parse_pyproject_dependencies(tmp_path: Path):
    sample_pyproject = tmp_path / "pyproject.toml"
    sample_pyproject.write_text(
        """
[project]
name = "demo-app"
version = "0.1.0"
dependencies = [
    "pydantic>=2.10.0",
    "httpx~=0.27.2",
    "sqlite-vec>=0.1.6",
    "graphflow",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff==0.8.0",
]
""",
        encoding="utf-8",
    )

    pkgs = parse_pyproject_dependencies(sample_pyproject)
    assert "pydantic" in pkgs
    assert "httpx" in pkgs
    assert "sqlite-vec" in pkgs
    assert "graphflow" in pkgs
    assert "pytest" in pkgs
    assert "ruff" in pkgs


@pytest.mark.asyncio
async def test_fetch_osv_advisories_offline():
    # Mock OSV API response
    def osv_handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content.decode("utf-8"))
        pkg_name = data.get("package", {}).get("name")
        if pkg_name == "httpx":
            return httpx.Response(
                200,
                json={
                    "vulns": [
                        {
                            "id": "GHSA-928r-24f4-5544",
                            "summary": "CRLF injection in httpx before 0.28.0",
                            "details": "A crafted request header can cause CRLF injection in httpx when parsing untrusted URLs.",
                            "published": "2024-10-15T00:00:00Z",
                        }
                    ]
                },
            )
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(osv_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        items = await fetch_osv_advisories(["httpx", "pydantic"], client=client)

    assert len(items) == 1
    advisory = items[0]
    assert advisory.id == "advisory:GHSA-928r-24f4-5544"
    assert advisory.kind == "advisory"
    assert "CRLF injection" in advisory.title
    assert "CRLF injection" in advisory.text
    assert advisory.url == "https://osv.dev/vulnerability/GHSA-928r-24f4-5544"
    assert advisory.source_id == "osv"
    assert advisory.status == "scored"


@pytest.mark.asyncio
async def test_check_stack_dependencies_integration(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    sample_pyproject = tmp_path / "pyproject.toml"
    sample_pyproject.write_text(
        """
[project]
name = "demo-app"
version = "0.1.0"
dependencies = [
    "httpx>=0.27.0",
]
""",
        encoding="utf-8",
    )

    def osv_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "vulns": [
                    {
                        "id": "GHSA-test-1111",
                        "summary": "Header spoofing vulnerability in httpx",
                        "details": "Improper sanitization of leading spaces in headers allows request smuggling.",
                        "published": "2024-10-20T12:00:00Z",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(osv_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        items = await check_stack_dependencies(db=db, pyproject_path=sample_pyproject, client=client)

    assert len(items) == 1
    stored = db.get_item(items[0].id)
    assert stored is not None
    assert stored.kind == "advisory"

    # Now verify claim extraction works on this advisory item:
    q = Question(
        id="q-stack-changes",
        text="What significant changes, breaking updates, or advisories affect my project stack?",
        watch=str(sample_pyproject),
    )
    db.upsert_question(q)

    # With heuristic extraction (model=None)
    claims = extract_claims_for_item(db=db, item=stored, questions=[q], model=None)
    assert isinstance(claims, list)
    # Let's verify origin_key is canonical url or advisory id
    assert stored.url == "https://osv.dev/vulnerability/GHSA-test-1111"
    db.close()


@pytest.mark.asyncio
async def test_stack_watch_asks_osv_about_the_installed_versions_only(tmp_path: Path):
    from importlib.metadata import version

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\ndependencies = ["httpx>=0.27", "not-installed-pkg-xyz>=1"]\n', encoding="utf-8")
    queries = []

    def osv_handler(request: httpx.Request) -> httpx.Response:
        queries.append(json.loads(request.content))
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(osv_handler)) as client:
        await check_stack_dependencies(db=Database(tmp_path / "v.db"), pyproject_path=pyproject, client=client)

    # A name-only query returns every advisory ever filed, including long-fixed ones
    assert queries == [{"package": {"name": "httpx", "ecosystem": "PyPI"}, "version": version("httpx")}]


ADVISORY = {
    "id": "GHSA-test-2222",
    "summary": "Request smuggling through crafted headers in httpx",
    "details": "Improper sanitization of leading spaces in headers.",
    "published": "2024-10-20T12:00:00Z",
}


@pytest.mark.asyncio
async def test_collect_runs_the_stack_watch_daily_and_answers_the_stack_question(tmp_path: Path):
    from gleaner.models import SourceRow
    from gleaner.questions import STACK_QUESTION_ID
    from gleaner.workflows import run_collect_tick

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\ndependencies = ["httpx>=0.27"]\n', encoding="utf-8")
    db = Database(tmp_path / "tick.db")
    db.upsert_question(Question(id="q-lg", text="What are the common failure modes of LangGraph in production?",
                                watch=str(pyproject)))
    db.upsert_source(SourceRow(id="blog", type="rss", params={"url": "https://example.com/rss.xml"}, cadence_s=60))
    osv_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.osv.dev":
            osv_calls.append(request)
            return httpx.Response(200, json={"vulns": [ADVISORY]})
        return httpx.Response(200, text='<rss version="2.0"><channel><title>b</title></channel></rss>')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await run_collect_tick(db, client=client)
        with db.connect() as conn:
            conn.execute("UPDATE sources SET next_due_at = NULL WHERE id = 'blog';")
        await run_collect_tick(db, client=client)

    assert len(osv_calls) == 1  # once a day, not every tick
    assert db.get_question(STACK_QUESTION_ID).text == "What changed in my stack?"
    claims = db.list_claims_for_question(STACK_QUESTION_ID)
    assert [c.quote for c in claims] == [ADVISORY["summary"]]  # a 2024 advisory still affects the installed version
    assert db.get_latest_answer(STACK_QUESTION_ID) is not None
    assert db.list_claims_for_question("q-lg") == []
    assert db.get_source("osv").failures == 0

