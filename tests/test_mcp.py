"""Tests for Gleaner Model Context Protocol (MCP) server.

Verifies JSON-RPC 2.0 stdio compliance:
- initialize and protocol negotiation
- ping
- tools/list schema advertisement
- tools/call execution across all exposed read-only tools
- error handling (unknown method, invalid params)
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta

from gleaner.db import Database
from gleaner.mcp import MCPServer
from gleaner.models import Answer, Claim, Item, Question, Sentence


def _create_test_db() -> Database:
    db = Database(":memory:")
    # Seed a source
    from gleaner.models import SourceRow

    db.upsert_source(SourceRow(id="arxiv", type="arxiv", params={}, cadence_s=3600, trust=1.0, enabled=True))

    # Seed a question
    q = Question(
        id="q-test-1",
        text="Does test-time compute scaling generalize across domain tasks?",
        active=True,
        watch="pyproject.toml",
        linked_interests=["reasoning", "benchmarks"],
        created_at=datetime.now(UTC),
    )
    db.upsert_question(q)

    # Seed an item
    it = Item(
        id="item-test-1",
        source_id="arxiv",
        title="Mid-Harness: Scaling Actions Between Model and Harness for Terminal Agents",
        url="https://arxiv.org/abs/2410.12345",
        text="We investigate whether allocating test-time compute at the model-harness boundary can improve action reliability.",
        status="scored",
        kind="paper",
        published_at=datetime.now(UTC) - timedelta(days=2),
        created_at=datetime.now(UTC),
    )
    db.upsert_item(it)
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO scores (item_id, profile_version, components, final, computed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                it.id,
                1,
                json.dumps({"relevance": 0.92, "novelty": 0.85, "quality": 0.90, "actionability": 0.8}),
                0.89,
                datetime.now(UTC).isoformat(),
            ),
        )

    # Seed a claim
    cl = Claim(
        id="claim-test-1",
        question_id=q.id,
        item_id=it.id,
        text="Allocating test-time compute improves action reliability.",
        quote="We investigate whether allocating test-time compute at the model-harness boundary can improve action reliability.",
        stance="supports",
        origin_key="arxiv:2410.12345",
        model="llama3.2:3b",
        created_at=datetime.now(UTC),
    )
    db.upsert_claim(cl)

    # Seed an answer
    ans = Answer(
        question_id=q.id,
        version=1,
        sentences=[Sentence(text="Allocating test-time compute improves action reliability.", claim_ids=[cl.id])],
        confidence="high",
        counterpoint_claim_id=None,
        what_changed="Initial baseline from arXiv pre-print.",
        synthesized_by="llama3.2:3b",
        created_at=datetime.now(UTC),
    )
    db.upsert_answer(ans)
    return db


def _run_rpc(server: MCPServer, request: dict) -> dict | None:
    stdin = io.StringIO(json.dumps(request) + "\n")
    stdout = io.StringIO()
    server.process_stream(stdin, stdout)
    output = stdout.getvalue().strip()
    if not output:
        return None
    return json.loads(output)


def test_mcp_initialize():
    db = _create_test_db()
    server = MCPServer(db)
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }
    resp = _run_rpc(server, req)
    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert "result" in resp
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert "tools" in resp["result"]["capabilities"]
    assert resp["result"]["serverInfo"]["name"] == "gleaner"


def test_mcp_ping():
    db = _create_test_db()
    server = MCPServer(db)
    req = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
    resp = _run_rpc(server, req)
    assert resp == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_mcp_tools_list():
    db = _create_test_db()
    server = MCPServer(db)
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
    resp = _run_rpc(server, req)
    assert resp is not None
    tools = resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "gleaner_list_questions" in tool_names
    assert "gleaner_get_standing_answer" in tool_names
    assert "gleaner_search_evidence" in tool_names
    assert "gleaner_get_feed" in tool_names
    assert "gleaner_get_hn_leads" in tool_names
    assert "gleaner_get_dossier" in tool_names


def test_mcp_tool_call_list_questions():
    db = _create_test_db()
    server = MCPServer(db)
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "gleaner_list_questions", "arguments": {}},
    }
    resp = _run_rpc(server, req)
    assert resp is not None
    assert "result" in resp
    content = resp["result"]["content"][0]["text"]
    data = json.loads(content)
    assert len(data) >= 1
    assert data[0]["id"] == "q-test-1"
    assert "test-time compute" in data[0]["text"]


def test_mcp_tool_call_get_standing_answer():
    db = _create_test_db()
    server = MCPServer(db)
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "gleaner_get_standing_answer", "arguments": {"question_id": "q-test-1"}},
    }
    resp = _run_rpc(server, req)
    assert resp is not None
    content = resp["result"]["content"][0]["text"]
    data = json.loads(content)
    assert data["question_id"] == "q-test-1"
    assert data["confidence"] == "high"
    assert data["synthesized_by"] == "llama3.2:3b"
    assert len(data["claims"]) >= 1


def test_mcp_tool_call_search_evidence():
    db = _create_test_db()
    server = MCPServer(db)
    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {"name": "gleaner_search_evidence", "arguments": {"query": "compute", "limit": 5}},
    }
    resp = _run_rpc(server, req)
    assert resp is not None
    content = resp["result"]["content"][0]["text"]
    data = json.loads(content)
    assert len(data) >= 1
    assert "allocating test-time compute" in data[0]["quote"].lower()


def test_mcp_tool_call_get_feed():
    db = _create_test_db()
    server = MCPServer(db)
    req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "gleaner_get_feed", "arguments": {"limit": 5, "mode": "balanced"}},
    }
    resp = _run_rpc(server, req)
    assert resp is not None
    content = resp["result"]["content"][0]["text"]
    data = json.loads(content)
    assert len(data) >= 1
    assert "Mid-Harness" in data[0]["title"]


def test_mcp_tool_call_get_dossier():
    db = _create_test_db()
    server = MCPServer(db)
    req = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {"name": "gleaner_get_dossier", "arguments": {}},
    }
    resp = _run_rpc(server, req)
    assert resp is not None
    content = resp["result"]["content"][0]["text"]
    assert "# Gleaner Intelligence Dossier" in content
    assert "Standing Questions & Living Answers" in content


def test_mcp_error_handling():
    db = _create_test_db()
    server = MCPServer(db)
    # 1. Unknown method
    req = {"jsonrpc": "2.0", "id": 9, "method": "nonexistent_method", "params": {}}
    resp = _run_rpc(server, req)
    assert resp is not None
    assert resp["error"]["code"] == -32601

    # 2. Unknown tool
    req = {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "unknown_tool", "arguments": {}}}
    resp = _run_rpc(server, req)
    assert resp is not None
    assert resp["result"]["isError"] is True
    assert "Unknown tool" in resp["result"]["content"][0]["text"]


def test_mcp_hn_leads_tool_returns_a_real_lead():
    """The tool read a field HNLead doesn't have, so it failed whenever a lead existed."""
    db = _create_test_db()
    now = datetime.now(UTC)
    db.upsert_hn_sighting("item-test-1", "123", now, 50, now)
    resp = _run_rpc(MCPServer(db), {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                                    "params": {"name": "gleaner_get_hn_leads", "arguments": {"limit": 3}}})
    assert "error" not in resp and not resp["result"].get("isError"), resp
    data = json.loads(resp["result"]["content"][0]["text"])
    assert data[0]["hn_points"] == 50 and data[0]["hn_url"].endswith("id=123")


def test_mcp_survives_malformed_messages_and_ignores_unknown_notifications():
    server = MCPServer(_create_test_db())
    lines = ["[1]", '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": null}',
             '{"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}}',
             '{"jsonrpc": "2.0", "id": 2, "method": "ping"}']
    stdout = io.StringIO()
    server.process_stream(io.StringIO("\n".join(lines) + "\n"), stdout)
    replies = [json.loads(x) for x in stdout.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [None, 1, 2]  # no reply to the notification
    assert replies[0]["error"]["code"] == -32600 and "error" in replies[1]
    assert replies[2]["result"] == {}
