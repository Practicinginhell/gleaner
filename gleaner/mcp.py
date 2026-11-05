"""Model Context Protocol (MCP) Server for Gleaner (ADR-0003, tasks Y).

Provides a local, zero-dependency JSON-RPC 2.0 stdio server conforming to the Model Context
Protocol specification (version 2024-11-05). Exposes read-only tools so external agent harnesses
(Block Buzz / Goose, Claude Code, Cursor) can query Gleaner's proactive discovery state:
- Standing questions and living cited answers
- Grounded verbatim evidence claims and source citations
- Top-ranked feed items and multi-perspective stories
- Early insights ("Ahead of Hacker News" leads)
- Comprehensive research dossiers (Obsidian/PKM Markdown)
"""

from __future__ import annotations

import json
import logging
import sys
from typing import IO, Any

from gleaner.db import Database
from gleaner.dossier import generate_dossier
from gleaner.feed import generate_feed
from gleaner.hn_lead import hn_leads

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "gleaner"
SERVER_VERSION = "0.1.0"

# Exposed MCP Tools specification
TOOLS_SPEC = [
    {
        "name": "gleaner_list_questions",
        "description": "List all active standing questions being tracked by gleaner.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "gleaner_get_standing_answer",
        "description": "Get the latest living cited answer for a standing question, including confidence level, delta (what changed), and grounded verbatim quotes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question_id": {
                    "type": "string",
                    "description": "The ID of the standing question (e.g. 'q-1', 'q-stack').",
                },
            },
            "required": ["question_id"],
        },
    },
    {
        "name": "gleaner_search_evidence",
        "description": "Search verified verbatim evidence quotes and claims stored in gleaner by topic or keyword.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query or keywords.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of claims to return (default: 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "gleaner_get_feed",
        "description": "Get top-ranked intelligence and story cards filtered and scored by Meta Llama.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of cards to return (default: 5).",
                },
                "mode": {
                    "type": "string",
                    "description": "Ranking mode: 'balanced', 'deep_dive', or 'novelty' (default: 'balanced').",
                },
            },
        },
    },
    {
        "name": "gleaner_get_hn_leads",
        "description": "Get early technical insights surfaced on followed sources days before appearing on Hacker News.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of leads to return (default: 5).",
                },
            },
        },
    },
    {
        "name": "gleaner_get_dossier",
        "description": "Generate a comprehensive, portable research dossier in Markdown format containing all standing answers, evidence citations, story cards, and early leads.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


class MCPServer:
    """Read-only Model Context Protocol server over stdio for Gleaner.

    Attributes:
        db: Database the tools read from.
    """

    def __init__(self, db: Database) -> None:
        """Bind the server to a database.

        Args:
            db: Database the tools read from.
        """
        self.db = db

    def process_message(self, req: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single JSON-RPC 2.0 message.

        Args:
            req: Decoded JSON-RPC request or notification.

        Returns:
            The JSON-RPC response, or None for a notification that needs no response.
        """
        method = req.get("method")
        msg_id = req.get("id")

        # 1. Notifications (no response): JSON-RPC never answers a message without an id
        if method == "notifications/initialized" or "id" not in req:
            return None

        # 2. Ping
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        # 3. Initialize
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {},
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                },
            }

        # 4. Tools list
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": TOOLS_SPEC,
                },
            }

        # 5. Tools call
        if method == "tools/call":
            params = req.get("params", {})
            tool_name = params.get("name")
            args = params.get("arguments", {})
            return self._handle_tool_call(msg_id, tool_name, args)

        # Unknown method error (-32601)
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }

    def _handle_tool_call(self, msg_id: Any, tool_name: str | None, args: dict[str, Any]) -> dict[str, Any]:
        """Run one tool and wrap its text in a JSON-RPC response.

        An unknown tool or a failing one comes back as an `isError` result, never an exception,
        so one bad call can't end the stdio session.

        Args:
            msg_id: JSON-RPC id of the request, echoed in the response.
            tool_name: Name of the tool to run.
            args: The tool's arguments from the request.

        Returns:
            The JSON-RPC response with the tool's text content.
        """
        try:
            if tool_name == "gleaner_list_questions":
                res = self._tool_list_questions()
            elif tool_name == "gleaner_get_standing_answer":
                res = self._tool_get_standing_answer(str(args.get("question_id", "")))
            elif tool_name == "gleaner_search_evidence":
                res = self._tool_search_evidence(str(args.get("query", "")), int(args.get("limit", 5)))
            elif tool_name == "gleaner_get_feed":
                res = self._tool_get_feed(int(args.get("limit", 5)), str(args.get("mode", "balanced")))
            elif tool_name == "gleaner_get_hn_leads":
                res = self._tool_get_hn_leads(int(args.get("limit", 5)))
            elif tool_name == "gleaner_get_dossier":
                res = self._tool_get_dossier()
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                        "isError": True,
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": res}],
                },
            }
        except Exception as exc:
            logger.exception("Error executing MCP tool %s: %s", tool_name, exc)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error executing tool {tool_name}: {exc}"}],
                    "isError": True,
                },
            }

    def _tool_list_questions(self) -> str:
        """Return the active standing questions as JSON."""
        questions = self.db.list_questions()
        data = [
            {
                "id": q.id,
                "text": q.text,
                "active": q.active,
                "watch": q.watch,
                "linked_interests": q.linked_interests,
            }
            for q in questions
        ]
        return json.dumps(data, indent=2)

    def _tool_get_standing_answer(self, question_id: str) -> str:
        """Return a question's latest answer, its cited quotes and its claims as JSON.

        Args:
            question_id: Id of the standing question.

        Returns:
            The answer as JSON, or a JSON error object when the question has no answer yet.
        """
        latest = self.db.get_latest_answer(question_id)
        if not latest:
            return json.dumps({"error": f"No answer found for question '{question_id}'."})

        claims = self.db.list_claims_for_question(question_id)
        claim_map = {c.id: c for c in claims}

        data = {
            "question_id": latest.question_id,
            "version": latest.version,
            "confidence": latest.confidence,
            "what_changed": latest.what_changed,
            "what_changed_by": latest.what_changed_by,
            "synthesized_by": latest.synthesized_by,
            "sentences": [
                {
                    "text": s.text,
                    "claim_ids": s.claim_ids,
                    "citations": [
                        {
                            "id": cid,
                            "quote": claim_map[cid].quote if cid in claim_map else "",
                            "origin_key": claim_map[cid].origin_key if cid in claim_map else "",
                            "stance": claim_map[cid].stance if cid in claim_map else "",
                        }
                        for cid in s.claim_ids
                        if cid in claim_map
                    ],
                }
                for s in latest.sentences
            ],
            "claims": [
                {
                    "id": c.id,
                    "text": c.text,
                    "quote": c.quote,
                    "stance": c.stance,
                    "origin_key": c.origin_key,
                    "model": c.model,
                }
                for c in claims
            ],
        }
        return json.dumps(data, indent=2)

    def _tool_search_evidence(self, query: str, limit: int = 5) -> str:
        """Return claims whose text or quote contains the query, as JSON.

        Args:
            query: Case-insensitive substring to match.
            limit: Maximum number of claims to return.

        Returns:
            The matching claims as a JSON list.
        """
        q_norm = query.strip().lower()
        questions = self.db.list_questions()
        matched_claims = []
        for q in questions:
            for c in self.db.list_claims_for_question(q.id):
                if q_norm in c.text.lower() or q_norm in c.quote.lower():
                    matched_claims.append({
                        "id": c.id,
                        "question_id": c.question_id,
                        "text": c.text,
                        "quote": c.quote,
                        "stance": c.stance,
                        "origin_key": c.origin_key,
                        "model": c.model,
                    })
                    if len(matched_claims) >= limit:
                        break
            if len(matched_claims) >= limit:
                break
        return json.dumps(matched_claims, indent=2)

    def _tool_get_feed(self, limit: int = 5, mode: str = "balanced") -> str:
        """Return the top feed cards with their score components as JSON.

        Args:
            limit: Number of cards to return.
            mode: Ranking mode passed to `generate_feed`.

        Returns:
            The cards as a JSON list.
        """
        cards = generate_feed(self.db, limit=limit, mode=mode)
        data = [
            {
                "id": c.item.id,
                "title": c.item.title,
                "url": c.item.url,
                "source_id": c.item.source_id,
                "kind": c.item.kind,
                "score": c.score,
                "relevance": c.components.relevance,
                "novelty": c.components.novelty,
                "quality": c.components.quality,
                "why": c.why,
                "summary": c.summary,
                "perspectives": c.perspectives,
                "also_covered_by": c.also_covered_by,
            }
            for c in cards
        ]
        return json.dumps(data, indent=2)

    def _tool_get_hn_leads(self, limit: int = 5) -> str:
        """Return items seen before they reached Hacker News, with the lead time, as JSON.

        Args:
            limit: Number of leads to return.

        Returns:
            The leads as a JSON list.
        """
        leads = hn_leads(self.db, limit=limit)
        data = [
            {
                "title": lead.item.title,
                "url": lead.item.url,
                "lead_seconds": lead.lead.total_seconds(),
                "lead_formatted": (
                    f"{lead.lead.total_seconds() / 86400.0:.1f} days"
                    if lead.lead.total_seconds() >= 86400
                    else f"{lead.lead.total_seconds() / 3600.0:.1f} hours"
                ),
                "hn_url": lead.hn_url,
                "hn_points": lead.points,
            }
            for lead in leads[:limit]
        ]
        return json.dumps(data, indent=2)

    def _tool_get_dossier(self) -> str:
        """Return the research dossier as Markdown."""
        return generate_dossier(self.db)

    def process_stream(self, stdin: IO[str], stdout: IO[str]) -> None:
        """Process JSON-RPC messages line by line over streams.

        A line that isn't valid JSON gets a parse error (-32700), one that isn't an object an invalid
        request error (-32600), and a message that fails an internal error (-32603). The loop goes on.

        Args:
            stdin: Stream of newline-delimited JSON-RPC messages.
            stdout: Stream the responses are written to, one per line.
        """
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                stdout.write(json.dumps(err_resp) + "\n")
                stdout.flush()
                continue

            if not isinstance(req, dict):
                resp: dict[str, Any] | None = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request: expected a JSON object"},
                }
            else:
                try:
                    resp = self.process_message(req)
                except Exception as exc:  # noqa: BLE001 - one bad message must not end the server
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req.get("id"),
                        "error": {"code": -32603, "message": f"Internal error: {exc}"},
                    }
            if resp is not None:
                stdout.write(json.dumps(resp) + "\n")
                stdout.flush()

    def run_stdio(self) -> None:
        """Run MCP server listening on sys.stdin and writing to sys.stdout."""
        self.process_stream(sys.stdin, sys.stdout)
