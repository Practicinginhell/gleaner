"""Tests for Nostr relay source connector (NIP-01, NIP-23)."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from gleaner.config import SourceConfig
from gleaner.sources.nostr import NostrSource, parse_nostr_event_to_item


def test_parse_nostr_event_to_item_nip23():
    event = {
        "id": "437b0687d3c009f4ab345ab335b32874b12c0c309c6c4c6d3b3de1a0b425415a",
        "pubkey": "npub1testauthor89abcdef0123456789abcdef0123456789abcdef0123456789",
        "created_at": 1730000000,
        "kind": 30023,
        "tags": [
            ["d", "reasoning-scaling"],
            ["title", "Test-Time Compute Scaling in Terminal Agents"],
            ["summary", "Investigating inference-time search at the harness boundary."],
            ["t", "ai"],
            ["t", "agents"],
        ],
        "content": "# Test-Time Compute Scaling\n\nAllocating compute at the harness boundary improves task success.",
    }
    item = parse_nostr_event_to_item(event, source_id="nostr-feed")
    assert item is not None
    assert item.id == "nostr-437b0687d3c009f4"
    assert item.title == "Test-Time Compute Scaling in Terminal Agents"
    assert item.kind == "paper" or item.kind == "article"
    assert "https://njump.me/" in item.url
    assert "Allocating compute" in item.text
    assert item.published_at == datetime.fromtimestamp(1730000000, tz=UTC)


def test_parse_nostr_event_to_item_kind1():
    event = {
        "id": "1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
        "pubkey": "pubkey123456",
        "created_at": 1730001000,
        "kind": 1,
        "tags": [],
        "content": "Just deployed the new local agent runtime. Zero cloud dependencies!",
    }
    item = parse_nostr_event_to_item(event, source_id="nostr-feed")
    assert item is not None
    assert item.id == "nostr-1111222233334444"
    assert item.kind == "discussion" or item.kind == "post"
    assert "Just deployed the new local agent runtime" in item.title
    assert "Zero cloud dependencies" in item.text


@pytest.mark.asyncio
async def test_nostr_source_fetch_via_http_gateway(monkeypatch):
    import gleaner.sources.nostr

    monkeypatch.setattr(gleaner.sources.nostr, "validate_safe_url", lambda url: url)  # offline: no DNS
    events = [
        {
            "id": "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899",
            "pubkey": "author123",
            "created_at": 1730002000,
            "kind": 30023,
            "tags": [["title", "Decentralized Agent Workspaces"]],
            "content": "Exploring peer-to-peer agent collaboration architectures.",
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"events": events})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        cfg = SourceConfig(
            id="nostr-src",
            type="nostr",
            params={"relay": "https://r.nostr.example/api", "kinds": [30023]},
            cadence="1h",
        )
        source = NostrSource()
        res = await source.fetch(cfg, since=None, client=client)
        assert len(res.items) == 1
        assert res.items[0].title == "Decentralized Agent Workspaces"


@pytest.mark.asyncio
async def test_nostr_source_handles_network_failure_gracefully(monkeypatch):
    import gleaner.sources.nostr

    monkeypatch.setattr(gleaner.sources.nostr, "validate_safe_url", lambda url: url)  # offline: no DNS
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        cfg = SourceConfig(
            id="nostr-src",
            type="nostr",
            params={"relay": "https://broken.nostr.example/api"},
            cadence="1h",
        )
        source = NostrSource()
        res = await source.fetch(cfg, since=None, client=client)
        assert res.items == []


def test_nostr_sources_can_be_configured():
    from gleaner.config import VALID_SOURCE_TYPES

    assert "nostr" in VALID_SOURCE_TYPES


@pytest.mark.parametrize("relay", ["http://127.0.0.1:8080/req", "ws://localhost:7000", "wss://10.0.0.5"])
def test_nostr_refuses_relays_on_private_addresses(relay):
    import asyncio

    cfg = SourceConfig(id="n", type="nostr", params={"relay": relay})
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("fetched a private relay")))
    with pytest.raises(ValueError):
        asyncio.run(NostrSource().fetch(cfg, None, client))
