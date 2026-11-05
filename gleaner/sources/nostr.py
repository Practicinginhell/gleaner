"""Nostr relay source connector (NIP-01, NIP-23).

Ingests decentralized technical articles (NIP-23 long-form content, kind 30023) and technical
notes (kind 1) from Nostr relays over WebSockets or HTTP relay gateways.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

import httpx

from gleaner.config import SourceConfig
from gleaner.models import Item
from gleaner.security import validate_safe_url
from gleaner.sources.base import FetchResult

logger = logging.getLogger(__name__)


def _extract_tag_value(tags: list[list[str]], tag_name: str) -> str | None:
    """Return the value of the first tag with this name, or None."""
    for t in tags:
        if len(t) >= 2 and t[0] == tag_name:
            return t[1]
    return None


def parse_nostr_event_to_item(event: dict[str, Any], source_id: str) -> Item | None:
    """Parse a Nostr event dictionary into a normalized Gleaner Item.

    Args:
        event: The event as decoded from the relay's JSON.
        source_id: The id of the source the item belongs to.

    Returns:
        The item (an article for kind 30023, else a post), or None when the event has no id.
    """
    event_id = str(event.get("id", "")).strip()
    if not event_id:
        return None

    kind = event.get("kind", 1)
    tags = event.get("tags", [])
    content = str(event.get("content", "")).strip()
    created_at_ts = event.get("created_at")

    dt = datetime.fromtimestamp(created_at_ts, tz=UTC) if created_at_ts else datetime.now(UTC)

    # 1. Title extraction
    title = _extract_tag_value(tags, "title") or _extract_tag_value(tags, "subject")
    if not title:
        # Fallback to first non-empty line of content
        first_line = content.split("\n", 1)[0].strip()
        first_line = re.sub(r"^[#*`\s-]+", "", first_line)
        title = first_line[:120] if first_line else "Nostr Event"

    # 2. Kind classification
    item_kind = "article" if kind == 30023 else "post"

    # 3. Canonical URL
    url = _extract_tag_value(tags, "canonical") or _extract_tag_value(tags, "url")
    if not url:
        url = f"https://njump.me/{event_id}"

    # 4. Author
    author = _extract_tag_value(tags, "author") or str(event.get("pubkey", ""))[:16]

    return Item(
        id=f"nostr-{event_id[:16]}",
        source_id=source_id,
        kind=item_kind,
        url=url,
        title=title,
        author=author,
        published_at=dt,
        fetched_at=datetime.now(UTC),
        lang="en",
        text=content,
        content_hash=None,
        signals={
            "nostr_kind": kind,
            "pubkey": event.get("pubkey", ""),
            "tags": [t[1] for t in tags if len(t) >= 2 and t[0] == "t"],
        },
        story_id=None,
        status="new",
    )


class NostrSource:
    """Source connector for Nostr relays (NIP-01, NIP-23).

    Attributes:
        type: Always ``"nostr"``.
    """

    type: str = "nostr"

    async def fetch(
        self,
        cfg: SourceConfig,
        since: datetime | None,
        client: httpx.AsyncClient,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> FetchResult:
        """Fetch technical articles and notes from a Nostr relay.

        The relay (``params['relay']`` or ``params['url']``) is an HTTP(S) gateway or a WebSocket
        relay. Any failure is logged and yields an empty result rather than an error.

        Args:
            cfg: The source's configuration; ``params`` may also set ``kinds`` and ``limit``.
            since: Only events created after this time, when given.
            client: The shared HTTP client, used for HTTP gateways.
            etag: Unused; relays have no conditional requests.
            last_modified: Unused; relays have no conditional requests.

        Returns:
            Up to ``limit`` items, with no caching state.

        Raises:
            ValueError: The relay URL is unsafe or targets a private or local address.
        """
        relay = cfg.params.get("relay") or cfg.params.get("url", "")
        if not relay:
            logger.warning("Nostr source %s has no 'relay' configured", cfg.id)
            return FetchResult()

        # a relay is fetched like any URL, with the same SSRF check (ws/wss checked as http/https)
        validate_safe_url(re.sub(r"^ws", "http", relay))

        kinds = cfg.params.get("kinds", [30023, 1])
        limit = int(cfg.params.get("limit", 20))
        since_ts = int(since.timestamp()) if since else None

        items: list[Item] = []

        # Case A: HTTP/HTTPS REST relay gateway (mirrors)
        if relay.startswith(("http://", "https://")):
            try:
                params: dict[str, Any] = {"limit": limit}
                if kinds:
                    params["kinds"] = ",".join(str(k) for k in kinds)
                if since_ts:
                    params["since"] = since_ts

                resp = await client.get(relay, params=params, timeout=10.0)
                if resp.status_code != 200:
                    logger.warning("Nostr HTTP relay %s returned status %d", relay, resp.status_code)
                    return FetchResult()

                data = resp.json()
                raw_events = data.get("events", data) if isinstance(data, dict) else data
                if isinstance(raw_events, list):
                    items.extend(
                        it
                        for ev in raw_events
                        if isinstance(ev, dict) and (it := parse_nostr_event_to_item(ev, cfg.id))
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Nostr HTTP fetch from %s failed: %s", relay, exc)
                return FetchResult()

        # Case B: WebSocket relay (native Nostr protocol)
        elif relay.startswith(("ws://", "wss://")):
            try:
                import websockets

                sub_id = f"gleaner-{cfg.id[:8]}"
                filter_obj: dict[str, Any] = {"kinds": kinds, "limit": limit}
                if since_ts:
                    filter_obj["since"] = since_ts
                req_msg = json.dumps(["REQ", sub_id, filter_obj])
                close_msg = json.dumps(["CLOSE", sub_id])

                async with websockets.connect(relay, open_timeout=5, close_timeout=5) as ws:
                    await ws.send(req_msg)
                    try:
                        while len(items) < limit:
                            msg_str = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            msg = json.loads(msg_str)
                            if not isinstance(msg, list) or len(msg) < 2:
                                continue
                            msg_type = msg[0]
                            if msg_type == "EVENT" and len(msg) >= 3 and isinstance(msg[2], dict):
                                if it := parse_nostr_event_to_item(msg[2], cfg.id):
                                    items.append(it)
                            elif msg_type == "EOSE":
                                break
                    except TimeoutError:
                        pass
                    finally:
                        await ws.send(close_msg)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Nostr WebSocket connection to %s failed: %s", relay, exc)
                return FetchResult()

        return FetchResult(items=items)
