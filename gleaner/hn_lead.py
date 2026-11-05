"""HN lead: kept items that the user's sources published before anyone posted them to Hacker News.

Each collect asks HN's search (hn.algolia.com, official and free) whether recent kept items
were posted, by their URL. A lead is an item posted to HN after its source published it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

from gleaner.db import Database
from gleaner.models import Item

logger = logging.getLogger(__name__)

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
CHECK_WINDOW = timedelta(days=30)  # older items have had their chance to reach HN
RECHECK_AFTER = timedelta(days=1)
CHECKS_PER_TICK = 30


@dataclass(frozen=True)
class HNLead:
    """A kept item that reached Hacker News after its source published it.

    Attributes:
        item: The kept item.
        posted_at: When the item was first posted to HN.
        published_at: When the item's source published it.
        points: The HN post's points when it was checked.
        hn_id: The HN post's id.
    """

    item: Item
    posted_at: datetime
    published_at: datetime
    points: int
    hn_id: str

    @property
    def lead(self) -> timedelta:
        """How long the source was ahead of HN."""
        return self.posted_at - self.published_at

    @property
    def hn_url(self) -> str:
        """The HN discussion page."""
        return f"https://news.ycombinator.com/item?id={self.hn_id}"


def url_key(url: str | None) -> str | None:
    """What identifies the item's link on HN.

    The arXiv id, else the URL without scheme or www.

    Args:
        url: The item's URL.

    Returns:
        The key, or None for no URL or a bare homepage (it would match every post from that site).
    """
    if not url:
        return None
    if m := re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?!\d)", url):
        return m.group(1)
    key = url.split("://", 1)[-1].removeprefix("www.").rstrip("/")
    return key if "/" in key else None  # a bare homepage would match every post from that site


def _utc(dt: datetime) -> datetime:
    """Return the datetime, reading a naive one as UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


async def check_hn_sightings(db: Database, client: httpx.AsyncClient, now: datetime, limit: int = CHECKS_PER_TICK) -> int:
    """Ask HN about recent kept items not checked today; returns how many were checked.

    The first HN post of the item is stored (or none found). A failed search skips the item
    until the next tick.

    Args:
        db: The database holding the items and their sightings.
        client: The shared HTTP client.
        now: The current time.
        limit: The most items to check in this call.

    Returns:
        How many items were checked.
    """
    # keyless (homepage) links are left out before the limit: they would hold its first slots forever
    candidates = [(it, key) for it in db.hn_check_candidates(checked_before=now - RECHECK_AFTER)
                  if it.published_at and _utc(it.published_at) >= now - CHECK_WINDOW and (key := url_key(it.url))]
    checked = 0
    for item, key in candidates[:limit]:
        try:
            resp = await client.get(HN_SEARCH_URL, params={"query": key, "restrictSearchableAttributes": "url", "tags": "story"})
            resp.raise_for_status()
            # the same key, not one containing it: arXiv 2610.12345 is not 2610.123456
            hits = [h for h in resp.json().get("hits", []) if url_key(h.get("url")) == key and h.get("created_at")]
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("HN search failed for %s: %s", item.id, exc)
            continue
        first = min(hits, key=lambda h: h["created_at"], default=None)
        db.upsert_hn_sighting(
            item.id,
            hn_id=str(first["objectID"]) if first else None,
            posted_at=datetime.fromisoformat(first["created_at"].replace("Z", "+00:00")) if first else None,
            points=int(first.get("points") or 0) if first else None,
            checked_at=now,
        )
        checked += 1
    return checked


def hn_leads(db: Database, limit: int = 6) -> list[HNLead]:
    """Items posted to HN after their source published them.

    Most recently posted first.

    The lead must beat the source's polling interval: gleaner reads a 3-hourly blog up to
    3 hours late, so a post that reached HN 8 minutes after publishing was not ahead for the user.

    Args:
        db: The database holding the items and their sightings.
        limit: The most leads to return.

    Returns:
        The leads, most recently posted first, one per HN post.
    """
    leads = []
    for row in db.list_hn_sightings():
        item = db.get_item(row["item_id"])
        if item is None or item.published_at is None:
            continue
        lead = HNLead(item=item, posted_at=_utc(row["posted_at"]), published_at=_utc(item.published_at),
                      points=row["points"], hn_id=row["hn_id"])
        src = db.get_source(item.source_id) if item.source_id else None
        if lead.lead > timedelta(seconds=src.cadence_s if src else 0):
            leads.append(lead)
    # one HN post can match items from several websites: list it once, by the earliest of them
    best: dict[str, HNLead] = {}
    for lead in leads:
        if lead.hn_id not in best or lead.lead > best[lead.hn_id].lead:
            best[lead.hn_id] = lead
    return sorted(best.values(), key=lambda lead: lead.posted_at, reverse=True)[:limit]
