"""Find new websites for a question: Muse Spark searches the web, gleaner checks every feed it names.

A question with no quotes is a dead end unless the user knows which websites to add.
Muse Spark (Responses API with its built-in `web_search` tool) looks for RSS or Atom feeds
that publish about the question. A model can name a feed that doesn't exist, so each one is
fetched with the same connector and the same rules a collect uses (SSRF check, DNS lookups off
the event loop) and kept only if it parses and has items. One redirect is followed and the feed
is offered under its final URL, since a collect doesn't follow redirects.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from gleaner.config import SourceConfig
from gleaner.llama import META_MODEL_API_URL
from gleaner.security import is_safe_external_url
from gleaner.sources.rss import RSSSource

RESPONSES_URL = f"{META_MODEL_API_URL}/responses"
SEARCH_MODEL = "muse-spark-1.3"
MAX_FEEDS = 5

PROMPT = """Use web search to find RSS or Atom feeds that regularly publish about this question:
{question}

Prefer blogs, research groups, project release feeds (for a GitHub project use
https://github.com/OWNER/REPO/releases.atom) and newsletters. Give the feed URL itself, not the home page.
Reply with only a JSON array of up to 6 objects: {{"title": "...", "feed_url": "https://...", "why": "one short sentence"}}"""


class DiscoverError(Exception):
    """A failure to find feeds, worded for the person who asked."""


def _output_text(body: dict[str, Any]) -> str:
    """Join the text blocks of a Responses API reply."""
    return "".join(
        block.get("text", "")
        for item in body.get("output", [])
        if item.get("type") == "message"
        for block in item.get("content", [])
        if block.get("type") == "output_text"
    )


def parse_candidates(text: str) -> list[dict[str, str]]:
    """Read the model's JSON array of feeds, tolerating prose or code fences around it.

    Args:
        text: The model's reply.

    Returns:
        Up to 6 `{"title", "url", "why"}` entries with http(s) URLs, without duplicates. The
        network safety check (`validate_safe_url`) runs when `check_feeds` fetches them.
    """
    # the first "[" that starts a JSON list of objects: replies also carry citations like [1]
    decoder = json.JSONDecoder()
    raw: Any = []
    for i, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and any(isinstance(v, dict) for v in value):
            raw = value
            break
    seen: set[str] = set()
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("feed_url") or entry.get("url") or "").strip()
        parts = urlparse(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or url in seen:
            continue
        seen.add(url)
        title = str(entry.get("title") or url)[:120]
        out.append({"title": title, "url": url, "why": str(entry.get("why") or "")[:240]})
    return out[:6]


def search_feeds(question: str, api_key: str, *, client: httpx.Client | None = None) -> list[dict[str, str]]:
    """Ask Muse Spark, with web search, for feeds about a question.

    Args:
        question: The standing question.
        api_key: The Meta Model API key.
        client: HTTP client to use; tests pass one with a mock transport.

    Returns:
        The candidate feeds (see `parse_candidates`), not yet checked.

    Raises:
        DiscoverError: Meta refused the request or couldn't be reached.
    """
    body = {
        "model": SEARCH_MODEL,
        "input": PROMPT.format(question=question),
        "tools": [{"type": "web_search"}],
        "max_output_tokens": 8192,
    }
    http = client or httpx.Client(timeout=90)
    try:
        res = http.post(RESPONSES_URL, headers={"Authorization": f"Bearer {api_key}"}, json=body)
    except httpx.HTTPError as exc:
        raise DiscoverError("Couldn't reach Meta to search the web.") from exc
    finally:
        if client is None:
            http.close()
    if res.status_code == 401:
        raise DiscoverError("Meta rejected MODEL_API_KEY.")
    if res.status_code != 200:
        raise DiscoverError(f"Muse Spark web search failed (HTTP {res.status_code}).")
    try:
        return parse_candidates(_output_text(res.json()))
    except (ValueError, TypeError, AttributeError) as exc:
        raise DiscoverError("Muse Spark sent an unexpected reply. Try again.") from exc


async def check_feeds(candidates: list[dict[str, str]], client: httpx.AsyncClient) -> list[dict[str, str]]:
    """Keep the candidates that are real feeds today, read the way a collect reads them.

    Args:
        candidates: Entries from `search_feeds`.
        client: The HTTP client. Like a collect's, it doesn't follow redirects itself.

    Returns:
        Up to `MAX_FEEDS` working feeds, each with its newest item's title as `latest`. A feed
        that redirects once to a safe URL is kept under that final URL.
    """

    async def probe(cand: dict[str, str]) -> dict[str, str] | None:
        url = cand["url"]
        for _ in range(2):  # the URL, then at most one redirect (http to https, apex to www)
            # the SSRF check resolves DNS: off the event loop, so a dead host can't stall the server
            if not await asyncio.to_thread(is_safe_external_url, url):
                return None
            cfg = SourceConfig(id="probe", type="rss", params={"url": url})
            try:
                result = await RSSSource().fetch(cfg, None, client)
            except httpx.HTTPStatusError as exc:
                location = exc.response.headers.get("location")
                if not (exc.response.is_redirect and location):
                    return None
                url = urljoin(url, location)
                continue
            except Exception:  # noqa: BLE001 - any failure means "not a working feed", which is the answer
                return None
            if not result.items:
                return None
            # offered under its final address, so the collect (which doesn't follow redirects) can read it
            return {**cand, "url": url, "latest": result.items[0].title}
        return None

    found = await asyncio.gather(*(probe(c) for c in candidates))
    seen: set[str] = set()
    kept = []
    for feed in found:
        if feed is not None and feed["url"] not in seen:
            seen.add(feed["url"])
            kept.append(feed)
    return kept[:MAX_FEEDS]
