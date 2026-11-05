"""Hacker News Algolia Search API connector."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import quote_plus

import httpx

from gleaner.config import SourceConfig
from gleaner.models import Item
from gleaner.normalize import canonicalize_url, compute_content_hash, generate_item_id, strip_html
from gleaner.sources.base import FetchResult


class HNSource:
    """Connector for Hacker News stories through the Algolia search API.

    Attributes:
        type: Always ``"hn"``.
    """

    type = "hn"

    async def fetch(
        self,
        cfg: SourceConfig,
        since: datetime | None,
        client: httpx.AsyncClient,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> FetchResult:
        """Fetch 30 stories: the front page, or the newest matching ``params['query']``.

        Args:
            cfg: The source's configuration; its ``params`` select what to fetch.
            since: Ignored; the conditional-request headers make the fetch incremental.
            client: The shared HTTP client.
            etag: The ETag from the previous fetch, sent as ``If-None-Match``.
            last_modified: The Last-Modified value from the previous fetch, sent as ``If-Modified-Since``.

        Returns:
            The parsed items with the response's caching headers; no items and the
            given headers when the server answers 304 Not Modified.

        Raises:
            httpx.HTTPStatusError: The API answers with an error status.
        """
        query = cfg.params.get("query", "front_page")
        if query == "front_page":
            url = "https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage=30"
        else:
            url = f"https://hn.algolia.com/api/v1/search_by_date?query={quote_plus(query)}&tags=story&hitsPerPage=30"

        headers = {"User-Agent": "DiscoveryFeedBot/1.0 (+https://github.com/langchain-ai/graphflow)"}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        resp = await client.get(url, headers=headers)
        if resp.status_code == 304:
            return FetchResult(items=[], etag=etag, last_modified=last_modified)

        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits") or []

        items: list[Item] = []
        for hit in hits:
            obj_id = str(hit.get("objectID") or "")
            title = strip_html(hit.get("title") or "") or "Untitled HN Post"
            external_url = hit.get("url")
            hn_url = f"https://news.ycombinator.com/item?id={obj_id}"
            url = external_url or hn_url
            author = hit.get("author")
            points = int(hit.get("points") or 0)
            comments = int(hit.get("num_comments") or 0)

            pub_date = None
            created_at = hit.get("created_at")
            if created_at:
                try:
                    pub_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except (AttributeError, ValueError):
                    pub_date = None

            canon_url = canonicalize_url(url)
            item_id = generate_item_id(canon_url, cfg.id, obj_id)
            c_hash = compute_content_hash(title)

            items.append(
                Item(
                    id=item_id,
                    source_id=cfg.id,
                    kind="post",
                    url=canon_url,
                    title=title,
                    author=author,
                    published_at=pub_date,
                    text=title,
                    content_hash=c_hash,
                    signals={
                        "points": points,
                        "comments": comments,
                        "hn_id": obj_id,
                        "hn_url": hn_url,
                        "all_sources": [cfg.id],
                    },
                )
            )

        return FetchResult(items=items, etag=resp.headers.get("etag"), last_modified=resp.headers.get("last-modified"))
