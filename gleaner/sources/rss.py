"""RSS and Atom feed connector."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

import httpx

from gleaner.config import SourceConfig
from gleaner.models import Item
from gleaner.normalize import canonicalize_url, compute_content_hash, generate_item_id, strip_html
from gleaner.security import validate_safe_url
from gleaner.sources.base import FetchResult

RSS1 = "{http://purl.org/rss/1.0/}"  # RSS 1.0 (RDF) puts its elements in this namespace


def _find_child(node: ET.Element, tags: tuple[str, ...]) -> ET.Element | None:
    """Return the node's first child matching any of the tags, tried in order."""
    for tag in tags:
        child = node.find(tag)
        if child is not None:
            return child
    return None


class RSSSource:
    """Connector for RSS and Atom feeds.

    Attributes:
        type: Always ``"rss"``.
    """

    type = "rss"

    async def fetch(
        self,
        cfg: SourceConfig,
        since: datetime | None,
        client: httpx.AsyncClient,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> FetchResult:
        """Fetch and parse the feed at ``params['url']``.

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
            ValueError: ``url`` is missing or unsafe, or the feed is not valid XML.
            httpx.HTTPStatusError: The server answers with an error status.
        """
        url = cfg.params.get("url")
        if not url:
            raise ValueError(f"RSS source '{cfg.id}' missing required 'url' param")
        validate_safe_url(url)

        headers: dict[str, str] = {
            "User-Agent": "DiscoveryFeedBot/1.0 (+https://github.com/langchain-ai/graphflow)"
        }
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        resp = await client.get(url, headers=headers)
        if resp.status_code == 304:
            return FetchResult(items=[], etag=etag, last_modified=last_modified)

        resp.raise_for_status()

        new_etag = resp.headers.get("etag")
        new_last_modified = resp.headers.get("last-modified")

        items = self._parse_feed_xml(resp.text, cfg.id, base_url=url)
        return FetchResult(items=items, etag=new_etag, last_modified=new_last_modified)

    def _parse_feed_xml(self, text: str, source_id: str, base_url: str = "") -> list[Item]:
        """Parse RSS/Atom items into article items.

        Args:
            text: The feed XML.
            source_id: The id of the source the items belong to.
            base_url: The feed's URL; relative links resolve against it.

        Returns:
            One item per RSS item or Atom entry.

        Raises:
            ValueError: The text is not valid XML.
        """
        items: list[Item] = []
        try:
            root = ET.fromstring(text)  # noqa: S314 - expat limits entity expansion; ElementTree resolves no external entities
        except ET.ParseError as e:
            raise ValueError(f"RSS source '{source_id}' returned invalid XML: {e}") from e

        # RSS 2.0 / 0.9 / 1.0 (channel -> item)
        item_nodes = root.findall(".//item") or root.findall(f".//{RSS1}item")
        if not item_nodes:
            # Atom entries
            item_nodes = root.findall(".//{http://www.w3.org/2005/Atom}entry") or root.findall(".//entry")

        for node in item_nodes:
            title_node = _find_child(node, ("title", "{http://www.w3.org/2005/Atom}title", f"{RSS1}title"))
            title = strip_html(title_node.text) if title_node is not None else "Untitled"

            # Link
            link = ""
            atom_links = node.findall("{http://www.w3.org/2005/Atom}link")
            # an Atom entry can list rel="replies" or "enclosure" first: the article is rel="alternate" (or no rel)
            link_node = next((n for n in atom_links if n.attrib.get("rel", "alternate") == "alternate"), None)
            if link_node is None:
                link_node = _find_child(node, ("link", f"{RSS1}link", "{http://www.w3.org/2005/Atom}link"))
            if link_node is not None:
                link = link_node.attrib.get("href") or link_node.text or ""
            link = link.strip()
            if link:
                link = urljoin(base_url, link)

            # Native ID: <guid> (RSS) or <id> (Atom), so link-less items don't collide
            guid_node = _find_child(node, ("guid", "{http://www.w3.org/2005/Atom}id"))
            native_id = (guid_node.text or "").strip() if guid_node is not None else ""

            # Description / content
            desc_node = _find_child(
                node,
                (
                    "description",
                    f"{RSS1}description",
                    "{http://www.w3.org/2005/Atom}summary",
                    "{http://www.w3.org/2005/Atom}content",
                    "{http://purl.org/rss/1.0/modules/content/}encoded",
                ),
            )
            raw_text = desc_node.text or "" if desc_node is not None else ""
            clean_text = strip_html(raw_text)

            # Date
            pub_date = None
            date_node = _find_child(
                node,
                (
                    "pubDate",
                    "{http://www.w3.org/2005/Atom}published",
                    "{http://www.w3.org/2005/Atom}updated",
                    "{http://purl.org/dc/elements/1.1/}date",
                ),
            )
            if date_node is not None and date_node.text:
                try:
                    pub_date = parsedate_to_datetime(date_node.text.strip())
                    if pub_date.tzinfo is None:
                        pub_date = pub_date.replace(tzinfo=UTC)
                except (TypeError, ValueError):
                    try:
                        pub_date = datetime.fromisoformat(date_node.text.strip())
                        if pub_date.tzinfo is None:  # "2026-10-30": read as UTC like the RFC 822 dates
                            pub_date = pub_date.replace(tzinfo=UTC)
                    except ValueError:
                        pub_date = None

            # Author
            author_node = _find_child(node, ("author", "{http://www.w3.org/2005/Atom}author"))
            author = None
            if author_node is not None:
                name_node = author_node.find("{http://www.w3.org/2005/Atom}name")
                author = name_node.text if name_node is not None else author_node.text

            canon_url = canonicalize_url(link)
            item_id = generate_item_id(canon_url, source_id, native_id or title)
            c_hash = compute_content_hash(clean_text)

            items.append(
                Item(
                    id=item_id,
                    source_id=source_id,
                    kind="article",
                    url=canon_url,
                    title=title,
                    author=author,
                    published_at=pub_date,
                    text=clean_text,
                    content_hash=c_hash,
                    signals={"all_sources": [source_id]},
                )
            )

        return items
