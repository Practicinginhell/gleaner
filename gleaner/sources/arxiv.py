"""arXiv export API connector."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import quote_plus

import httpx

from gleaner.config import SourceConfig
from gleaner.models import Item
from gleaner.normalize import canonicalize_url, compute_content_hash, generate_item_id, strip_html
from gleaner.sources.base import FetchResult

ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArxivSource:
    """Connector for arXiv papers matching categories and a search query.

    Attributes:
        type: Always ``"arxiv"``.
    """

    type = "arxiv"

    async def fetch(
        self,
        cfg: SourceConfig,
        since: datetime | None,
        client: httpx.AsyncClient,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> FetchResult:
        """Fetch the 25 newest papers matching the source's categories and query.

        ``params`` may hold ``categories`` (ORed) and ``query``; with neither, the search is ``all:agent``.

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
            httpx.HTTPStatusError: arXiv answers with an error status.
            ValueError: arXiv returns invalid XML.
        """
        categories = cfg.params.get("categories") or []
        query_text = cfg.params.get("query") or ""

        query_parts = []
        if categories:
            cat_query = " OR ".join(f"cat:{c}" for c in categories)
            query_parts.append(f"({cat_query})")
        if query_text:
            query_parts.append(f"all:{query_text}")

        full_query = " AND ".join(query_parts) if query_parts else "all:agent"
        url = f"https://export.arxiv.org/api/query?search_query={quote_plus(full_query)}&max_results=25&sortBy=submittedDate&sortOrder=descending"

        headers = {"User-Agent": "DiscoveryFeedBot/1.0 (+https://github.com/langchain-ai/graphflow)"}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        resp = await client.get(url, headers=headers)
        if resp.status_code == 304:
            return FetchResult(items=[], etag=etag, last_modified=last_modified)

        resp.raise_for_status()

        items = self._parse_arxiv_atom(resp.text, cfg.id)
        return FetchResult(items=items, etag=resp.headers.get("etag"), last_modified=resp.headers.get("last-modified"))

    def _parse_arxiv_atom(self, text: str, source_id: str) -> list[Item]:
        """Parse an arXiv Atom response into paper items.

        Args:
            text: The Atom XML.
            source_id: The id of the source the items belong to.

        Returns:
            One item per entry, linked to its abstract page, with the PDF link in its signals.

        Raises:
            ValueError: The text is not valid XML.
        """
        items: list[Item] = []
        try:
            root = ET.fromstring(text)  # noqa: S314 - expat limits entity expansion; ElementTree resolves no external entities
        except ET.ParseError as e:
            raise ValueError(f"arXiv source '{source_id}' returned invalid XML: {e}") from e

        for entry in root.findall(f".//{ATOM_NS}entry"):
            id_node = entry.find(f"{ATOM_NS}id")
            raw_id = (id_node.text or "").strip() if id_node is not None else ""

            title_node = entry.find(f"{ATOM_NS}title")
            title = " ".join(strip_html(title_node.text).split()) if title_node is not None else "Untitled Paper"

            summary_node = entry.find(f"{ATOM_NS}summary")
            summary = strip_html((summary_node.text or "").strip()) if summary_node is not None else ""

            pub_date = None
            pub_node = entry.find(f"{ATOM_NS}published")
            if pub_node is not None and pub_node.text:
                try:
                    pub_date = datetime.fromisoformat(pub_node.text.strip().replace("Z", "+00:00"))
                except ValueError:
                    pub_date = None

            # Authors
            authors = []
            for author_node in entry.findall(f"{ATOM_NS}author"):
                name_node = author_node.find(f"{ATOM_NS}name")
                if name_node is not None and name_node.text:
                    authors.append(name_node.text.strip())
            author_str = ", ".join(authors) if authors else None

            # Links
            alt_link = raw_id
            pdf_link = None
            for link_node in entry.findall(f"{ATOM_NS}link"):
                rel = link_node.attrib.get("rel")
                href = link_node.attrib.get("href", "")
                title_attr = link_node.attrib.get("title", "")
                if rel == "alternate" and href:
                    alt_link = href
                elif title_attr == "pdf" or "pdf" in href:
                    pdf_link = href

            canon_url = canonicalize_url(alt_link)
            item_id = generate_item_id(canon_url, source_id, raw_id)
            c_hash = compute_content_hash(summary)

            items.append(
                Item(
                    id=item_id,
                    source_id=source_id,
                    kind="paper",
                    url=canon_url,
                    title=title,
                    author=author_str,
                    published_at=pub_date,
                    text=summary,
                    content_hash=c_hash,
                    signals={"pdf_url": pdf_link, "all_sources": [source_id]},
                )
            )

        return items
