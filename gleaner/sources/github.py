"""GitHub releases and repository connector."""

from __future__ import annotations

import os
import re
from datetime import datetime

import httpx

from gleaner.config import SourceConfig
from gleaner.models import Item
from gleaner.normalize import canonicalize_url, compute_content_hash, generate_item_id, strip_html
from gleaner.sources.base import FetchResult

# GitHub owner (alphanumerics and hyphens, no leading hyphen, <= 39 chars) / repository name.
_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9._-]{1,100}")


class GitHubSource:
    """Connector for one GitHub repository's releases.

    Attributes:
        type: Always ``"github"``.
    """

    type = "github"

    async def fetch(
        self,
        cfg: SourceConfig,
        since: datetime | None,
        client: httpx.AsyncClient,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> FetchResult:
        """Fetch the releases of the repository named in ``params['repo']``.

        Sends ``GITHUB_TOKEN`` as a bearer token when it is set.

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
            ValueError: ``repo`` is missing or not exactly ``owner/repo``.
            httpx.HTTPStatusError: GitHub answers with an error status.
        """
        repo = cfg.params.get("repo")
        if not repo:
            raise ValueError(f"GitHub source '{cfg.id}' missing required 'repo' param (e.g. 'owner/repo')")
        # The repo is interpolated into an authenticated API path, so it must be exactly owner/repo.
        if not isinstance(repo, str) or not _REPO_RE.fullmatch(repo) or repo.endswith(("/.", "/..")):
            raise ValueError(f"GitHub source '{cfg.id}' has invalid 'repo' param {repo!r} (expected 'owner/repo')")

        url = f"https://api.github.com/repos/{repo}/releases"
        headers: dict[str, str] = {
            "User-Agent": "DiscoveryFeedBot/1.0 (+https://github.com/langchain-ai/graphflow)",
            "Accept": "application/vnd.github+json",
        }
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        resp = await client.get(url, headers=headers)
        if resp.status_code == 304:
            return FetchResult(items=[], etag=etag, last_modified=last_modified)

        resp.raise_for_status()
        releases = resp.json()
        if not isinstance(releases, list):
            releases = []

        items: list[Item] = []
        for rel in releases:
            tag = rel.get("tag_name") or ""
            rel_name = strip_html(rel.get("name") or "") or tag or "Release"
            body = strip_html(rel.get("body") or "")
            html_url = rel.get("html_url") or ""
            rel_id = str(rel.get("id") or tag)

            pub_date = None
            pub_str = rel.get("published_at")
            if pub_str:
                try:
                    pub_date = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                except (AttributeError, ValueError):
                    pub_date = None

            author = None
            author_data = rel.get("author")
            if isinstance(author_data, dict):
                author = author_data.get("login")

            canon_url = canonicalize_url(html_url)
            item_id = generate_item_id(canon_url, cfg.id, rel_id)
            c_hash = compute_content_hash(body)

            items.append(
                Item(
                    id=item_id,
                    source_id=cfg.id,
                    kind="release",
                    url=canon_url,
                    title=f"{repo} {rel_name}",
                    author=author,
                    published_at=pub_date,
                    text=body,
                    content_hash=c_hash,
                    signals={"tag_name": tag, "repo": repo, "all_sources": [cfg.id]},
                )
            )

        return FetchResult(items=items, etag=resp.headers.get("etag"), last_modified=resp.headers.get("last-modified"))
