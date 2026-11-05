"""Base protocols and types for source connectors."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, Field

from gleaner.config import SourceConfig
from gleaner.models import Item


class FetchResult(BaseModel):
    """One fetch's items plus the state to send with the next fetch.

    Attributes:
        items: The normalized items fetched.
        etag: The response's ETag, for the next conditional request.
        last_modified: The response's Last-Modified value, for the next conditional request.
        next_cursor: A paging cursor for the next fetch, when the source has one.
    """

    items: list[Item] = Field(default_factory=list)
    etag: str | None = None
    last_modified: str | None = None
    next_cursor: str | None = None


@runtime_checkable
class Source(Protocol):
    """A connector that fetches items from one type of source.

    Attributes:
        type: The source type this connector serves, as written in ``sources.yaml``.
    """

    type: str

    async def fetch(
        self,
        cfg: SourceConfig,
        since: datetime | None,
        client: httpx.AsyncClient,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> FetchResult:
        """Fetch items from the source politely and incrementally.

        Args:
            cfg: The source's configuration.
            since: When the source was last fetched, or None for a first fetch.
            client: The shared HTTP client.
            etag: The ETag from the previous fetch, for a conditional request.
            last_modified: The Last-Modified value from the previous fetch, for a conditional request.

        Returns:
            The fetched items and the caching state for the next fetch.
        """
        ...
