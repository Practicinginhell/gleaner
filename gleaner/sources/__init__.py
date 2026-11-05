"""Source connectors registry and dispatch."""

from __future__ import annotations

from gleaner.sources.arxiv import ArxivSource
from gleaner.sources.base import FetchResult, Source
from gleaner.sources.github import GitHubSource
from gleaner.sources.hn import HNSource
from gleaner.sources.nostr import NostrSource
from gleaner.sources.rss import RSSSource

_CONNECTORS: dict[str, Source] = {
    "rss": RSSSource(),
    "arxiv": ArxivSource(),
    "hn": HNSource(),
    "github": GitHubSource(),
    "nostr": NostrSource(),
}


def get_connector(source_type: str) -> Source:
    """Return the shared connector for a source type.

    Args:
        source_type: The ``type`` of a configured source, such as ``"rss"``.

    Returns:
        The registered connector instance.

    Raises:
        ValueError: No connector is registered for ``source_type``.
    """
    if source_type not in _CONNECTORS:
        raise ValueError(f"No connector registered for source type '{source_type}'")
    return _CONNECTORS[source_type]


__all__ = [
    "ArxivSource",
    "FetchResult",
    "GitHubSource",
    "HNSource",
    "NostrSource",
    "RSSSource",
    "Source",
    "get_connector",
]
