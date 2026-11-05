"""URL canonicalization, HTML sanitization, and text normalization."""

from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# Query params to drop for tracking / click counting
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ref",
    "ref_src",
    "source",
    "mc_cid",
    "mc_eid",
    "_hsenc",
    "_hsmi",
}


def canonicalize_url(url: str | None) -> str | None:
    """Canonicalize a URL.

    Lowercases the host and strips fragments, tracking params and the trailing slash.

    Args:
        url: The URL to canonicalize.

    Returns:
        The canonical URL, with query parameters sorted. None for anything that is not an
        http(s) URL with a host (including unparseable input), so callers never keep an
        unsafe original.
    """
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    if not url:
        return None

    try:
        parsed = urlparse(url)
        _ = parsed.port  # raises ValueError on an out-of-range or non-numeric port
    except ValueError:
        return None

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https") or not parsed.hostname:
        return None
    netloc = parsed.netloc.lower()

    # Clean query parameters
    query_params = parse_qsl(parsed.query, keep_blank_values=False)
    filtered_params = [
        (k, v) for k, v in query_params if k.lower() not in TRACKING_PARAMS and not k.lower().startswith("utm_")
    ]
    # Sort for deterministic URL
    filtered_params.sort(key=lambda x: x[0])
    clean_query = urlencode(filtered_params)

    # Standardize path: drop a trailing slash, but the root is always "/"
    path = parsed.path.rstrip("/") or "/"

    # Reassemble without fragment
    clean_url = urlunparse((scheme, netloc, path, parsed.params, clean_query, ""))
    return clean_url


_HIDDEN_TAGS = frozenset({"script", "style", "template"})
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)
_BLOCK_TAGS = frozenset({"p", "div", "br", "hr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"})
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE)


def _is_hidden(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
    """Whether the element is invisible.

    Script, style and template elements are, and so is one hidden by an attribute or inline style.
    """
    if tag in _HIDDEN_TAGS:
        return True
    for name, value in attrs:
        if name == "hidden":
            return True
        if name == "aria-hidden" and (value or "").strip().lower() == "true":
            return True
        if name == "style" and value and _HIDDEN_STYLE.search(value):
            return True
    return False


class _TextExtractor(HTMLParser):
    """Collects visible text; drops whole hidden subtrees however deeply they nest.

    Attributes:
        parts: The visible text fragments and separators, in document order.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_stack: list[str] = []  # open tags inside (and including) the outermost hidden element

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._hidden_stack or _is_hidden(tag, attrs):
            if tag not in _VOID_TAGS:
                self._hidden_stack.append(tag)
            return
        self.parts.append("\n" if tag in _BLOCK_TAGS else " ")

    def handle_endtag(self, tag: str) -> None:
        if self._hidden_stack:
            # Close up to the matching open tag; a stray closing tag is ignored.
            if tag in self._hidden_stack:
                while self._hidden_stack.pop() != tag:
                    pass
            return
        self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._hidden_stack:
            self.parts.append(data)


def strip_html(raw_html: str | None) -> str:
    """Convert HTML to clean plain text.

    Comments, script/style and hidden elements are dropped.

    Hidden means a `hidden` attribute, `aria-hidden="true"`, or an inline `display:none` /
    `visibility:hidden` style; everything nested inside such an element is dropped.

    Args:
        raw_html: The HTML to convert.

    Returns:
        The text with whitespace collapsed and blank lines dropped; "" for empty or non-string input.
    """
    if not raw_html or not isinstance(raw_html, str):
        return ""

    parser = _TextExtractor()
    parser.feed(raw_html)
    parser.close()
    text = "".join(parser.parts)

    # Normalize whitespace
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text.strip()


def compute_content_hash(text: str | None) -> str:
    """Compute sha256 hash of normalized text for deduplication.

    Args:
        text: The text; case and whitespace differences don't change the hash.

    Returns:
        The hex digest; empty or None text hashes as the empty string.
    """
    if not text:
        return hashlib.sha256(b"").hexdigest()
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def generate_item_id(canonical_url: str | None, source_id: str, native_id: str | None = None) -> str:
    """Generate a deterministic sha256 ID for an item.

    It is based on the canonical URL, or on source+native_id.

    Args:
        canonical_url: The item's canonical URL; when set, it alone decides the ID.
        source_id: The id of the item's source, used when there is no URL.
        native_id: The source's own id for the item, used when there is no URL.

    Returns:
        The hex digest.
    """
    if canonical_url:
        return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
    ident = f"{source_id}:{native_id or 'none'}"
    return hashlib.sha256(ident.encode("utf-8")).hexdigest()


def short_summary(text: str, limit: int = 200) -> str:
    """Shorten text for a card at a word boundary, with an ellipsis when it was cut.

    Args:
        text: The item's text.
        limit: The most characters to keep before the ellipsis.

    Returns:
        The text, or its first whole words up to ``limit`` characters followed by "…".
    """
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return cut + "…"
