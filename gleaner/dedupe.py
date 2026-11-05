"""Exact and MinHash near-duplicate deduplication for items."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from gleaner.models import Item
from gleaner.normalize import canonicalize_url, compute_content_hash, generate_item_id

if TYPE_CHECKING:
    from gleaner.db import Database

# 128 seed pairs for universal hashing
_HASH_PRIME = (1 << 61) - 1
_SEEDS: list[tuple[int, int]] = [
    ((i * 10007 + 3) % _HASH_PRIME, (i * 20011 + 7) % _HASH_PRIME) for i in range(1, 129)
]


class MinHashDeduper:
    """MinHash LSH implementation for detecting near-duplicate text.

    Attributes:
        threshold: The estimated Jaccard similarity at which two texts are near-duplicates.
        num_perm: The signature length, at most the 128 available seeds.
        shingle_len: The number of words in each shingle.
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, shingle_len: int = 5) -> None:
        """Initialize the deduper.

        Args:
            threshold: The estimated Jaccard similarity at which two texts are near-duplicates.
            num_perm: The signature length; capped at the 128 available seeds.
            shingle_len: The number of words in each shingle.
        """
        self.threshold = threshold
        self.num_perm = min(num_perm, len(_SEEDS))
        self.shingle_len = shingle_len

    def _get_shingles(self, text: str) -> set[str]:
        """Return the text's word shingles.

        A text shorter than one shingle is a single shingle.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < self.shingle_len:
            return {" ".join(words)} if words else set()
        return {
            " ".join(words[i : i + self.shingle_len])
            for i in range(len(words) - self.shingle_len + 1)
        }

    def compute_signature(self, text: str) -> list[int] | None:
        """MinHash signature of the text.

        None when it has no words, since there is nothing to compare.

        Args:
            text: The text to sign.

        Returns:
            One minimum hash per permutation, or None.
        """
        shingles = self._get_shingles(text)
        if not shingles:
            return None

        shingle_hashes = [
            int.from_bytes(hashlib.md5(s.encode("utf-8"), usedforsecurity=False).digest()[:8], "little")
            for s in shingles
        ]

        sig = []
        for i in range(self.num_perm):
            a, b = _SEEDS[i]
            min_val = min((a * h + b) % _HASH_PRIME for h in shingle_hashes)
            sig.append(min_val)
        return sig

    def jaccard_similarity(self, sig1: list[int] | None, sig2: list[int] | None) -> float:
        """Estimate the Jaccard similarity of two texts from their signatures.

        Args:
            sig1: The first signature, or None.
            sig2: The second signature, or None.

        Returns:
            The share of matching positions; 0.0 when either signature is missing.
        """
        if not sig1 or not sig2:
            return 0.0
        matches = sum(1 for x, y in zip(sig1, sig2, strict=True) if x == y)
        return matches / len(sig1)

    def is_near_duplicate(self, text1: str, text2: str) -> bool:
        """Whether two texts reach the near-duplicate threshold; blank texts never do.

        Args:
            text1: The first text.
            text2: The second text.

        Returns:
            True when the estimated similarity is at least ``threshold``.
        """
        if not text1.strip() or not text2.strip():
            return False
        sig1 = self.compute_signature(text1)
        sig2 = self.compute_signature(text2)
        return self.jaccard_similarity(sig1, sig2) >= self.threshold


def dedupe_items(items: Sequence[Item], minhash_threshold: float = 0.8) -> list[Item]:
    """Deduplicate items by canonical URL, content hash and MinHash.

    URL and hash matches are exact; MinHash catches near-duplicates.

    Items are canonicalized in place (URL, ID, content hash), and each duplicate's sources and
    signals are merged into the item it duplicates.

    Args:
        items: The fetched items, in order; the first of each duplicate group survives.
        minhash_threshold: The estimated Jaccard similarity at which two texts are near-duplicates.

    Returns:
        The surviving items, in input order.
    """
    deduper = MinHashDeduper(threshold=minhash_threshold)
    by_canonical_url: dict[str, Item] = {}
    by_content_hash: dict[str, Item] = {}
    survivors: list[Item] = []
    signatures: list[list[int] | None] = []

    for item in items:
        canon_url = canonicalize_url(item.url)
        content_h = item.content_hash or compute_content_hash(item.text)

        # A canonical URL decides the ID; without one keep the connector's (native-ID based) ID.
        # A URL that fails canonicalization is unsafe or unusable, so it is cleared, never kept.
        if canon_url:
            item.id = generate_item_id(canon_url, item.source_id or "unknown")
        item.url = canon_url
        item.content_hash = content_h

        # Check exact canonical URL match
        if canon_url and canon_url in by_canonical_url:
            existing = by_canonical_url[canon_url]
            _merge_item_sources(existing, item)
            continue

        # Check exact content hash match
        if content_h in by_content_hash and content_h != hashlib.sha256(b"").hexdigest():
            existing = by_content_hash[content_h]
            _merge_item_sources(existing, item)
            continue

        # Check near-duplicate MinHash match against survivors (items without text are never near-dups)
        sig = deduper.compute_signature(item.text)
        is_near = False
        for existing, exist_sig in zip(survivors, signatures, strict=True):
            if sig is not None and exist_sig is not None and deduper.jaccard_similarity(sig, exist_sig) >= minhash_threshold:
                _merge_item_sources(existing, item)
                is_near = True
                break

        if is_near:
            continue

        # New unique item
        if not item.signals:
            item.signals = {}
        if item.source_id and "all_sources" not in item.signals:
            item.signals["all_sources"] = [item.source_id]

        if canon_url:
            by_canonical_url[canon_url] = item
        by_content_hash[content_h] = item
        survivors.append(item)
        signatures.append(sig)

    return survivors


def adopt_stored(db: Database, items: Sequence[Item]) -> list[Item]:
    """Keep the stored content when another source already has the item.

    The new source is merged in.

    Ids hash the canonical URL, so the same link from a blog and from Hacker News is one row.
    Stored as is, the later fetch would replace the article with a bare HN title.

    Args:
        db: The database holding the stored items.
        items: The newly fetched items.

    Returns:
        The items, each replaced by its stored version when another source stored it first.
    """
    adopted = []
    for item in items:
        stored = db.get_item(item.id)
        if stored is not None and stored.source_id != item.source_id:
            stored.status = item.status  # processed like any re-fetch
            _merge_item_sources(stored, item)
            item = stored
        adopted.append(item)
    return adopted


def _merge_item_sources(target: Item, incoming: Item) -> None:
    """Merge an incoming duplicate item's metadata and sources into the target item.

    Args:
        target: The surviving item; updated in place.
        incoming: The duplicate; its numeric signals win where they are higher.
    """
    if not target.signals:
        target.signals = {}
    sources = set(target.signals.get("all_sources", []))
    if target.source_id:
        sources.add(target.source_id)
    if incoming.source_id:
        sources.add(incoming.source_id)
    target.signals["all_sources"] = sorted(sources)

    # Merge signals like points/comments/stars if incoming has higher
    for k, v in (incoming.signals or {}).items():
        if k == "all_sources":
            continue
        if k not in target.signals or (isinstance(v, (int, float)) and v > target.signals.get(k, 0)):
            target.signals[k] = v
