"""Relevance filtering cascade: Stage A (rules) and Stage B (embedding gate)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from gleaner.embed import cosine_similarity
from gleaner.models import Item

CLICKBAIT_PHRASES = (
    "you won't believe",
    "you will not believe",
    "one weird trick",
    "shocking truth",
    "secret methods",
    "will blow your mind",
    "what happens next",
    "doctors hate him",
    "make money fast",
)


def filter_rules(
    item: Item,
    exclusions: Sequence[str] = (),
    min_article_length: int = 200,
) -> tuple[bool, str]:
    """Stage A: free rule-based filter checking exclusions, spam/clickbait, and length.

    Args:
        item: The item to check (title and text).
        exclusions: Keywords that drop the item when found as a whole word, case-insensitive.
        min_article_length: Minimum text length, in characters, for article and news items.

    Returns:
        ``(passed, reason)``: whether the item passed, and why it was dropped (or that it passed).
    """
    # 1. Exclusions
    combined_text = f"{item.title} {item.text}".lower()
    for exc in exclusions:
        exc_clean = exc.strip().lower()
        # Lookarounds, not \b: \b never matches beside a non-word edge such as "C++", ".NET" or "#ad".
        if exc_clean and re.search(r"(?<!\w)" + re.escape(exc_clean) + r"(?!\w)", combined_text):
            return False, f"Matched exclusion keyword '{exc_clean}'"

    # 2. Clickbait heuristics
    title = item.title.strip()
    if len(title) >= 20:
        alpha_chars = [c for c in title if c.isalpha()]
        if alpha_chars and (sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)) > 0.65:
            return False, "Clickbait: excessive uppercase characters in title"

    if re.search(r"[!?]{3,}", title):
        return False, "Clickbait: excessive punctuation in title"

    title_lower = title.lower()
    for phrase in CLICKBAIT_PHRASES:
        if phrase in title_lower:
            return False, f"Clickbait phrase detected: '{phrase}'"

    # 3. Minimum article length
    if item.kind in ("article", "news") and len(item.text.strip()) < min_article_length:
        return False, f"Article too short ({len(item.text.strip())} < {min_article_length} chars)"

    return True, "Passed Stage A rules"


def filter_embedding_gate(
    item: Item,
    item_vec: list[float],
    interests: Sequence[tuple[str, list[float], float]],
    threshold: float = 0.35,
) -> tuple[bool, float, str]:
    """Stage B: cheap semantic embedding gate.

    Args:
        item: The item being gated (not read; the vector stands for it).
        item_vec: The item's embedding.
        interests: Sequence of ``(interest_id, centroid_vector, weight)``.
        threshold: Minimum weighted similarity to pass.

    Returns:
        ``(passed, max_similarity, best_interest_id)``, where the similarity is the cosine times
        the interest's weight. With no interests, ``(True, 1.0, "default")``.
    """
    if not interests:
        return True, 1.0, "default"

    best_sim = -1.0
    best_id = ""

    for int_id, centroid, weight in interests:
        raw_sim = cosine_similarity(item_vec, centroid)
        weighted_sim = raw_sim * weight
        if weighted_sim > best_sim:
            best_sim = weighted_sim
            best_id = int_id

    passed = best_sim >= threshold
    return passed, best_sim, best_id
