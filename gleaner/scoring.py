"""Deterministic, cacheable scoring components and composite ranking formula."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from gleaner.embed import cosine_similarity
from gleaner.models import Item, Score, ScoreBreakdown

HALF_LIVES_SECONDS = {
    "news": 86400.0,  # 1 day
    "post": 86400.0,  # 1 day
    "release": 259200.0,  # 3 days
    "paper": 1209600.0,  # 14 days
    "article": 259200.0,  # 3 days
    "opportunity": 1209600.0,  # 14 days
}

RANKING_MODES: dict[str, dict[str, float]] = {
    "balanced": {
        "relevance": 0.45,
        "novelty": 0.15,
        "quality": 0.15,
        "momentum": 0.10,
        "actionability": 0.15,
    },
    "deep_research": {
        "relevance": 0.35,
        "novelty": 0.30,
        "quality": 0.20,
        "momentum": 0.05,
        "actionability": 0.10,
    },
    "breaking_radar": {
        "relevance": 0.25,
        "novelty": 0.10,
        "quality": 0.15,
        "momentum": 0.40,
        "actionability": 0.10,
    },
    "actionable": {
        "relevance": 0.25,
        "novelty": 0.10,
        "quality": 0.15,
        "momentum": 0.10,
        "actionability": 0.40,
    },
}


def rescore_with_mode(components: ScoreBreakdown, mode: str = "balanced") -> float:
    """Recalculate the composite final score using a ranking mode's weights.

    Args:
        components: The stored score components.
        mode: A key of ``RANKING_MODES``; an unknown mode falls back to ``"balanced"``.

    Returns:
        The decayed weighted sum, rounded to 4 places.
    """
    weights = RANKING_MODES.get(mode, RANKING_MODES["balanced"])
    raw = (
        weights["relevance"] * components.relevance
        + weights["novelty"] * components.novelty
        + weights["quality"] * components.quality
        + weights["momentum"] * components.momentum
        + weights["actionability"] * components.actionability
    )
    return round(components.decay * raw, 4)



def compute_decay(published_at: datetime | None, kind: str = "news", now: datetime | None = None) -> float:
    """Compute exponential time decay based on the content kind's half-life.

    Args:
        published_at: When the item was published; None means no decay. Naive times are UTC.
        kind: Item kind, a key of ``HALF_LIVES_SECONDS`` (unknown kinds use one day).
        now: The reference time; defaults to the current UTC time.

    Returns:
        The decay factor, clamped to [0.05, 1.0].
    """
    if published_at is None:
        return 1.0
    if now is None:
        now = datetime.now(UTC)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    age_seconds = max(0.0, (now - published_at).total_seconds())
    half_life_s = HALF_LIVES_SECONDS.get(kind, 86400.0)
    decay = math.exp(-0.693147 * age_seconds / half_life_s)
    return max(0.05, min(1.0, decay))


def compute_relevance(embed_sim: float, llm_relevance: float | None = None, weight: float = 1.0) -> float:
    """Blend embedding similarity with judge relevance, times the interest weight.

    Args:
        embed_sim: Unweighted embedding similarity to the matched interest.
        llm_relevance: The judge's relevance; when None, the embedding similarity is used alone.
        weight: The interest's weight.

    Returns:
        ``(0.4 * embed_sim + 0.6 * llm_relevance) * weight``, clamped to [0, 1].
    """
    if llm_relevance is not None:
        rel = (0.4 * embed_sim + 0.6 * llm_relevance) * weight
    else:
        rel = embed_sim * weight
    return max(0.0, min(1.0, rel))


def compute_novelty(
    item_vec: list[float],
    recent_seen_vecs: Sequence[list[float]] = (),
    llm_novelty: float | None = None,
) -> float:
    """Compute novelty as distance from the most similar recently seen item.

    Optionally averaged with the judge's novelty.

    Args:
        item_vec: The item's embedding.
        recent_seen_vecs: Embeddings of recently scored items; none means fully novel.
        llm_novelty: The judge's novelty; when given, averaged 50/50 with the embedding novelty.

    Returns:
        The novelty in [0, 1].
    """
    if recent_seen_vecs:
        max_sim = max(cosine_similarity(item_vec, v) for v in recent_seen_vecs)
        emb_novelty = max(0.0, 1.0 - max_sim)
    else:
        emb_novelty = 1.0

    if llm_novelty is not None:
        return 0.5 * emb_novelty + 0.5 * llm_novelty
    return emb_novelty


def compute_source_quality(trust: float = 0.5, likes: int = 0, dislikes: int = 0) -> float:
    """Source quality as a Beta posterior mean.

    Args:
        trust: The source's configured trust, worth five pseudo-observations.
        likes: Likes on the source's items.
        dislikes: Dislikes on the source's items.

    Returns:
        ``(1 + likes + trust*5) / (2 + likes + dislikes + 5)``.
    """
    return (1.0 + likes + trust * 5.0) / (2.0 + likes + dislikes + 5.0)


def compute_momentum(story_source_count: int = 1, signals: dict[str, Any] | None = None) -> float:
    """Compute momentum: a multi-source coverage bonus plus engagement.

    Engagement counts HN points and comments, GitHub stars and paper citations.

    Args:
        story_source_count: How many sources cover the item's story.
        signals: The item's engagement signals (``points``, ``comments``, ``stars``, ``citations``).

    Returns:
        The momentum in [0, 1]: a coverage bonus up to 0.4 plus engagement up to 0.6.
    """
    signals = signals or {}
    points = float(signals.get("points") or 0)
    comments = float(signals.get("comments") or 0)
    stars = float(signals.get("stars") or 0)
    citations = float(signals.get("citations") or 0)
    multi_source_bonus = min(0.4, max(0, story_source_count - 1) * 0.2)
    engagement = min(
        0.6,
        (points / 300.0) * 0.4 + (comments / 80.0) * 0.2 + (stars / 1000.0) * 0.4 + (citations / 50.0) * 0.4,
    )
    return min(1.0, multi_source_bonus + engagement)


def compute_actionability(
    kind: str = "news",
    signals: dict[str, Any] | None = None,
    judge_actionability: float | None = None,
) -> float:
    """Actionability from the item's kind, blended with the judge's value when judged.

    Args:
        kind: The item kind: release/opportunity give a prior of 0.8, paper 0.6, anything else 0.3.
        signals: The item's signals (not used).
        judge_actionability: The judge's actionability; when given, blended 70/30 with the prior.

    Returns:
        The actionability in [0, 1].
    """
    if kind in ("release", "opportunity"):
        prior = 0.8
    elif kind == "paper":
        prior = 0.6
    else:
        prior = 0.3
    if judge_actionability is None:
        return prior
    return 0.7 * judge_actionability + 0.3 * prior


def score_item(
    item: Item,
    item_vec: list[float],
    embed_sim: float = 0.5,
    llm_relevance: float | None = None,
    llm_novelty: float | None = None,
    judge_actionability: float | None = None,
    source_trust: float = 0.5,
    likes: int = 0,
    dislikes: int = 0,
    story_source_count: int = 1,
    recent_seen_vecs: Sequence[list[float]] = (),
    interest_weight: float = 1.0,
    profile_version: int = 1,
    now: datetime | None = None,
) -> Score:
    """Compute the score breakdown and final decayed score for an item.

    Args:
        item: The item; its kind, signals and publish time feed the components.
        item_vec: The item's embedding, for novelty.
        embed_sim: Unweighted embedding similarity to the matched interest.
        llm_relevance: The judge's relevance, or None to use embeddings alone.
        llm_novelty: The judge's novelty, or None to use embeddings alone.
        judge_actionability: The judge's actionability, or None for the kind prior alone.
        source_trust: The source's configured trust.
        likes: Likes on the source's items.
        dislikes: Dislikes on the source's items.
        story_source_count: How many sources cover the item's story.
        recent_seen_vecs: Embeddings of recently scored items, for novelty.
        interest_weight: The matched interest's weight.
        profile_version: The interest profile version the score is computed for.
        now: The reference time for decay; defaults to the current UTC time.

    Returns:
        The ``Score``, with components and final score rounded to 4 places.
    """
    relevance = compute_relevance(embed_sim, llm_relevance, interest_weight)
    novelty = compute_novelty(item_vec, recent_seen_vecs, llm_novelty)
    quality = compute_source_quality(source_trust, likes, dislikes)
    momentum = compute_momentum(story_source_count, item.signals)
    actionability = compute_actionability(item.kind, item.signals, judge_actionability)
    decay = compute_decay(item.published_at, item.kind, now=now)

    raw_weighted = (
        0.45 * relevance
        + 0.15 * novelty
        + 0.15 * quality
        + 0.10 * momentum
        + 0.15 * actionability
    )
    final = decay * raw_weighted

    breakdown = ScoreBreakdown(
        relevance=round(relevance, 4),
        novelty=round(novelty, 4),
        quality=round(quality, 4),
        momentum=round(momentum, 4),
        actionability=round(actionability, 4),
        decay=round(decay, 4),
    )

    return Score(
        item_id=item.id,
        profile_version=profile_version,
        components=breakdown,
        final=round(final, 4),
    )
