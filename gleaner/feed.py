"""Feed selection, MMR diversity re-ranking, and feed card generation."""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, Field

from gleaner.db import Database
from gleaner.embed import cosine_similarity
from gleaner.models import Item, ScoreBreakdown
from gleaner.normalize import short_summary
from gleaner.scoring import compute_decay, rescore_with_mode
from gleaner.stories import synthesize_story_perspectives


class FeedCard(BaseModel):
    """One card in the ranked feed.

    Attributes:
        item: The item shown.
        score: The item's final score under the chosen ranking mode.
        components: The score breakdown, with decay recomputed at read time.
        why: Why the item was shown.
        summary: A short summary of the item.
        also_covered_by: Other sources covering the same story.
        perspectives: Perspective key to takeaway, across the story's items.
        exploring: True for an exploration slot rather than a top-scored item.
        diversified: True when MMR moved the card away from its pure score position.
    """

    item: Item
    score: float
    components: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    why: str = ""
    summary: str = ""
    also_covered_by: list[str] = Field(default_factory=list)
    perspectives: dict[str, str] = Field(default_factory=dict)
    exploring: bool = False
    diversified: bool = False


def select_feed_items(
    candidates: Sequence[tuple[Item, float, list[float]]],
    max_per_source: int = 3,
    limit: int = 20,
    lambda_mmr: float = 0.7,
) -> list[tuple[Item, float, list[float]]]:
    """Select diverse feed items: one per story, capped per source, MMR re-ranked.

    Each story's highest-scoring item represents it; its `signals` gain `also_covered_by` and
    `perspectives` from the story's other items.

    Args:
        candidates: (item, score, embedding vector) tuples.
        max_per_source: Most cards any one source may get.
        limit: Most cards to return.
        lambda_mmr: MMR weight on score versus dissimilarity to cards already selected.

    Returns:
        The selected (item, score, vector) tuples, in feed order.
    """
    if not candidates:
        return []

    # 1. Group by story_id: pick highest scoring representative per story
    by_story: dict[str, list[tuple[Item, float, list[float]]]] = {}
    for item, score, vec in candidates:
        st_id = item.story_id or f"single-{item.id}"
        by_story.setdefault(st_id, []).append((item, score, vec))

    story_representatives: list[tuple[Item, float, list[float]]] = []
    for group in by_story.values():
        # Sort group by score descending
        group.sort(key=lambda x: x[1], reverse=True)
        rep_item, rep_score, rep_vec = group[0]

        # Gather other sources and synthesize multi-perspective insights
        other_sources = set()
        group_items = [it for it, _, _ in group]
        for other_item in group_items[1:]:
            if other_item.source_id and other_item.source_id != rep_item.source_id:
                other_sources.add(other_item.source_id)
        if other_sources or len(group_items) > 1:
            if not rep_item.signals:
                rep_item.signals = {}
            if other_sources:
                rep_item.signals["also_covered_by"] = sorted(other_sources)
            persp = synthesize_story_perspectives(st_id, group_items)
            if persp:
                rep_item.signals["perspectives"] = persp

        story_representatives.append((rep_item, rep_score, rep_vec))

    # Sort candidates by raw score descending
    story_representatives.sort(key=lambda x: x[1], reverse=True)

    # 2. MMR re-ranking with per-source caps
    selected: list[tuple[Item, float, list[float]]] = []
    source_counts: dict[str, int] = {}
    remaining = list(story_representatives)

    while remaining and len(selected) < limit:
        best_idx = -1
        best_mmr_score = -float("inf")

        for idx, (item, score, vec) in enumerate(remaining):
            # Check source cap
            src = item.source_id or "unknown"
            if source_counts.get(src, 0) >= max_per_source:
                continue

            if not selected:
                mmr_val = score
            else:
                max_sim = max(cosine_similarity(vec, s_vec) for _, _, s_vec in selected)
                mmr_val = lambda_mmr * score - (1.0 - lambda_mmr) * max_sim

            if mmr_val > best_mmr_score:
                best_mmr_score = mmr_val
                best_idx = idx

        if best_idx == -1:
            # All remaining candidates exceed source caps or exhausted
            break

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        src = chosen[0].source_id or "unknown"
        source_counts[src] = source_counts.get(src, 0) + 1

    return selected


EXPLORATION_SHARE = 0.10


def generate_feed(db: Database, limit: int = 20, mode: str = "balanced") -> list[FeedCard]:
    """Generate the ranked feed from scored items, optionally reweighted by mode.

    Only items whose status is still 'scored' or 'exploring' and whose source is enabled are
    shown, so a later quarantine or a muted source takes effect at read time. About 10% of the
    slots (at least one) are reserved for 'exploring' items, which never take more than that.

    Args:
        db: Database to read scored items from.
        limit: Most cards to return.
        mode: Ranking mode passed to `rescore_with_mode`; "balanced" keeps the stored order.

    Returns:
        The feed cards, in display order.
    """
    with db.connect() as conn:
        rows = conn.execute(
            """
        SELECT * FROM (
            SELECT items.*, scores.final, scores.components, judgments.why, judgments.summary,
                   ROW_NUMBER() OVER (PARTITION BY items.status ORDER BY scores.final DESC) AS status_rank
            FROM items
            JOIN scores ON scores.item_id = items.id
            LEFT JOIN judgments ON judgments.item_id = items.id
            LEFT JOIN sources ON sources.id = items.source_id
            WHERE items.status IN ('scored', 'exploring')
              AND COALESCE(sources.enabled, 1) = 1
        )
        -- Top 100 per status, so low-scoring exploring items are not crowded out of the window
        WHERE status_rank <= 100
        ORDER BY final DESC;
        """
        ).fetchall()

    if not rows:
        return []

    candidates: list[tuple[Item, float, list[float]]] = []
    card_metadata: dict[str, dict] = {}

    for r in rows:
        item = db._row_to_item(r)
        components_dict = json.loads(r["components"]) if r["components"] else {}
        components = ScoreBreakdown(**components_dict)
        # Recompute exponential decay at read time so rankings stay fresh
        components.decay = compute_decay(item.published_at, kind=item.kind)
        score_val = rescore_with_mode(components, mode=mode)

        card_metadata[item.id] = {
            "components": components,
            "why": r["why"] or f"Matches your active interests (score: {score_val:.2f})",
            # summaries stored before short_summary were cut mid-word at 200 characters: redo those
            "summary": short_summary(item.text) if r["summary"] in (None, "", item.text[:200]) else r["summary"],
        }
        # Retrieve real item embedding vector from vector store (fallback to placeholder only if vector missing)
        vec = db.get_item_vector(item.id)
        if vec is None:
            vec = [0.1] * db.vec_dim
        candidates.append((item, score_val, vec))

    if mode != "balanced":
        candidates.sort(key=lambda x: x[1], reverse=True)

    main_pool = [c for c in candidates if c[0].status != "exploring"]
    explore_pool = [c for c in candidates if c[0].status == "exploring"]
    budget = min(len(explore_pool), max(1, round(limit * EXPLORATION_SHARE))) if limit > 0 else 0
    main = select_feed_items(main_pool, max_per_source=3, limit=limit - budget)
    shown_stories = {it.story_id for it, _, _ in main if it.story_id}
    explore_pool = [c for c in explore_pool if not c[0].story_id or c[0].story_id not in shown_stories]
    explore = select_feed_items(explore_pool, max_per_source=3, limit=budget)
    # Exploring cards sit at every 10th slot (or at the end of a short feed) rather than all at the bottom
    selected = list(main)
    for k, pick in enumerate(explore):
        selected.insert(min(len(selected), (k + 1) * 10 - 1), pick)

    # Compare final selection order with pure raw-score order to flag MMR diversity shifts
    pure_ranked_ids = [c[0].id for c in sorted(candidates, key=lambda x: x[1], reverse=True)]
    cards: list[FeedCard] = []
    for idx, (item, score_val, _) in enumerate(selected):
        meta = card_metadata.get(item.id, {})
        also_covered = item.signals.get("also_covered_by", [])
        perspectives = item.signals.get("perspectives", {})
        pure_idx = pure_ranked_ids.index(item.id) if item.id in pure_ranked_ids else idx
        is_diversified = (idx != pure_idx) and (item.status != "exploring")
        cards.append(
            FeedCard(
                item=item,
                score=score_val,
                components=meta.get("components", ScoreBreakdown()),
                why=meta.get("why", ""),
                summary=meta.get("summary", ""),
                also_covered_by=also_covered,
                perspectives=perspectives,
                exploring=(item.status == "exploring"),
                diversified=is_diversified,
            )
        )
    return cards
