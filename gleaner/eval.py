"""Evaluation harness for precision@20, nDCG@20, noise rate, and filter recall."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

from gleaner.db import Database
from gleaner.feed import generate_feed
from gleaner.models import Item, SourceRow


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int = 20) -> float:
    """Fraction of relevant items in the top K.

    Args:
        ranked_ids: Item ids in ranked order.
        relevant_ids: The ids labeled relevant.
        k: The cutoff rank.

    Returns:
        The fraction over the items actually shown (fewer than K when the list is short); 0.0 when empty.
    """
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(top_k)


def dcg_at_k(ranked_ids: Sequence[str], relevance_scores: dict[str, float], k: int = 20) -> float:
    """Discounted Cumulative Gain at rank K.

    Args:
        ranked_ids: Item ids in ranked order.
        relevance_scores: Graded relevance per id; missing ids count as 0.
        k: The cutoff rank.

    Returns:
        The DCG.
    """
    dcg = 0.0
    for idx, doc_id in enumerate(ranked_ids[:k]):
        rel = relevance_scores.get(doc_id, 0.0)
        # Position discount: log2(rank + 1), rank starts at 1 so log2(idx + 2)
        dcg += rel / math.log2(idx + 2)
    return dcg


def ndcg_at_k(ranked_ids: Sequence[str], relevance_scores: dict[str, float], k: int = 20) -> float:
    """Normalized Discounted Cumulative Gain at rank K.

    Args:
        ranked_ids: Item ids in ranked order.
        relevance_scores: Graded relevance per id; missing ids count as 0.
        k: The cutoff rank.

    Returns:
        DCG divided by the ideal ordering's DCG; 0.0 when the ideal is zero.
    """
    dcg = dcg_at_k(ranked_ids, relevance_scores, k=k)
    ideal_ranked = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
    idcg = dcg_at_k(ideal_ranked, relevance_scores, k=k)
    if idcg <= 1e-12:
        return 0.0
    return dcg / idcg


def noise_rate(shown_ids: Sequence[str], disliked_ids: set[str]) -> float:
    """Share of shown items that are irrelevant or disliked.

    Args:
        shown_ids: The ids shown to the user.
        disliked_ids: The ids labeled irrelevant or disliked.

    Returns:
        The share; 0.0 when nothing was shown.
    """
    if not shown_ids:
        return 0.0
    noise_count = sum(1 for doc_id in shown_ids if doc_id in disliked_ids)
    return noise_count / len(shown_ids)


def filter_recall(surviving_ids: set[str], relevant_ground_truth: set[str]) -> float:
    """Fraction of truly relevant items that survived stage A/B filtering.

    Args:
        surviving_ids: The ids that survived filtering.
        relevant_ground_truth: The ids labeled relevant.

    Returns:
        The fraction; 1.0 when nothing is labeled relevant.
    """
    if not relevant_ground_truth:
        return 1.0
    retained_relevant = len(surviving_ids & relevant_ground_truth)
    return retained_relevant / len(relevant_ground_truth)


def _seed_eval_db(db: Database, data: dict) -> list[Item]:
    """Load the labeled fixture.

    Each item goes under its labeled source, each interest with its keywords.

    Args:
        db: The database to seed.
        data: The parsed fixture, with ``interests`` and ``items`` lists.

    Returns:
        The stored items.
    """
    for spec in data.get("interests") or []:
        if isinstance(spec, str):
            spec = {"id": spec}
        int_id = spec["id"]
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO interests (id, label, description, keywords, exclusions, weight, active, profile_version)
                VALUES (?, ?, ?, ?, '[]', 1.0, 1, 1);
                """,
                (
                    int_id,
                    spec.get("label") or int_id.replace("-", " ").title(),
                    spec.get("description") or f"Research and developments in {int_id}",
                    json.dumps(spec.get("keywords") or []),
                ),
            )

    items: list[Item] = []
    for it in data.get("items") or []:
        source_id = it.get("source") or "eval-src"
        if db.get_source(source_id) is None:
            db.upsert_source(SourceRow(id=source_id, type="rss", params={}, cadence_s=3600))
        item = Item(id=it["id"], source_id=source_id, title=it.get("title", ""), text=it.get("text") or it.get("title", ""))
        db.upsert_item(item)
        items.append(item)
    return items


def run_eval(labels_path: str | Path, db: Database | None = None) -> dict[str, float]:
    """Run deterministic evaluation of the ranker against a labeled fixture corpus.

    The fixture is always loaded into a fresh in-memory database and run through the real
    scoring pipeline and feed selector, so the metrics measure the ranker and nothing else.
    ``db`` is accepted for older callers and is neither read nor written.

    Args:
        labels_path: The labeled fixture JSON.
        db: Ignored.

    Returns:
        Precision@20, nDCG@20, noise rate and filter recall (rounded to 4 places), the number
        of items ranked, and whether the targets were met.
    """
    from gleaner.workflows import process_and_score_items

    data = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    labeled_items = data.get("items") or []

    relevant_ids = {it["id"] for it in labeled_items if it.get("relevant")}
    disliked_ids = {it["id"] for it in labeled_items if not it.get("relevant")}
    relevance_scores = {it["id"]: float(it.get("relevance_score", 0.0)) for it in labeled_items}

    eval_db = Database(":memory:")
    try:
        items = _seed_eval_db(eval_db, data)
        if items:
            process_and_score_items(eval_db, items)
        ranked_ids = [c.item.id for c in generate_feed(eval_db, limit=20)]
        with eval_db.connect() as conn:
            survived_rows = conn.execute("SELECT id FROM items WHERE status IN ('scored', 'exploring');").fetchall()
        surviving_ids = {r["id"] for r in survived_rows}
    finally:
        eval_db.close()

    p_at_20 = precision_at_k(ranked_ids, relevant_ids, k=20)
    ndcg_20 = ndcg_at_k(ranked_ids, relevance_scores, k=20)
    noise = noise_rate(ranked_ids, disliked_ids)
    recall = filter_recall(surviving_ids, relevant_ids)
    target_pass = (p_at_20 >= 0.60 and recall >= 0.80 and noise <= 0.25)

    return {
        "precision@20": round(p_at_20, 4),
        "ndcg@20": round(ndcg_20, 4),
        "noise_rate": round(noise, 4),
        "filter_recall": round(recall, 4),
        "total_evaluated": len(ranked_ids),
        "target_pass": target_pass,
    }
