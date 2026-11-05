"""Phase 7: Evaluation metrics, precision@20, nDCG, noise rate, and eval harness tests."""

import json
from pathlib import Path

from gleaner.db import Database
from gleaner.eval import (
    filter_recall,
    ndcg_at_k,
    noise_rate,
    precision_at_k,
    run_eval,
)
from gleaner.models import Item, SourceRow

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "discovery"


def test_metric_precision_at_k():
    ranked = [f"id-{i}" for i in range(20)]
    relevant = {f"id-{i}" for i in range(15)}  # 15 relevant items in top 20
    prec = precision_at_k(ranked, relevant, k=20)
    assert abs(prec - 0.75) < 1e-6


def test_metric_ndcg_at_k():
    ranked_perfect = ["doc-1", "doc-2", "doc-3"]
    scores = {"doc-1": 3.0, "doc-2": 2.0, "doc-3": 1.0}
    perfect_ndcg = ndcg_at_k(ranked_perfect, scores, k=3)
    assert abs(perfect_ndcg - 1.0) < 1e-6

    # Inverted ranking has lower nDCG
    ranked_inverted = ["doc-3", "doc-2", "doc-1"]
    inverted_ndcg = ndcg_at_k(ranked_inverted, scores, k=3)
    assert inverted_ndcg < 1.0
    assert inverted_ndcg > 0.5


def test_metric_noise_rate_and_filter_recall():
    shown = ["item-1", "item-2", "item-3", "item-4"]
    disliked = {"item-1"}  # 1 out of 4 is noise
    assert abs(noise_rate(shown, disliked) - 0.25) < 1e-6

    relevant_ground_truth = {"r-1", "r-2", "r-3", "r-4", "r-5"}
    survived_filters = {"r-1", "r-2", "r-3", "r-4", "other-1"}  # 4 out of 5 survived
    assert abs(filter_recall(survived_filters, relevant_ground_truth) - 0.80) < 1e-6


def test_eval_harness_measures_the_ranker_on_its_own_seeded_db():
    eval_json = FIXTURES_DIR / "labeled_eval.json"
    metrics = run_eval(eval_json, Database(":memory:"))

    for key in ("precision@20", "ndcg@20", "noise_rate", "filter_recall", "total_evaluated", "target_pass"):
        assert key in metrics
    assert metrics["target_pass"] is True
    assert metrics["filter_recall"] >= 0.8

    # The ranked feed beats showing the fixture in insertion order
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    insertion = [it["id"] for it in data["items"]]
    relevant = {it["id"] for it in data["items"] if it["relevant"]}
    graded = {it["id"]: float(it["relevance_score"]) for it in data["items"]}
    assert metrics["precision@20"] > precision_at_k(insertion, relevant, k=20)
    assert metrics["ndcg@20"] > ndcg_at_k(insertion, graded, k=20)

    # Deterministic: a second run gives the same numbers
    assert run_eval(eval_json, Database(":memory:")) == metrics


def test_eval_ignores_unrelated_items_in_the_callers_db():
    eval_json = FIXTURES_DIR / "labeled_eval.json"
    fresh = run_eval(eval_json, Database(":memory:"))

    busy = Database(":memory:")
    busy.upsert_source(SourceRow(id="src-other", type="rss", params={}, cadence_s=3600))
    for i in range(5):
        busy.upsert_item(Item(id=f"other-{i}", source_id="src-other", title=f"Unrelated {i}", text="Other data."))
    assert run_eval(eval_json, busy) == fresh
