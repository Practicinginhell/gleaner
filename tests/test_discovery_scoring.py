"""Scoring components: momentum and actionability."""



from gleaner.scoring import compute_actionability, compute_momentum


def test_momentum_is_never_negative():
    assert compute_momentum(story_source_count=0) >= 0.0
    assert compute_momentum(story_source_count=0, signals={}) == compute_momentum(story_source_count=1, signals={})


def test_momentum_uses_stars_and_citations():
    assert compute_momentum(1, {"stars": 800}) > compute_momentum(1, {})
    assert compute_momentum(1, {"citations": 40}) > compute_momentum(1, {})
    assert compute_momentum(5, {"points": 10_000, "stars": 10_000, "citations": 10_000}) <= 1.0


def test_actionability_kind_prior_counts_when_judged():
    judged = 0.5
    release = compute_actionability(kind="release", judge_actionability=judged)
    news = compute_actionability(kind="news", judge_actionability=judged)
    assert release > news
    assert 0.0 <= news <= release <= 1.0


def test_actionability_without_judge_is_kind_prior():
    assert compute_actionability(kind="opportunity") == 0.8
    assert compute_actionability(kind="paper") == 0.6
    assert compute_actionability(kind="news") == 0.3
