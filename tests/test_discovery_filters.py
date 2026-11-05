"""Phase 2: Relevance filtering stages A (rules) and B (embedding gate) tests."""



from gleaner.embed import FakeEmbedder
from gleaner.filters import filter_embedding_gate, filter_rules
from gleaner.models import Item


def test_stage_a_rules_filters_exclusions():
    item = Item(
        id="item-crypto",
        title="Decentralized Crypto Blockchain Mining",
        text="A detailed deep dive into proof of stake and crypto trading algorithms.",
    )
    passed, reason = filter_rules(item, exclusions=["crypto", "blockchain"])
    assert passed is False
    assert "exclusion" in reason.lower()


def test_stage_a_rules_filters_clickbait():
    item = Item(
        id="item-clickbait",
        title="YOU WON'T BELIEVE THIS ONE SHOCKING TRICK TO BECOME A MILLIONAIRE!!!",
        text="Click here right now to find out the secret methods they don't want you to know about.",
    )
    passed, reason = filter_rules(item, exclusions=[])
    assert passed is False
    assert "clickbait" in reason.lower()


def test_stage_a_rules_filters_short_articles():
    item = Item(
        id="item-short",
        kind="article",
        title="Too Short",
        text="Only 20 characters.",
    )
    passed, reason = filter_rules(item, exclusions=[])
    assert passed is False
    assert "short" in reason.lower()


def test_stage_b_embedding_gate_drops_off_topic_with_zero_model_calls():
    embedder = FakeEmbedder(dim=64)

    # Interest centroid: Agent Frameworks
    interest_vec = embedder.embed(
        ["LangGraph, graphflow, and autonomous multi-agent orchestration frameworks."]
    )[0]
    interests = [("agent-frameworks", interest_vec, 1.0)]

    on_topic = Item(
        id="on-topic",
        title="Agent Workflows in LangGraph",
        text="Building complex multi-agent workflows with LangGraph and graphflow runtime.",
    )
    off_topic = Item(
        id="off-topic",
        title="Baking Sourdough Bread at Home",
        text="How to feed sourdough starter yeast culture for sourdough bread baking.",
    )

    on_vec = embedder.embed([on_topic.title + " " + on_topic.text])[0]
    off_vec = embedder.embed([off_topic.title + " " + off_topic.text])[0]

    passed_on, sim_on, best_id_on = filter_embedding_gate(on_topic, on_vec, interests, threshold=0.35)
    passed_off, sim_off, _best_id_off = filter_embedding_gate(off_topic, off_vec, interests, threshold=0.35)

    assert passed_on is True
    assert best_id_on == "agent-frameworks"
    assert sim_on >= 0.35

    assert passed_off is False
    assert sim_off < 0.35


def test_stage_a_exclusions_match_terms_with_non_word_edges():
    cases = [
        ("C++", "Why I love C++ templates"),
        (".NET", "What's new in .NET 9"),
        ("#ad", "Great laptop deal #ad"),
    ]
    for exclusion, title in cases:
        item = Item(id="x", kind="post", title=title, text="")
        passed, reason = filter_rules(item, exclusions=[exclusion])
        assert passed is False, f"{exclusion!r} should exclude {title!r}"
        assert "exclusion" in reason.lower()


def test_stage_a_exclusions_still_respect_word_boundaries():
    item = Item(id="x", kind="post", title="Applied cryptography basics", text="")
    passed, _ = filter_rules(item, exclusions=["crypto", "C++"])
    assert passed is True
