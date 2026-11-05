"""Phase 0: config loading and validation tests."""

from pathlib import Path

import pytest

from gleaner.config import (
    DiscoveryConfigError,
    load_interests_config,
    load_sources_config,
    parse_cadence,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "discovery"


def test_parse_cadence():
    assert parse_cadence("30s") == 30
    assert parse_cadence("15m") == 900
    assert parse_cadence("2h") == 7200
    assert parse_cadence("1d") == 86400
    with pytest.raises(ValueError):
        parse_cadence("invalid")
    with pytest.raises(ValueError):
        parse_cadence("-5m")


def test_load_valid_sources():
    sources = load_sources_config(FIXTURES_DIR / "sources.yaml")
    assert len(sources) == 4
    hn = next(s for s in sources if s.id == "hn-top")
    assert hn.type == "hn"
    assert hn.cadence_s == 1800
    assert hn.trust == 0.8
    assert hn.enabled is True

    arxiv = next(s for s in sources if s.id == "arxiv-ai")
    assert arxiv.type == "arxiv"
    assert arxiv.cadence_s == 43200
    assert arxiv.params["categories"] == ["cs.AI", "cs.CL"]


def test_load_valid_interests():
    interests = load_interests_config(FIXTURES_DIR / "interests.yaml")
    assert len(interests) == 2
    agent = next(i for i in interests if i.id == "agent-frameworks")
    assert agent.label == "Agent Frameworks"
    assert "graphflow" in agent.keywords
    assert "crypto" in agent.exclusions
    assert agent.weight == 1.0
    assert agent.active is True


def test_reject_unknown_source_type():
    path = FIXTURES_DIR / "invalid_type_sources.yaml"
    with pytest.raises(DiscoveryConfigError) as exc_info:
        load_sources_config(path)
    msg = str(exc_info.value)
    assert str(path.name) in msg
    assert "type" in msg.lower()


def test_reject_invalid_cadence():
    path = FIXTURES_DIR / "invalid_cadence_sources.yaml"
    with pytest.raises(DiscoveryConfigError) as exc_info:
        load_sources_config(path)
    msg = str(exc_info.value)
    assert str(path.name) in msg
    assert "cadence" in msg.lower()


def test_reject_duplicate_ids(tmp_path):
    dup_file = tmp_path / "dup.yaml"
    dup_file.write_text("""sources:
  - id: "same"
    type: "rss"
    params: { url: "http://example.com/1" }
    cadence: "1h"
  - id: "same"
    type: "rss"
    params: { url: "http://example.com/2" }
    cadence: "1h"
""")
    with pytest.raises(DiscoveryConfigError) as exc_info:
        load_sources_config(dup_file)
    assert "duplicate" in str(exc_info.value).lower()


_SRC = '  - id: "s1"\n    type: "rss"\n    params: { url: "https://example.com/feed.xml" }\n'
_INT = '  - id: "i1"\n    label: "Agents"\n    description: "Agent frameworks."\n'


@pytest.mark.parametrize(
    ("loader", "body", "field"),
    [
        (load_sources_config, "- just a list\n", "sources"),
        (load_sources_config, f"sources:\n{_SRC}    cadence: 30\n", "cadence"),
        (load_sources_config, f"sources:\n{_SRC}    trust: high\n", "trust"),
        (load_sources_config, f"sources:\n{_SRC}    trust: [1]\n", "trust"),
        (load_sources_config, 'sources:\n  - id: "s1"\n    type: [rss]\n', "type"),
        (load_interests_config, "just a string\n", "interests"),
        (load_interests_config, f"interests:\n{_INT}    weight: heavy\n", "weight"),
        (load_interests_config, f"interests:\n{_INT}    keywords: agents\n", "keywords"),
        (load_interests_config, f"interests:\n{_INT}    exclusions: crypto\n", "exclusions"),
        (load_interests_config, f"interests:\n{_INT}    kind: [topic]\n", "kind"),
    ],
)
def test_malformed_values_fail_with_file_and_field_named(tmp_path, loader, body, field):
    path = tmp_path / "bad_config.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(DiscoveryConfigError) as exc_info:
        loader(path)
    msg = str(exc_info.value)
    assert "bad_config.yaml" in msg
    assert f"'{field}'" in msg


def test_hackathon_content_rules_exclude_medical_political_and_war_items():
    """The event bans medical, political and war content; every interest carries the same exclusions."""
    from gleaner.config import load_interests_config
    from gleaner.filters import filter_rules
    from gleaner.models import Item

    interests = load_interests_config("config/interests.yaml")
    rules = {tuple(i.exclusions) for i in interests}
    assert len(rules) == 1  # one shared list: deactivating an interest can't drop the rules
    exclusions = list(rules.pop())

    body = " An agent workflow with tool calls, evaluated on several public benchmarks." * 5
    for title in ("Agentic LLM reasoning in pediatric diabetes outcome prediction",
                  "French national assembly votes 0.055 per GB internet tax",
                  "Drone swarms and the war in the region"):
        assert filter_rules(Item(id="x", title=title, text=body), exclusions=exclusions)[0] is False, title
    for title in ("Software agents for warehouse robots", "Local-first sync with SQLite"):
        assert filter_rules(Item(id="y", title=title, text=body), exclusions=exclusions)[0] is True, title
