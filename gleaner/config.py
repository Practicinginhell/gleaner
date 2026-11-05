"""Configuration loading and validation for discovery feed."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

VALID_SOURCE_TYPES = {"rss", "arxiv", "hn", "github", "nostr", "reddit", "web", "bluesky", "mastodon"}
VALID_INTEREST_KINDS = {"topic", "goal", "entity"}


class DiscoveryConfigError(ValueError):
    """Raised when configuration in YAML is invalid, incomplete, or inconsistent."""


def parse_cadence(cadence: str) -> int:
    """Parse cadence string like '30s', '15m', '2h', '1d' into seconds.

    Args:
        cadence: A positive number followed by a unit: s, m, h or d.

    Returns:
        The cadence in seconds.

    Raises:
        ValueError: If ``cadence`` is not a string in that format or is not greater than zero.
    """
    if not isinstance(cadence, str):
        raise ValueError(f"Invalid cadence {cadence!r}. Format must be a string <number>[s|m|h|d], e.g. '30m' or '2h'.")
    match = re.match(r"^(\d+)([smhd])$", cadence.strip().lower())
    if not match:
        raise ValueError(f"Invalid cadence '{cadence}'. Format must be <number>[s|m|h|d], e.g. '30m' or '2h'.")
    val, unit = int(match.group(1)), match.group(2)
    if val <= 0:
        raise ValueError(f"Cadence must be greater than zero, got '{cadence}'.")
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return val * multipliers[unit]


class SourceConfig(BaseModel):
    """One validated source entry from the sources config.

    Attributes:
        id: Unique source id.
        type: Connector type, one of ``VALID_SOURCE_TYPES``.
        params: Connector-specific settings, such as the feed URL.
        cadence: Fetch cadence as written, e.g. '1h'.
        cadence_s: The cadence in seconds.
        enabled: Whether the source is fetched at all.
        trust: How much the source is trusted, from 0.0 to 1.0.
    """

    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    cadence: str = "1h"
    cadence_s: int = 3600
    enabled: bool = True
    trust: float = 0.5


class InterestConfig(BaseModel):
    """One validated interest entry from the interests config.

    Attributes:
        id: Unique interest id.
        label: Short name shown to the user.
        description: What the interest covers, in the user's words.
        keywords: Terms that signal a match.
        exclusions: Terms that rule a match out.
        weight: Relative importance of the interest.
        kind: One of ``VALID_INTEREST_KINDS``.
        active: Whether the interest is in use.
    """

    id: str
    label: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    weight: float = 1.0
    kind: str = "topic"
    active: bool = True


def _float_field(item: dict[str, Any], key: str, default: float, where: str) -> float:
    """Return ``item[key]`` as a float, or raise DiscoveryConfigError naming ``where``."""
    value = item.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise DiscoveryConfigError(f"Field '{key}' {where} must be a number, got {value!r}") from e


def _str_list_field(item: dict[str, Any], key: str, where: str) -> list[str]:
    """Return ``item[key]`` as a list of strings.

    Otherwise raise DiscoveryConfigError naming ``where``.
    """
    value = item.get(key) or []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise DiscoveryConfigError(f"Field '{key}' {where} must be a list of strings, got {value!r}")
    return list(value)


def load_sources_config(path: str | Path) -> list[SourceConfig]:
    """Load and validate the sources YAML file.

    Args:
        path: Path to the sources YAML file.

    Returns:
        One config per source entry, in file order.

    Raises:
        DiscoveryConfigError: If the file is missing, is not valid YAML, or has an invalid, duplicate or
            incomplete entry.
    """
    p = Path(path)
    if not p.is_file():
        raise DiscoveryConfigError(f"Sources config file '{p.name}' not found at {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        raise DiscoveryConfigError(f"Failed to parse YAML in '{p.name}': {e}") from e

    if not isinstance(data, dict):
        raise DiscoveryConfigError(f"'{p.name}' must be a mapping with a 'sources' list at the top level")
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list):
        raise DiscoveryConfigError(f"Field 'sources' in '{p.name}' must be a list")

    seen_ids: set[str] = set()
    configs: list[SourceConfig] = []
    for idx, item in enumerate(raw_sources):
        if not isinstance(item, dict):
            raise DiscoveryConfigError(f"Entry {idx} in '{p.name}' must be a dictionary")
        src_id = item.get("id")
        if not src_id or not isinstance(src_id, str):
            raise DiscoveryConfigError(f"Missing or invalid field 'id' at entry {idx} in '{p.name}'")
        if src_id in seen_ids:
            raise DiscoveryConfigError(f"Duplicate source id '{src_id}' found in '{p.name}'")
        seen_ids.add(src_id)

        src_type = item.get("type")
        if not isinstance(src_type, str) or src_type not in VALID_SOURCE_TYPES:
            raise DiscoveryConfigError(
                f"Invalid field 'type' '{src_type}' in '{p.name}' for source '{src_id}'. Must be one of {sorted(VALID_SOURCE_TYPES)}"
            )

        cadence_str = item.get("cadence", "1h")
        try:
            cadence_s = parse_cadence(cadence_str)
        except ValueError as e:
            raise DiscoveryConfigError(
                f"Invalid field 'cadence' '{cadence_str}' in '{p.name}' for source '{src_id}': {e}"
            ) from e

        trust = _float_field(item, "trust", 0.5, f"in '{p.name}' for source '{src_id}'")
        if not (0.0 <= trust <= 1.0):
            raise DiscoveryConfigError(f"Field 'trust' in '{p.name}' for source '{src_id}' must be between 0.0 and 1.0")

        configs.append(
            SourceConfig(
                id=src_id,
                type=src_type,
                params=item.get("params") or {},
                cadence=cadence_str,
                cadence_s=cadence_s,
                enabled=bool(item.get("enabled", True)),
                trust=trust,
            )
        )
    return configs


def load_interests_config(path: str | Path) -> list[InterestConfig]:
    """Load and validate the interests YAML file.

    Args:
        path: Path to the interests YAML file.

    Returns:
        One config per interest entry, in file order.

    Raises:
        DiscoveryConfigError: If the file is missing or its content is invalid.
    """
    p = Path(path)
    if not p.is_file():
        raise DiscoveryConfigError(f"Interests config file '{p.name}' not found at {p}")
    return parse_interests_yaml(p.read_text(encoding="utf-8"), p.name)


def parse_interests_yaml(text: str, name: str) -> list[InterestConfig]:
    """Validate interests YAML text.

    Args:
        text: The YAML text.
        name: The file name, used in every error message.

    Returns:
        One config per interest entry, in file order.

    Raises:
        DiscoveryConfigError: If the text is not valid YAML or has an invalid, duplicate or incomplete entry.
    """
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise DiscoveryConfigError(f"Failed to parse YAML in '{name}': {e}") from e

    if not isinstance(data, dict):
        raise DiscoveryConfigError(f"'{name}' must be a mapping with an 'interests' list at the top level")
    raw_interests = data.get("interests")
    if not isinstance(raw_interests, list):
        raise DiscoveryConfigError(f"Field 'interests' in '{name}' must be a list")

    seen_ids: set[str] = set()
    configs: list[InterestConfig] = []
    for idx, item in enumerate(raw_interests):
        if not isinstance(item, dict):
            raise DiscoveryConfigError(f"Entry {idx} in '{name}' must be a dictionary")
        int_id = item.get("id")
        if not int_id or not isinstance(int_id, str):
            raise DiscoveryConfigError(f"Missing or invalid field 'id' at entry {idx} in '{name}'")
        if int_id in seen_ids:
            raise DiscoveryConfigError(f"Duplicate interest id '{int_id}' found in '{name}'")
        seen_ids.add(int_id)

        label = item.get("label")
        if not label or not isinstance(label, str):
            raise DiscoveryConfigError(f"Missing or invalid field 'label' in '{name}' for interest '{int_id}'")

        description = item.get("description")
        if not description or not isinstance(description, str):
            raise DiscoveryConfigError(f"Missing or invalid field 'description' in '{name}' for interest '{int_id}'")

        kind = item.get("kind", "topic")
        if not isinstance(kind, str) or kind not in VALID_INTEREST_KINDS:
            raise DiscoveryConfigError(
                f"Invalid field 'kind' '{kind}' in '{name}' for interest '{int_id}'. Must be one of {sorted(VALID_INTEREST_KINDS)}"
            )

        where = f"in '{name}' for interest '{int_id}'"
        weight = _float_field(item, "weight", 1.0, where)
        configs.append(
            InterestConfig(
                id=int_id,
                label=label,
                description=description,
                keywords=_str_list_field(item, "keywords", where),
                exclusions=_str_list_field(item, "exclusions", where),
                weight=weight,
                kind=kind,
                active=bool(item.get("active", True)),
            )
        )
    return configs


class QuestionConfig(BaseModel):
    """One validated standing question from questions.yaml.

    Attributes:
        id: Unique question id.
        text: The question.
        active: Whether the question is shown and kept up to date.
        linked_interests: Ids of interests the question draws on.
        watch: Path of a dependency file to watch; falls back to the file's top-level ``watch``.
    """

    id: str
    text: str
    active: bool = True
    linked_interests: list[str] = Field(default_factory=list)
    watch: str | None = None


class QuestionsConfig(BaseModel):
    """The validated content of questions.yaml.

    Attributes:
        questions: The standing questions, in file order.
        watch: Top-level dependency file to watch, e.g. 'pyproject.toml'.
    """

    questions: list[QuestionConfig] = Field(default_factory=list)
    watch: str | None = None


def parse_questions_yaml(content: str, name: str = "questions.yaml") -> QuestionsConfig:
    """Parse and validate questions.yaml configuration.

    Args:
        content: The YAML text.
        name: The file name, used in error messages.

    Returns:
        The questions and the top-level watch path.

    Raises:
        DiscoveryConfigError: If the text is not valid YAML, is not a mapping, lacks a 'questions' list, or a
            question has no valid id or text.
    """
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise DiscoveryConfigError(f"Invalid YAML in '{name}': {e}") from e

    if not isinstance(data, dict):
        raise DiscoveryConfigError(f"'{name}' root must be a mapping, got {type(data).__name__}")

    raw_questions = data.get("questions")
    if raw_questions is None:
        raise DiscoveryConfigError(f"'{name}' must contain a 'questions' key")
    if not isinstance(raw_questions, list):
        raise DiscoveryConfigError(f"'questions' in '{name}' must be a list, got {type(raw_questions).__name__}")

    questions = []
    for i, item in enumerate(raw_questions):
        if not isinstance(item, dict):
            raise DiscoveryConfigError(f"Question #{i} in '{name}' must be a mapping")
        q_id = item.get("id")
        if not q_id or not isinstance(q_id, str):
            raise DiscoveryConfigError(f"Missing or invalid 'id' in question #{i}")
        text = item.get("text")
        if not text or not isinstance(text, str):
            raise DiscoveryConfigError(f"Missing or invalid 'text' in question '{q_id}'")

        linked = item.get("linked_interests", [])
        if not isinstance(linked, list):
            linked = []

        questions.append(
            QuestionConfig(
                id=q_id,
                text=text.strip(),
                active=bool(item.get("active", True)),
                linked_interests=[str(x) for x in linked],
                watch=item.get("watch") or data.get("watch"),
            )
        )

    watch_path = data.get("watch")
    if watch_path and not isinstance(watch_path, str):
        watch_path = None

    return QuestionsConfig(questions=questions, watch=watch_path)
