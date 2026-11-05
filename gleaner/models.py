"""Data models for discovery feed."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class SourceRow(BaseModel):
    """A configured source as stored in the database, with its fetch state.

    Attributes:
        id: Unique source id from the sources config.
        type: Connector type, e.g. 'rss', 'arxiv' or 'hn'.
        params: Connector-specific settings, such as the feed URL.
        cadence_s: Seconds between fetches.
        next_due_at: When the source is next due to be fetched; None means now.
        etag: HTTP ETag from the last fetch, for conditional requests.
        last_modified: HTTP Last-Modified from the last fetch, for conditional requests.
        failures: Count of consecutive failed fetches.
        trust: How much the source is trusted, from 0.0 to 1.0.
        enabled: Whether the source is fetched at all.
    """

    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    cadence_s: int = 3600
    next_due_at: datetime | None = None
    etag: str | None = None
    last_modified: str | None = None
    failures: int = 0
    trust: float = 0.5
    enabled: bool = True


class Item(BaseModel):
    """One collected piece of content in the common shape, stored once.

    Attributes:
        id: Unique item id.
        source_id: Id of the source it was collected from, if any.
        kind: Content kind, e.g. 'news'; sets the half-life of its time decay.
        url: Link to the original.
        title: Title of the content.
        author: Author name, when known.
        published_at: When the original was published, when known.
        fetched_at: When gleaner collected it.
        lang: Language code.
        text: The content text (a snippet, not a full copy of an article).
        content_hash: Hash of the content, used to find duplicates.
        signals: Extra source signals, e.g. ``all_sources`` after deduplication.
        story_id: Id of the story the item belongs to, if clustered.
        status: Pipeline status, e.g. 'raw', 'scored', 'exploring' or 'dropped_stage_a'.
    """

    id: str
    source_id: str | None = None
    kind: str = "news"
    url: str | None = None
    title: str
    author: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime | None = Field(default_factory=utcnow)
    lang: str = "en"
    text: str = ""
    content_hash: str | None = None
    signals: dict[str, Any] = Field(default_factory=dict)
    story_id: str | None = None
    status: str = "raw"


class Story(BaseModel):
    """A group of items covering the same event across sources.

    Attributes:
        id: Unique story id.
        centroid: Mean embedding of the story's items.
        first_seen: When the first item of the story was seen.
        last_seen: When the latest item of the story was seen.
        item_count: Number of items in the story.
        source_count: Number of distinct sources reporting the story.
    """

    id: str
    centroid: list[float] | None = None
    first_seen: datetime = Field(default_factory=utcnow)
    last_seen: datetime = Field(default_factory=utcnow)
    item_count: int = 1
    source_count: int = 1


class Interest(BaseModel):
    """One thing the user cares about.

    Attributes:
        id: Unique interest id.
        label: Short name shown to the user.
        description: What the interest covers, in the user's words.
        keywords: Terms that signal a match.
        exclusions: Terms that rule a match out.
        weight: Relative importance of the interest.
        kind: One of 'topic', 'goal' or 'entity'.
        active: Whether the interest is in use.
        profile_version: Profile version the interest belongs to.
    """

    id: str
    label: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    weight: float = 1.0
    kind: str = "topic"
    active: bool = True
    profile_version: int = 1


class Judgment(BaseModel):
    """The judge model's verdict on an item for one profile version.

    Attributes:
        item_id: Id of the judged item.
        profile_version: Profile version the item was judged against.
        model: Name of the model that judged it.
        relevance: Relevance score per interest id.
        novelty: How new the content is, from 0.0 to 1.0.
        actionability: How much the user could act on it, from 0.0 to 1.0.
        type: Item type the judge assigned, e.g. 'update' or 'noise'.
        why: The judge's short reason.
        summary: Short summary of the item.
        lang: Language code of the item.
        judged_at: When the judgment was made.
    """

    item_id: str
    profile_version: int = 1
    model: str = "default"
    relevance: dict[str, float] = Field(default_factory=dict)
    novelty: float = 0.0
    actionability: float = 0.0
    type: str = "update"
    why: str = ""
    summary: str = ""
    lang: str = "en"
    judged_at: datetime = Field(default_factory=utcnow)


class ScoreBreakdown(BaseModel):
    """The components that make up an item's final score.

    Attributes:
        relevance: Match to the user's interests.
        novelty: How new the content is.
        quality: Quality of the item and its source.
        momentum: How fast its story is spreading.
        actionability: How much the user could act on it.
        decay: Time-decay factor that multiplies the weighted sum.
    """

    relevance: float = 0.0
    novelty: float = 0.0
    quality: float = 0.0
    momentum: float = 0.0
    actionability: float = 0.0
    decay: float = 1.0


class Score(BaseModel):
    """An item's ranking score for one profile version.

    Attributes:
        item_id: Id of the scored item.
        profile_version: Profile version the score was computed for.
        components: The score's breakdown.
        final: The final score used for ranking.
        computed_at: When the score was computed.
    """

    item_id: str
    profile_version: int = 1
    components: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    final: float = 0.0
    computed_at: datetime = Field(default_factory=utcnow)


class Opportunity(BaseModel):
    """An item with an action and usually a deadline (CFP, grant, hackathon).

    Attributes:
        id: Unique opportunity id.
        item_id: Id of the item it was found in.
        type: Opportunity type, e.g. 'cfp' or 'grant'.
        deadline: When it closes, when known.
        eligibility: Who can apply, when stated.
        link: Where to apply.
        amount: Prize or funding amount, as written.
        status: Lifecycle status, e.g. 'open'.
    """

    id: str
    item_id: str
    type: str
    deadline: datetime | None = None
    eligibility: str | None = None
    link: str | None = None
    amount: str | None = None
    status: str = "open"


class Trend(BaseModel):
    """A topic rising across independent sources.

    Attributes:
        id: Unique trend id.
        label: Short name of the trend.
        story_ids: Ids of the stories behind the trend.
        z_score: How far the surge stands above normal, in standard deviations.
        window: Time window the surge was measured over, e.g. '28d'.
        detected_at: When the trend was detected.
    """

    id: str
    label: str
    story_ids: list[str] = Field(default_factory=list)
    z_score: float = 0.0
    window: str = "28d"
    detected_at: datetime = Field(default_factory=utcnow)


class Question(BaseModel):
    """A standing question the user is working on (spec M17).

    Attributes:
        id: Unique question id.
        text: The question.
        active: Whether the question is shown and kept up to date.
        linked_interests: Ids of interests the question draws on.
        watch: Path of a dependency file to watch, e.g. 'pyproject.toml'.
        profile_version: Profile version the question belongs to.
        created_at: When the question was added.
    """

    id: str
    text: str
    active: bool = True
    linked_interests: list[str] = Field(default_factory=list)
    watch: str | None = None
    profile_version: int = 1
    created_at: datetime = Field(default_factory=utcnow)


class Claim(BaseModel):
    """One evidence statement from an item for a question.

    Attributes:
        id: Unique claim id.
        question_id: Id of the question it bears on.
        item_id: Id of the item it was taken from.
        text: The statement.
        stance: 'supports', 'contradicts' or 'context'.
        quote: Verbatim quote from the item that backs the statement.
        origin_key: Primary source the item is based on; claims sharing it are one piece of evidence.
        model: Name of the model that extracted it.
        created_at: When the claim was made.
    """

    id: str
    question_id: str
    item_id: str
    text: str
    stance: str = "supports"  # 'supports' | 'contradicts' | 'context'
    quote: str
    origin_key: str
    model: str = "default"
    created_at: datetime = Field(default_factory=utcnow)


class Sentence(BaseModel):
    """One sentence of a living answer with the claims it cites.

    Attributes:
        text: The sentence.
        claim_ids: Ids of the claims it cites.
    """

    text: str
    claim_ids: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    """A question's living answer at one version.

    Attributes:
        question_id: Id of the question it answers.
        version: Answer version, counting up from 1.
        sentences: The answer's cited sentences.
        confidence: 'insufficient', 'low', 'medium' or 'high'.
        counterpoint_claim_id: Id of the strongest claim contradicting the answer.
        what_changed: Summary of what changed since the previous version.
        what_changed_by: The model that wrote ``what_changed``; None when code wrote it.
        synthesized_by: The model that synthesized the answer, e.g. 'muse-spark-1.1'.
        created_at: When this version was made.
    """

    question_id: str
    version: int = 1
    sentences: list[Sentence] = Field(default_factory=list)
    confidence: str = "insufficient"  # 'insufficient' | 'low' | 'medium' | 'high'
    counterpoint_claim_id: str | None = None
    what_changed: str | None = None
    what_changed_by: str | None = None  # the model that wrote `what_changed`; None when code wrote it
    synthesized_by: str | None = None  # the model that synthesized the living answer (e.g. 'muse-spark-1.1')
    created_at: datetime = Field(default_factory=utcnow)


class AnswerDiff(BaseModel):
    """What changed between two versions of a living answer.

    Also records which claims caused the change.

    Attributes:
        question_id: Id of the question.
        from_version: The older answer version.
        to_version: The newer answer version.
        added_sentences: Sentences only in the newer version.
        removed_sentences: Sentences only in the older version.
        what_changed: Summary of the change.
        new_claim_ids: Ids of the claims that caused the change.
    """

    question_id: str
    from_version: int
    to_version: int
    added_sentences: list[Sentence] = Field(default_factory=list)
    removed_sentences: list[Sentence] = Field(default_factory=list)
    what_changed: str = ""
    new_claim_ids: list[str] = Field(default_factory=list)


class TrackRecord(BaseModel):
    """One answer change and the user's verdict on it.

    Attributes:
        id: Unique record id.
        question_id: Id of the question.
        answer_version: Version of the answer that changed.
        verdict: 'useful', 'wrong' or 'already_knew'; None until the user rates it.
        hn_lead_days: Days the answer was ahead of Hacker News, when found there.
        created_at: When the record was made.
    """

    id: str
    question_id: str
    answer_version: int
    verdict: str | None = None  # 'useful' | 'wrong' | 'already_knew'
    hn_lead_days: float | None = None
    created_at: datetime = Field(default_factory=utcnow)
