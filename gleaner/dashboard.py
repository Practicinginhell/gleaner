# ruff: noqa: RUF001 - page copy uses typographic apostrophes and dashes on purpose
"""The local web dashboard: one server-rendered HTML page with inline CSS and JS.

The page opens on the Brief: what changed for the user's standing questions, with the
evidence quoted, then a short list of other items worth reading. The full feed, digest and
the rest are one tab away.

Every value from feeds, models or the database is untrusted: it is escaped with
`html.escape` here and with `esc()` in the page script, and links go through `_safe_href` /
`safeUrl()` so only http(s) URLs are rendered.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import quote, urlsplit

from gleaner.digest import Digest
from gleaner.feed import FeedCard
from gleaner.hn_lead import HNLead
from gleaner.llama import is_meta_model
from gleaner.models import Answer, Claim, Item, Question, SourceRow, Trend
from gleaner.questions import STACK_QUESTION_ID
from gleaner.scoring import RANKING_MODES

KIND_LABELS = frozendict(paper="Paper", release="Release", post="Discussion", story="Discussion",
                         discussion="Discussion", article="Article", news="Article")
CATEGORY_OF_KIND = frozendict(paper="paper", release="release", post="discussion", story="discussion",
                              discussion="discussion", article="article", news="article")
SOURCE_TYPE_LABELS = frozendict(rss="RSS", hn="Hacker News", arxiv="arXiv", github="GitHub")
CONFIDENCE_LABELS = frozendict(high="High confidence", medium="Medium confidence", low="Low confidence",
                               insufficient="Unconfirmed")
CONFIDENCE_ORDER = ("insufficient", "low", "medium", "high")
# Reasons the keyword fallback writes; they repeat the score, so the feed hides them
FALLBACK_WHY_PREFIXES = ("Evaluated via fallback", "Matches your active interests")
QUOTES_SHOWN = 3


@dataclass
class QuestionView:
    """One standing question with what the page shows about it.

    Attributes:
        question: The standing question itself.
        latest: The newest living answer, if one was written.
        previous: The answer before ``latest``, used to show what changed.
        claims: Quoted evidence linked to the question.
        hit_rate: The user's ratings of the answers (``total``, ``useful``, ``hit_rate_pct``).
        items: The source items behind the claims, keyed by item id.
    """

    question: Question
    latest: Answer | None = None
    previous: Answer | None = None
    claims: list[Claim] = field(default_factory=list)
    hit_rate: dict[str, Any] = field(default_factory=dict)
    items: dict[str, Item] = field(default_factory=dict)

    @property
    def judged(self) -> bool:
        """True when a model judged the evidence or wrote the living answer.

        Judged means a claim has a stance other than "context", or the latest answer was
        synthesized by a model.
        """
        return any(c.stance != "context" for c in self.claims) or bool(self.latest and self.latest.synthesized_by)


def _safe_href(url: str | None) -> str:
    """Return the URL HTML-escaped if it is http(s), else '#'.

    So feed links can't run script.
    """
    if url and urlsplit(url.strip()).scheme.lower() in {"http", "https"}:
        return html.escape(url.strip())
    return "#"


def _e(value: object) -> str:
    """HTML-escape any value."""
    return html.escape(str(value))


def _short_date(dt: datetime | None) -> str:
    """'3 Nov' for a datetime, '' for None."""
    return f"{dt.day} {dt:%b}" if dt else ""


def _n(value: int) -> str:
    """A count with thousands separators."""
    return f"{value:,}"


def _plural(count: int, noun: str) -> str:
    """'1 finding', '2 findings': counts the page shows next to a noun."""
    return f"{_n(count)} {noun}{'' if count == 1 else 's'}"


# ---------------------------------------------------------------- feed


def render_card(c: FeedCard) -> str:
    """Render one feed item as an HTML card.

    `renderCardHtml` in the page script must produce the same markup.

    Args:
        c: The ranked feed card to render.

    Returns:
        The card's ``<article>`` HTML, with every untrusted value escaped.
    """
    item_id = _e(c.item.id)
    kind = (c.item.kind or "article").lower()
    title = _e(c.item.title)
    url = _safe_href(c.item.url)
    p = c.components
    meta = [f'<span class="src">{_e(c.item.source_id or "unknown")}</span>',
            f"<span>{_e(KIND_LABELS.get(kind, kind.title()))}</span>"]
    if c.item.published_at:
        meta.append(f'<time datetime="{_e(c.item.published_at.isoformat())}">{_short_date(c.item.published_at)}</time>')
    badges = ""
    if c.exploring:
        badges += '<span class="pill pill-tint" title="Just outside your interests">Exploring</span>'
    if c.also_covered_by:
        badges += f'<span class="pill" title="Also covered by other sources">+{len(c.also_covered_by)} sources</span>'
    why = "" if c.why.startswith(FALLBACK_WHY_PREFIXES) else f'<p class="card-why">{_e(c.why)}</p>'
    summary = f'<p class="card-summary">{_e(c.summary)}</p>' if c.summary and c.summary != c.item.title else ""
    perspectives = ""
    if c.perspectives:
        rows = "".join(f'<div><dt>{_e(k.replace("_", " ").title())}</dt><dd>{_e(v)}</dd></div>' for k, v in c.perspectives.items())
        perspectives = f'<dl class="perspectives">{rows}</dl>'
    breakdown = (f"Score {c.score:.2f}: relevance {p.relevance:.2f}, novelty {p.novelty:.2f}, quality {p.quality:.2f}, "
                 f"momentum {p.momentum:.2f}, actionability {p.actionability:.2f}")
    return f"""
<article class="feed-card" id="card-{item_id}" data-id="{item_id}" data-kind="{_e(kind)}" data-category="{CATEGORY_OF_KIND.get(kind, "article")}">
  <div class="card-main">
    <div class="card-meta">{'<span aria-hidden="true">·</span>'.join(meta)}{badges}</div>
    <h3 class="card-title"><a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a></h3>
    {why}{summary}{perspectives}
  </div>
  <div class="card-side">
    <button type="button" class="score" title="{breakdown}" data-score="{c.score:.2f}" data-rel="{p.relevance:.2f}" data-nov="{p.novelty:.2f}" data-qual="{p.quality:.2f}" data-mom="{p.momentum:.2f}" data-act="{p.actionability:.2f}" data-title="{title}" data-why="{_e(c.why)}" onclick="inspectScore(this)">Why this?</button>
    <div class="card-actions">
      <button type="button" class="act" data-id="{item_id}" data-action="like" aria-pressed="false" aria-label="More like this" title="More like this" onclick="submitFeedback(this.dataset.id, this.dataset.action, this)">{ICON_UP}</button>
      <button type="button" class="act" data-id="{item_id}" data-action="dislike" aria-pressed="false" aria-label="Less like this" title="Less like this" onclick="submitFeedback(this.dataset.id, this.dataset.action, this)">{ICON_DOWN}</button>
    </div>
  </div>
</article>"""


# ---------------------------------------------------------------- brief


def _new_today(view: QuestionView, today: date) -> list[Claim]:
    """The question's claims created today, in local time."""
    return [c for c in view.claims if c.created_at.astimezone().date() == today]


def _status_pill(view: QuestionView, today: date) -> str:
    """The pill by a question's title: confidence, new today, all clear or no evidence.

    Returns '' when there are claims but none new, since the count is in the meta line.
    """
    if view.judged and view.latest:
        conf = view.latest.confidence.lower()
        tone = {"high": "pill-green", "medium": "pill-tint", "low": "pill-orange"}.get(conf, "")
        return f'<span class="pill {tone}">{_e(CONFIDENCE_LABELS.get(conf, conf.title()))}{_confidence_move(view)}</span>'
    new = _new_today(view, today)
    if new:
        return f'<span class="pill pill-tint">{len(new)} new today</span>'
    if view.claims:
        return ""  # the count is in the line below the title
    if view.question.id == STACK_QUESTION_ID:
        return '<span class="pill pill-green">All clear</span>'
    return '<span class="pill">No evidence yet</span>'


def _confidence_move(view: QuestionView) -> str:
    """'↑ from low' when confidence moved since the previous answer, else nothing."""
    if view.previous is None or view.latest is None:
        return ""
    before, now = view.previous.confidence.lower(), view.latest.confidence.lower()
    if before == now or before not in CONFIDENCE_ORDER or now not in CONFIDENCE_ORDER:
        return ""
    arrow = "↑" if CONFIDENCE_ORDER.index(now) > CONFIDENCE_ORDER.index(before) else "↓"
    return f'<span class="pill-move">{arrow} from {CONFIDENCE_LABELS[before].removesuffix(" confidence").lower()}</span>'


def _hit_rate_text(hit_rate: dict[str, Any]) -> str:
    """'Your ratings: 3 useful · 1 knew it', or 'No ratings yet'.

    Counts, not a percent: the hit rate leaves 'knew it' out, so "3 of 4 (100%)" read as wrong.
    """
    parts = [f"{hit_rate.get(key, 0)} {label}" for key, label in RATING_LABELS if hit_rate.get(key)]
    return f"Your ratings: {' · '.join(parts)}" if parts else "No ratings yet"


RATING_LABELS = (("useful", "useful"), ("wrong", "wrong"), ("already_knew", "knew it"))


def _quote(claim: Claim, item: Item | None) -> str:
    """One piece of evidence: the verbatim quote and where it came from."""
    if item is not None:
        when = f" · {_short_date(item.published_at)}" if item.published_at else ""
        src = (f'<a href="{_safe_href(item.url)}" target="_blank" rel="noopener noreferrer">{_e(item.title)}</a>'
               f'<span class="q-src-meta">{_e(item.source_id or "")}{when}</span>')
    else:
        src = f'<span class="q-src-meta">{_e(claim.origin_key)}</span>'
    return f'<li id="claim-{_e(claim.id)}"><blockquote>“{_e(claim.quote)}”</blockquote><div class="q-src">{src}</div></li>'


def _judged_answer(view: QuestionView) -> str:
    """Render a model-written answer.

    Changed sentences are highlighted, every sentence is cited, and the counterpoint is
    shown. Citations are numbered in order of first use, and the quotes list follows that order.

    Args:
        view: The question; its ``latest`` answer is rendered against ``previous``.

    Returns:
        The answer HTML, or '' when there is no answer with sentences.
    """
    latest = view.latest
    if latest is None or not latest.sentences:
        return ""
    claims = {c.id: c for c in view.claims}
    order: list[str] = []

    def cite(ids: list[str]) -> str:
        for cid in ids:
            if cid not in order:
                order.append(cid)
        return "".join(f'<a class="cite" href="#claim-{_e(cid)}">{order.index(cid) + 1}</a>' for cid in ids)

    now = {s.text for s in latest.sentences}
    # the answer reads as an answer: the old wording is one tap away, not crossed out above it
    removed = " ".join(f"<del>{_e(s.text)}</del>" for s in (view.previous.sentences if view.previous else []) if s.text not in now)
    body = " ".join(_e(s.text) + cite(s.claim_ids) for s in latest.sentences)
    changed = f'<p class="changed"><strong>What changed:</strong> {_e(latest.what_changed)}</p>' if latest.what_changed else ""
    old = (f'<details class="old-wording"><summary>Show the old wording</summary><p>{removed}</p></details>'
           if removed else "")
    counter = ""
    if latest.counterpoint_claim_id in claims:
        cc = claims[latest.counterpoint_claim_id]
        counter = f'<p class="counter"><strong>Strongest counterpoint:</strong> {_e(cc.text)}{cite([cc.id])}</p>'
    notes = "".join(_quote(claims[cid], view.items.get(claims[cid].item_id)) for cid in order if cid in claims)
    return f'<p class="answer">{body}</p>{changed}{old}{counter}<ol class="quotes">{notes}</ol>'


def render_question(view: QuestionView, today: date) -> str:
    """Render a standing question for the Brief.

    Shows the answer when a model wrote one, else the newest quoted evidence.

    Args:
        view: The question and what the page shows about it.
        today: The user's local date, to tell what is new today.

    Returns:
        The question card's ``<article>`` HTML.
    """
    q = view.question
    qid = _e(q.id)
    # websites, not papers: 16 arXiv papers from one feed read as "16 sources"
    websites = {(view.items[c.item_id].source_id if c.item_id in view.items else c.origin_key) for c in view.claims}
    if view.judged:
        body = _judged_answer(view)
        meta = f"{_plural(len(view.claims), 'quote')} from {_plural(len(websites), 'website')}"
    elif view.claims:
        newest = sorted(view.claims, key=lambda c: c.created_at, reverse=True)
        shown = "".join(_quote(c, view.items.get(c.item_id)) for c in newest[:QUOTES_SHOWN])
        rest = newest[QUOTES_SHOWN:]
        more = (f'<details class="more"><summary>Show {len(rest)} more</summary><ol class="quotes">'
                + "".join(_quote(c, view.items.get(c.item_id)) for c in rest) + "</ol></details>") if rest else ""
        takeaway = ""
        if view.latest and view.latest.what_changed_by:
            takeaway = (f'<p class="takeaway">{_e(view.latest.what_changed)}</p>'
                        f'<p class="byline">Summary by {_e(view.latest.what_changed_by)} from the quotes below</p>')
        body = f'{takeaway}<p class="label">What the websites say</p><ol class="quotes">{shown}</ol>{more}'
        if q.id == STACK_QUESTION_ID:
            n = len(view.claims)
            meta = f"{n} advisor{'ies' if n != 1 else 'y'} for the versions installed here · from osv.dev"
        else:
            checked = [c for c in view.claims if c.model != "heuristic"]
            by = ", ".join(sorted({c.model for c in checked}))
            if not checked:
                how = "found by word match, not checked by Llama yet"
            elif len(checked) == len(view.claims):
                how = f"checked by {by}"
            else:
                how = f"{len(checked)} checked by {by}, the rest found by word match"
            meta = f"{_plural(len(view.claims), 'quote')} from {_plural(len(websites), 'website')} · {how}"
    elif q.id == STACK_QUESTION_ID:
        body = ('<p class="answer-empty">No known advisories affect the versions installed here. '
                "gleaner checks OSV once a day.</p>")
        meta = f"Watching {q.watch}" if q.watch else ""
    else:
        body = ('<p class="answer-empty">Nothing yet. gleaner looks for this in everything it collects and '
                "quotes what it finds here. Choose Collect to look now.</p>")
        meta = ""
    return f"""
<article class="q-card" id="question-{qid}">
  <header class="q-head">
    <h2>{_e(q.text)}</h2>
    {_status_pill(view, today)}
  </header>
  <p class="q-meta">{_e(meta)}</p>
  {body}
  {_question_footer(view, qid)}
</article>"""


def _question_footer(view: QuestionView, qid: str) -> str:
    """Rating and answer actions when the card has evidence to judge, and Remove on every card."""
    remove = (f'<button type="button" class="btn-plain q-remove" data-qid="{qid}" onclick="removeQuestion(this.dataset.qid, this)" '
              'title="Take this question off the Brief">Remove</button>')
    find = (f'<button type="button" class="btn-plain q-find" data-qid="{qid}" onclick="findFeeds(this.dataset.qid, this)" '
            'title="Search the web for websites that write about this">Find websites</button>')
    if not view.claims:  # old ratings or answers on a card that now shows "Nothing yet" read as a bug
        return f'<footer class="q-foot q-foot-empty">{find}{remove}</footer>'
    hit = _hit_rate_text(view.hit_rate)
    write = (f'<button type="button" class="btn-plain q-write" data-qid="{qid}" onclick="synthesizeLivingAnswer(this.dataset.qid, this)" '
             'title="Write a cited answer from these quotes with the answer model">Write answer</button>') if view.claims else ""
    return f"""<footer class="q-foot">
    <span class="foot-note">Was this useful?</span>
    <div class="seg seg-sm" role="group" aria-label="Rate this answer">
      <button type="button" data-qid="{qid}" data-rating="useful" aria-pressed="false" title="This helped me" onclick="submitQuestionFeedback(this.dataset.qid, this.dataset.rating, this)">Useful</button>
      <button type="button" data-qid="{qid}" data-rating="wrong" aria-pressed="false" title="This is wrong" onclick="submitQuestionFeedback(this.dataset.qid, this.dataset.rating, this)">Wrong</button>
      <button type="button" data-qid="{qid}" data-rating="already_knew" aria-pressed="false" title="I already knew this" onclick="submitQuestionFeedback(this.dataset.qid, this.dataset.rating, this)">Knew it</button>
    </div>
    <span id="hitrate-{qid}" class="foot-note">{hit if view.hit_rate.get("total") else ""}</span>
    {write}<details class="q-more"><summary class="btn-plain" title="More actions">More</summary><div class="q-more-pop">{find}{remove}</div></details>
  </footer>"""


def _headline(questions: list[QuestionView], today: date) -> str:
    """The Brief's headline: changed answers, then new evidence, then a quiet fallback."""
    if not questions:
        return "Add the questions you’re working on."
    changed = [v for v in questions if v.judged and v.latest and v.latest.what_changed
               and v.latest.created_at.astimezone().date() == today]
    if changed:
        return f"{len(changed)} answer{'s' if len(changed) > 1 else ''} changed today."
    fresh = [v for v in questions if _new_today(v, today)]
    if fresh:
        if len(fresh) == len(questions):
            return "New quotes for all your questions."
        return f"New quotes for {len(fresh)} of your {len(questions)} questions."
    if any(v.claims for v in questions):
        return "Nothing new on your questions today."
    return "No answers yet."


def _also_reading(cards: list[FeedCard], evidence_ids: set[str]) -> str:
    """Up to four top feed items not already quoted as evidence, or '' if none."""
    rows = [c for c in cards if c.item.id not in evidence_ids][:4]
    if not rows:
        return ""
    items = "".join(
        f'<li><a href="{_safe_href(c.item.url)}" target="_blank" rel="noopener noreferrer">{_e(c.item.title)}</a>'
        f'<span class="row-sub">{_e(c.item.source_id or "")} · {_e(KIND_LABELS.get((c.item.kind or "article").lower(), "Article"))}</span></li>'
        for c in rows
    )
    return (f'<section class="side-card" aria-labelledby="also-h"><h3 id="also-h">Also worth reading</h3>'
            f'<ol class="reading">{items}</ol>'
            f'<button type="button" class="btn-plain" onclick="switchTab(\'feed\')">Open the feed</button></section>')


def _lead_text(delta: timedelta) -> str:
    """'5 hours' under a day, else '2 days'."""
    hours = delta.total_seconds() / 3600
    if hours < 24:
        n = max(1, round(hours))
        return f"{n} hour{'s' if n != 1 else ''}"
    n = round(hours / 24)
    return f"{n} day{'s' if n != 1 else ''}"


def _hn_card(leads: list[HNLead]) -> str:
    """Items the user's sources had before anyone posted them to Hacker News."""
    if not leads:
        return ""
    rows = "".join(
        f'<li><a href="{_safe_href(lead.item.url)}" target="_blank" rel="noopener noreferrer">{_e(lead.item.title)}</a>'
        f'<span class="row-sub">{_lead_text(lead.lead)} before it was posted · {_n(lead.points)} points · '
        f'<a href="{_safe_href(lead.hn_url)}" target="_blank" rel="noopener noreferrer">HN thread</a></span></li>'
        for lead in leads
    )
    return (f'<section class="side-card" aria-labelledby="hn-h"><h3 id="hn-h">Ahead of Hacker News</h3>'
            f'<p class="side-note">Hacker News is a popular tech news site. Your websites had these stories first.</p>'
            f'<ol class="reading">{rows}</ol></section>')


def _collect_details(telem: dict[str, Any] | None, llama_running: bool, sources: list[SourceRow], failing: list[SourceRow]) -> tuple[str, str]:
    """The last collect's key-value rows and website-health button."""
    if not telem or not telem.get("fetched"):
        rows = "<dt>Last collect</dt><dd><em>no data yet</em></dd>"
    else:
        judged = telem.get("judged_llama") or 0
        rows = (f"<dt>Collected</dt><dd>{_n(telem['fetched'])}</dd>"
                f"<dt>Matched your interests</dt><dd>{_n(telem.get('scored') or 0)}</dd>"
                f"<dt>Checked by Llama</dt><dd>{_n(judged) if llama_running or judged else 'Off'}</dd>"
                f"<dt>Blocked as unsafe</dt><dd>{_n(telem.get('blocked_prompt_guard') or 0)}</dd>")
        if judged and telem.get("compute_saved_pct") is not None:
            rows += f'<dt>Filtered before Llama</dt><dd>{telem["compute_saved_pct"]:.1f}%</dd>'
    ok = sum(1 for s in sources if s.enabled and not s.failures)
    enabled = sum(1 for s in sources if s.enabled)
    health = (f'<button type="button" class="btn-plain{" warn" if failing else ""}" onclick="switchTab(\'sources\')">'
              f"{ok} of {enabled} websites working</button>")
    return rows, health


def _impact_card(
    impact: dict[str, Any] | None,
    leads: list[HNLead],
    telem: dict[str, Any] | None,
    llama_running: bool,
    sources: list[SourceRow],
    failing: list[SourceRow],
) -> str:
    """Totals since the first fetch, with the last collect tucked underneath.

    The counts come from ``Database.impact_counts``.
    """
    if not impact or not impact.get("read"):
        return ""
    read = impact["read"]
    since = datetime.strptime(impact["since"], "%Y-%m-%d").strftime("%b %-d") if impact.get("since") else ""
    tiles = [
        _tile(_n(read), "items read"),
        _tile(_n(impact["kept"]), f"kept for you ({impact['kept'] / read:.0%})"),
        _tile(f"{impact['filtered'] / read:.0%}", "skipped as off-topic before any AI read it"),
    ]
    if leads:
        earliest = max(leads, key=lambda lead: lead.lead)
        tiles.append(_tile(_n(len(leads)), f"seen before Hacker News, up to {_lead_text(earliest.lead)} early"))
    else:
        tiles.append(_tile(_n(impact["quotes"]), "quotes on your questions"))
    checked = (f"{_n(impact['checked'])} checked by {', '.join(impact['checked_by'])}" if impact["checked"]
               else "none checked by a model yet")
    rows, health = _collect_details(telem, llama_running, sources, failing)
    return (f'<section class="side-card" aria-labelledby="impact-h"><h3 id="impact-h">What gleaner did for you</h3>'
            f'<p class="side-note">Since {since} · counted from your database</p>'
            f'<div class="tiles">{"".join(tiles)}</div>'
            f'<p class="side-note tiles-foot">{_plural(impact["quotes"], "quote")} on your questions · {checked}</p>'
            f'<details class="more side-more"><summary>Last collect</summary><dl class="kv">{rows}</dl></details>{health}</section>')


def _tile(figure: str, caption: str) -> str:
    """One figure with its caption in the impact card."""
    return f'<div class="tile"><span class="tile-num">{figure}</span><span class="tile-cap">{caption}</span></div>'


def _collect_card(telem: dict[str, Any] | None, llama_running: bool, sources: list[SourceRow], failing: list[SourceRow]) -> str:
    """The side card with the last collect's counts and a link to source health, shown before impact exists."""
    rows, health = _collect_details(telem, llama_running, sources, failing)
    return f'<section class="side-card" aria-labelledby="collect-h"><h3 id="collect-h">Last collect</h3><dl class="kv">{rows}</dl>{health}</section>'


# ---------------------------------------------------------------- other panels


def _source_target(s: SourceRow) -> str:
    """The website, repo or search query a source reads, for its subtitle."""
    if url := s.params.get("url"):
        return (urlsplit(str(url)).hostname or "").removeprefix("www.")
    return str(s.params.get("repo") or s.params.get("query") or "")


def _sources_rows(sources: list[SourceRow]) -> str:
    """One list row per source with status, cadence and actions, or an empty-state row."""
    rows = []
    for s in sources:
        sid = _e(s.id)
        if not s.enabled:
            status = '<span class="pill">Paused</span>'
        elif s.failures:
            status = f'<span class="pill pill-orange">Failing ({s.failures})</span>'
        else:
            status = '<span class="pill pill-green">OK</span>'
        minutes = s.cadence_s // 60
        cadence = f"every {minutes // 60} h" if minutes >= 60 and minutes % 60 == 0 else f"every {minutes} min"
        target = _source_target(s)
        sub_head = f"{_e(target)} · " if target else ""
        rows.append(f"""
<li class="list-row">
  <div class="row-text"><span class="row-title">{sid}</span><span class="row-sub">{sub_head}{_e(SOURCE_TYPE_LABELS.get(s.type, s.type))} · {cadence} · trust {s.trust:.0%}</span></div>
  {status}
  <div class="row-actions">
    <button type="button" class="btn-plain" data-sid="{sid}" onclick="fetchSourceNow(this.dataset.sid, this)">Fetch now</button>
    <button type="button" class="btn-plain" data-sid="{sid}" onclick="toggleSource(this.dataset.sid, {"false" if s.enabled else "true"}, this)">{"Pause" if s.enabled else "Resume"}</button>
    <button type="button" class="btn-plain" data-sid="{sid}" onclick="viewSourceLogs(this.dataset.sid)">Logs</button>
  </div>
</li>""")
    return "".join(rows) or '<li class="empty">No sources yet. Add them to config/sources.yaml.</li>'


OPPORTUNITY_LABELS = frozendict(cfp="Call for papers", grant="Grant", event="Event", hackathon="Hackathon")


def _opportunity_rows(opportunities: list[dict[str, Any]], today: date) -> str:
    """One list row per opportunity, with deadline urgency and a calendar link.

    A deadline that doesn't parse as a date is shown as given.
    """
    rows = []
    for op in opportunities:
        urgency = ""
        title = op.get("title") or str(op["type"]).title()
        extra_bits = []
        if op.get("amount"):
            extra_bits.append(str(op["amount"]))
        if op.get("eligibility"):
            extra_bits.append(str(op["eligibility"]))
        sub_desc = f" · {' · '.join(extra_bits)}" if extra_bits else ""
        due = "No deadline given"
        if op["deadline"]:
            due = f"Due {op['deadline']}"
            try:
                day = datetime.fromisoformat(str(op["deadline"]).split("T")[0]).date()
                due = f"Due {day.day} {day:%b %Y}"
                days = (day - today).days
                if days < 0:
                    urgency = '<span class="pill">Expired</span>'
                elif days <= 1:
                    urgency = f'<span class="pill pill-red">{"Due today" if days == 0 else "1 day left"}</span>'
                else:
                    urgency = f'<span class="pill {"pill-orange" if days <= 7 else ""}">{days} days left</span>'
            except ValueError, TypeError:
                urgency = f'<span class="pill">{_e(op["deadline"])}</span>'
        ics = (f'<a class="btn-plain" href="/api/opportunities/{_e(quote(op["id"], safe=""))}/ics" download>Add to Calendar</a>'
               if op["deadline"] else "")
        link_btn = (f'<a class="btn-plain" href="{_safe_href(op["link"])}" target="_blank" rel="noopener noreferrer">Details</a>'
                    if op.get("link") else "")
        rows.append(f"""
<li class="list-row">
  <div class="row-text"><span class="row-title">{_e(title)}</span><span class="row-sub">{_e(OPPORTUNITY_LABELS.get(str(op["type"]), str(op["type"]).title()))} · {_e(due)}{_e(sub_desc)}</span></div>
  {urgency}<div class="row-actions">{link_btn}{ics}</div>
</li>""")
    return "".join(rows)


def _collected_text(last_collect: str | None, today: date) -> str:
    """'Collected today at 01:59' in the user's local time (stored times are UTC)."""
    if not last_collect:
        return "Not collected yet"
    try:
        when = datetime.fromisoformat(last_collect.replace("Z", "+00:00"))
    except ValueError:
        return f"Collected {last_collect[:16]}"
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    local = when.astimezone()
    day = "today" if local.date() == today else _short_date(local)
    return f"Collected {day} at {local:%H:%M}"


def _digest_heading(digest: Digest) -> str:
    """'Top 5 for Wednesday, 4 November' for the default title; a model-written title as given."""
    if not digest.title.startswith("Daily Briefing"):
        return digest.title
    try:
        day = date.fromisoformat(digest.date)
    except ValueError:
        return digest.title
    return f"Top {len(digest.bullets) or 5} for {day:%A}, {day.day} {day:%B}"


def _empty(title: str, body: str) -> str:
    """An empty-state block; callers pass trusted, static copy."""
    return f'<div class="empty"><p>{title}</p><p>{body}</p></div>'


# ---------------------------------------------------------------- page


def render_dashboard(
    *,
    cards: list[FeedCard],
    digest: Digest,
    trends: list[Trend],
    sources: list[SourceRow],
    opportunities: list[dict[str, Any]],
    quarantine: list[dict[str, Any]],
    questions: list[QuestionView],
    telemetry: dict[str, Any] | None,
    last_collect: str | None,
    judge_model: str | None,
    today: date,
    hn_leads: list[HNLead],
    impact: dict[str, Any] | None = None,
    digest_items: dict[str, Item] | None = None,
) -> str:
    """Render the whole dashboard page.

    Args:
        cards: The ranked feed.
        digest: Today's digest.
        trends: Current trends.
        sources: Configured sources with their health.
        opportunities: Opportunities with deadlines.
        quarantine: Items blocked as prompt injection.
        questions: The standing questions for the Brief.
        telemetry: Counts from the last collect, or None before the first one.
        last_collect: When the last collect ran (ISO time, UTC), or None.
        judge_model: The model that judged the last collect, or None.
        today: The user's local date.
        hn_leads: Items the user's sources had before Hacker News.
        impact: Totals since the first fetch (``Database.impact_counts``), or None.
        digest_items: The digest's cited items by id, to link each point to its item.

    Returns:
        The complete HTML document, with CSS and JS inline.
    """
    telem = telemetry or {"blocked_prompt_guard": len(quarantine)}
    failing = [s for s in sources if s.enabled and s.failures]

    if telem.get("judge_degraded"):
        judge_text = "Llama unavailable: ranked by similarity"
    elif telemetry and telemetry.get("fetched") and not telemetry.get("judged_llama"):
        # the model name comes from any past judgment; the last collect may have had none
        judge_text = "Llama didn’t check the last collect"
    elif judge_model:
        judge_text = f"Checked by {judge_model}"
    else:
        judge_text = "Not checked by a model yet"
    llama_running = (bool(judge_model) and is_meta_model(judge_model) and not telem.get("judge_degraded")
                     and not (telemetry and telemetry.get("fetched") and not telemetry.get("judged_llama")))
    collected = _collected_text(last_collect, today)
    status_bits = [collected, judge_text]
    if failing:
        status_bits.append(f"{len(failing)} source{'s' if len(failing) > 1 else ''} failing")
    status_html = '<span aria-hidden="true">·</span>'.join(f"<span>{_e(b)}</span>" for b in status_bits)

    # Brief
    headline = _headline(questions, today)
    if telemetry and telemetry.get("fetched"):
        lede = f"{_n(telemetry.get('scored') or 0)} of the {_n(telemetry['fetched'])} items in the last collect matched your interests."
    else:
        lede = "Choose Collect to read your sources."
    writers = sorted({v.latest.what_changed_by for v in questions if v.latest and v.latest.what_changed_by and not v.judged})
    if questions and not llama_running and writers:
        lede += f" {', '.join(writers)} sums up what’s new. Sources are still matched by keyword."
    elif questions and not llama_running:
        lede += " Llama is off, so gleaner quotes matching sources instead of writing answers."
    questions_sorted = sorted(questions, key=lambda v: (len(_new_today(v, today)), len(v.claims)), reverse=True)
    questions_html = "".join(render_question(v, today) for v in questions_sorted) or _empty(
        "No standing questions yet.", "Choose Add question to start.")
    evidence_ids = {c.item_id for v in questions for c in v.claims}

    # Feed
    counts = {cat: sum(1 for c in cards if CATEGORY_OF_KIND.get((c.item.kind or "article").lower(), "article") == cat)
              for cat in ("paper", "release", "discussion", "article")}
    category_chips = "".join(
        f'<button type="button" class="chip" data-category="{cat}" aria-pressed="false" onclick="filterByCategory(this.dataset.category, this)">{label} <span>{counts[cat]}</span></button>'
        for cat, label in (("paper", "Papers"), ("release", "Releases"), ("discussion", "Discussions"), ("article", "Articles"))
    )
    cards_html = "".join(render_card(c) for c in cards) or _empty("No items yet.", "Choose Collect to read your sources.")

    # Digest, trends, deadlines, quarantine
    if digest.bullets:
        bullets = "".join(
            f'<li><p>{_e(b.text)}</p><div class="cites">'
            + "".join(f'<a href="{_safe_href(it.url)}" target="_blank" rel="noopener noreferrer">{_e(it.source_id or it.title)}</a>'
                      for cid in b.item_ids if (it := (digest_items or {}).get(cid)) is not None)
            + "</div></li>"
            for b in digest.bullets
        )
    else:
        bullets = '<li class="empty">Nothing in today’s digest yet. Choose Collect to gather items.</li>'
    trends_html = "".join(
        f'<li class="list-row"><div class="row-text"><span class="row-title">{_e(t.label)}</span>'
        f'<span class="row-sub">{_plural(len(t.story_ids), "item") if t.story_ids else "Picking up"} across your websites in the last {_e(t.window)}</span></div></li>'
        for t in trends
    )
    trends_html = f'<ul class="group">{trends_html}</ul>' if trends_html else _empty(
        "No trends yet.", "A trend appears when several independent sources pick up the same story at once.")
    opps = _opportunity_rows(opportunities, today)
    opps_html = f'<ul class="group">{opps}</ul>' if opps else _empty(
        "No deadlines yet.", "Grants, calls for papers and events with a deadline show up here, with a calendar reminder.")
    quarantine_html = "".join(
        f'<article class="q-card"><span class="pill pill-red">Blocked</span><h2>{_e(r["title"])}</h2>'
        f'<p class="q-meta">It tried to give the AI orders, so gleaner blocked it.</p><pre class="snippet">{_e((r["text"] or "")[:300])}</pre></article>'
        for r in quarantine
    ) or _empty("Nothing blocked.", "Text that tries to give the AI orders is stopped here before any AI reads it.")

    primary_tabs = (
        ("brief", "My questions", ""),
        ("feed", "For you", f'<span class="tab-n">{len(cards)}</span>'),
        ("opportunities", "Deadlines", f'<span class="tab-n">{len(opportunities)}</span>' if opportunities else ""),
        ("ask", "Ask", ""),
    )
    more_tabs = (
        ("digest", "Top 5", ""),
        ("trends", "Trending", f'<span class="tab-n">{len(trends)}</span>' if trends else ""),
        ("sources", "Websites", f'<span class="tab-n warn">{len(failing)} failing</span>' if failing else ""),
        ("quarantine", "Blocked", f'<span class="tab-n">{len(quarantine)}</span>' if quarantine else ""),
    )
    primary_html = "".join(
        f'<button type="button" id="tab-btn-{name}" class="tab-btn" role="tab" data-tab="{name}" aria-controls="tab-{name}" '
        f'aria-selected="{"true" if i == 0 else "false"}" tabindex="{0 if i == 0 else -1}" onclick="switchTab(this.dataset.tab, this)">{label}{badge}</button>'
        for i, (name, label, badge) in enumerate(primary_tabs)
    )
    more_html = "".join(
        f'<button type="button" id="tab-btn-{name}" class="tab-btn" role="tab" data-tab="{name}" aria-controls="tab-{name}" '
        f'aria-selected="false" tabindex="-1" onclick="switchTab(this.dataset.tab, this)">{label}{badge}</button>'
        for name, label, badge in more_tabs
    )
    more_warn = f'<span class="tab-n warn">{len(failing)}</span>' if failing else ""
    tab_buttons = f'{primary_html}<details id="nav-more" class="nav-more"><summary class="tab-btn"><span id="nav-more-label">More</span>{more_warn}</summary><div class="nav-more-pop">{more_html}</div></details>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>gleaner</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%230066cc'/%3E%3Ctext x='16' y='23' font-family='sans-serif' font-size='20' font-weight='700' text-anchor='middle' fill='white'%3Eg%3C/text%3E%3C/svg%3E">
<script>
try {{ var t = localStorage.getItem('gleaner-theme'); if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t); }} catch (e) {{}}
try {{ if (localStorage.getItem('gleaner-welcome-done')) document.documentElement.setAttribute('data-welcome', 'done'); }} catch (e) {{}}
</script>
<style>{CSS}</style>
</head>
<body>
<header class="bar">
  <div class="bar-inner">
    <span class="wordmark">gleaner</span>
    <nav class="tabs" role="tablist" aria-label="Views">{tab_buttons}</nav>
    <div class="bar-actions">
      <button type="button" id="btn-guide" class="icon-btn" onclick="openGuideModal()" aria-label="How gleaner works" title="How gleaner works (?)">{ICON_HELP}</button>
      <button type="button" id="theme-toggle-btn" class="icon-btn" onclick="toggleTheme()" aria-label="Switch appearance" title="Switch appearance">{ICON_THEME}</button>
      <button type="button" id="btn-voice" class="btn-secondary voice-btn" onclick="toggleVoice()" title="Say a command or ask a question (V)">{ICON_MIC}<span>Talk</span></button>
      <button type="button" id="btn-collect" class="btn-primary" title="Read all your websites now" onclick="triggerCollect(this)">{ICON_REFRESH}<span>Collect</span></button>
    </div>
  </div>
</header>

<main class="page">
  <section id="tab-brief" class="tab-content" role="tabpanel" aria-labelledby="tab-btn-brief">
    <div class="brief-grid">
      <div class="brief-main">
        <section id="welcome" class="welcome" aria-label="What gleaner does">
          <p class="welcome-lead">gleaner reads {_plural(sum(1 for s in sources if s.enabled), "website")} for you and finds the sentences that answer your questions.</p>
          <ol class="welcome-steps">
            <li><strong>Add a question</strong> you want to keep up with.</li>
            <li>Press <strong>Collect</strong> to read the websites.</li>
            <li><strong>Read</strong> what they say, then tap Useful or Wrong.</li>
          </ol>
          <button type="button" class="btn-secondary" onclick="dismissWelcome()">Got it</button>
        </section>
        <p class="eyebrow">{today.strftime("%A")}, {today.day} {today:%B}</p>
        <h1>{_e(headline)}</h1>
        <p class="lede">{_e(lede)}</p>
        <button type="button" class="status-line" onclick="openEnvDiagnostics()" title="System status">
          <span class="dot{" dot-warn" if failing or not llama_running else ""}" aria-hidden="true"></span>{status_html}
        </button>
        <div class="panel-row" style="margin-top: 28px; align-items: center;">
          <h2 class="section-title">Your questions</h2>
          <div class="section-actions">
            <a href="/api/export/dossier" download class="btn-plain" title="Download every question, answer and quote as Markdown (Obsidian, Notion)">Export</a>
            <button type="button" id="btn-add-question" class="btn-secondary" onclick="openAddQuestionModal()">Add question</button>
          </div>
        </div>
        <div id="questions-container" class="stack" style="margin-top: 14px;">{questions_html}</div>
      </div>
      <aside class="brief-side">
        <section class="side-card" aria-labelledby="next-h">
          <h3 id="next-h">What to do next</h3>
          <p class="side-note">Up to three concrete actions from your recent quotes, deadlines and open questions.</p>
          <div id="next-steps-body"></div>
          <button type="button" id="btn-next-steps" class="btn-secondary" onclick="suggestNextSteps(this)">Suggest next steps</button>
        </section>
        {_impact_card(impact, hn_leads, telemetry, llama_running, sources, failing) or _collect_card(telemetry, llama_running, sources, failing)}
        {_hn_card(hn_leads)}
        {_also_reading(cards, evidence_ids)}
      </aside>
    </div>
  </section>

  <section id="tab-feed" class="tab-content" role="tabpanel" aria-labelledby="tab-btn-feed" hidden>
    <div class="feed-grid">
      <div class="feed-tools">
        <label class="search">{ICON_SEARCH}<span class="sr-only">Search the feed</span>
          <input id="feed-search" type="search" placeholder="Search" autocomplete="off" oninput="filterFeed(this.value)">
        </label>
        <p class="label">Rank by</p>
        <div class="seg seg-col" role="group" aria-label="Rank by">
          <button type="button" class="mode-pill" data-mode="balanced" aria-pressed="true" onclick="setRankingMode(this.dataset.mode, this)">Best match</button>
          <button type="button" class="mode-pill" data-mode="deep_research" aria-pressed="false" onclick="setRankingMode(this.dataset.mode, this)">Newest ideas</button>
          <button type="button" class="mode-pill" data-mode="breaking_radar" aria-pressed="false" onclick="setRankingMode(this.dataset.mode, this)">Popular now</button>
          <button type="button" class="mode-pill" data-mode="actionable" aria-pressed="false" onclick="setRankingMode(this.dataset.mode, this)">Things to do</button>
        </div>
        <p id="mode-desc" class="foot-note">What matches your interests best.</p>
        <p class="label">Show</p>
        <div class="chip-row" role="group" aria-label="Show">{category_chips}</div>
        <p class="foot-note exports">Export <a href="/api/export/feed?format=json" download>JSON</a> · <a href="/api/export/feed?format=csv" download>CSV</a></p>
      </div>
      <div class="feed-list">
        <p class="foot-note feed-intro">Everything gleaner kept for you, best match first.</p>
        <p id="filter-count" class="foot-note" aria-live="polite"></p>
        <div id="cards-container" class="group cards">{cards_html}</div>
      </div>
    </div>
  </section>

  <section id="tab-digest" class="tab-content narrow" role="tabpanel" aria-labelledby="tab-btn-digest" hidden>
    <article class="digest">
      <p class="eyebrow">Download: <a href="/api/export/digest?format=markdown" download>Markdown</a> · <a href="/api/export/digest?format=json" download>JSON</a></p>
      <h1>{_e(_digest_heading(digest))}</h1>
      <p class="lede">{_e(digest.summary)}</p>
      <ul class="digest-list">{bullets}</ul>
      <p class="foot-note">Each point is checked against the items it cites. Points without a citation are dropped.</p>
    </article>
  </section>

  <section id="tab-trends" class="tab-content narrow" role="tabpanel" aria-labelledby="tab-btn-trends" hidden>
    <h1 class="panel-title">Trending</h1>
    <p class="foot-note">Stories that several websites are talking about at the same time.</p>{trends_html}
  </section>

  <section id="tab-opportunities" class="tab-content narrow" role="tabpanel" aria-labelledby="tab-btn-opportunities" hidden>
    <div class="panel-row">
      <div>
        <h1 class="panel-title">Deadlines</h1>
        <p class="foot-note">Calls for papers, research grants, and deadlines with calendar export.</p>
      </div>
      <button type="button" id="btn-seed-deadlines" class="btn-secondary" onclick="seedSampleOpportunities(this)">Load Sample Deadlines</button>
    </div>
    {opps_html}
  </section>

  <section id="tab-ask" class="tab-content narrow" role="tabpanel" aria-labelledby="tab-btn-ask" hidden>
    <h1 class="panel-title">Ask gleaner</h1>
    <p class="foot-note">Ask about anything gleaner has read. It answers with links to where it found it. To ask out loud, press Talk and speak. gleaner reads the answer back.</p>
    <form class="ask-form" onsubmit="askQuestion(event)">
      <label class="search search-lg">{ICON_SEARCH}<span class="sr-only">Question</span>
        <input id="ask-input" type="text" placeholder="What changed in LangGraph this week?" autocomplete="off">
      </label>
      <button type="submit" class="btn-primary">Ask</button>
      <button type="button" id="btn-ask-voice" class="btn-secondary voice-btn" onclick="toggleVoice()" title="Ask out loud. gleaner reads the answer back.">{ICON_MIC}<span>Talk</span></button>
    </form>
    <div class="chip-row">
      <button type="button" class="chip" onclick="askSuggestion(this.textContent)">What are the latest updates on Graphflow and LangGraph?</button>
      <button type="button" class="chip" onclick="askSuggestion(this.textContent)">Summarize test-time compute scaling</button>
      <button type="button" class="chip" onclick="askSuggestion(this.textContent)">Which deadlines are coming up?</button>
    </div>
    <div id="ask-result" class="ask-result" aria-live="polite" hidden></div>
  </section>

  <section id="tab-sources" class="tab-content narrow" role="tabpanel" aria-labelledby="tab-btn-sources" hidden>
    <div class="panel-row">
      <div>
        <h1 class="panel-title">Websites</h1>
        <p class="foot-note">The websites gleaner reads for you.</p>
      </div>
      <button type="button" id="btn-add-source" class="btn-primary" onclick="openAddSourceModal()">+ Add website</button>
    </div>
    <ul class="group">{_sources_rows(sources)}</ul>
  </section>

  <section id="tab-quarantine" class="tab-content narrow" role="tabpanel" aria-labelledby="tab-btn-quarantine" hidden>
    <div class="panel-row"><div><h1 class="panel-title">Blocked</h1><p class="foot-note">Text that tried to give the AI orders. gleaner blocked it before any AI read it.</p></div>
      <button type="button" class="btn-secondary" onclick="simulateAttack(this)">Try a fake attack</button></div>
    <div id="quarantine-container" class="stack">{quarantine_html}</div>
  </section>
</main>

<footer class="page footer">
  <p>gleaner runs on this computer. Your data stays in one SQLite file. Built during the Meta Global AI Developer Hackathon, track “Build for Your World”.</p>
</footer>

<dialog id="guide-modal" class="sheet" aria-labelledby="guide-title">
  <header class="sheet-head"><h2 id="guide-title">How gleaner works</h2><button type="button" class="icon-btn" onclick="closeModal('guide-modal')" aria-label="Close">{ICON_CLOSE}</button></header>
  <div class="sheet-body guide">
    <p>You tell gleaner the questions you’re working on. It reads your sources, keeps what matches your interests, and quotes the sentences that answer each question. Llama checks each quote, and the answer model writes a cited answer from them and shows what changed since yesterday.</p>
    <h3>Every day</h3>
    <ol>
      <li><strong>Collect</strong> reads your websites, drops duplicates and off-topic items, and blocks text that tries to give the AI orders (prompt injection).</li>
      <li><strong>My questions</strong> shows new quotes for each question. Rate an answer <strong>Useful</strong>, <strong>Wrong</strong>, or <strong>Knew it</strong> if it told you nothing new. <strong>Write answer</strong> asks for a fresh answer now.</li>
      <li><strong>For you</strong> ranks everything else. More like this and Less like this tune your interests.</li>
    </ol>
    <h3>Score breakdown</h3>
    <dl class="kv">
      <dt>Relevance</dt><dd>Similarity to your interests, blended with Llama’s rating when it runs.</dd>
      <dt>Novelty</dt><dd>How different the item is from recent items.</dd>
      <dt>Quality</dt><dd>The source’s trust, adjusted by your likes and dislikes of that source.</dd>
      <dt>Momentum</dt><dd>Coverage by several sources, plus Hacker News points and comments, GitHub stars and citations.</dd>
      <dt>Actionability</dt><dd>Releases and opportunities first, then papers, blended with Llama’s rating when it runs.</dd>
    </dl>
    <h3>Voice</h3>
    <p>Press <strong>Talk</strong> and speak. It stops when you stop talking, or press Stop. Meta’s Muse Voice Transcribe turns what you said into text. Say “open deadlines”, “collect”, “read the top 5”, “add a question: …”, or ask anything, and gleaner reads the answer aloud.</p>
    <h3>Keyboard</h3>
    <p><kbd>/</kbd> search the feed · <kbd>1</kbd> to <kbd>8</kbd> switch views · <kbd>V</kbd> talk · <kbd>?</kbd> this guide · <kbd>Esc</kbd> close</p>
  </div>
</dialog>
<dialog id="env-modal" class="sheet" aria-labelledby="env-title">
  <header class="sheet-head"><h2 id="env-title">System status</h2><button type="button" class="icon-btn" onclick="closeModal('env-modal')" aria-label="Close">{ICON_CLOSE}</button></header>
  <div id="env-modal-body" class="sheet-body"></div>
</dialog>
<dialog id="logs-modal" class="sheet sheet-wide" aria-labelledby="logs-modal-title">
  <header class="sheet-head"><h2 id="logs-modal-title">Fetch log</h2><button type="button" class="icon-btn" onclick="closeModal('logs-modal')" aria-label="Close">{ICON_CLOSE}</button></header>
  <div id="logs-modal-body" class="sheet-body"></div>
</dialog>
<dialog id="add-source-modal" class="sheet" aria-labelledby="add-source-title">
  <header class="sheet-head"><h2 id="add-source-title">Add a website</h2><button type="button" class="icon-btn" onclick="closeModal('add-source-modal')" aria-label="Close">{ICON_CLOSE}</button></header>
  <form id="add-source-form" class="sheet-body form-sheet" onsubmit="submitAddSource(event)">
    <p class="foot-note" style="margin-top:0">Add a website for gleaner to read. It checks every new post and keeps the ones that match your interests.</p>
    <label class="form-label" for="source-type">What kind of website?</label>
    <select id="source-type" class="form-select" onchange="onSourceTypeChange(this.value)">
      <option value="rss">Blog or news site (its RSS feed)</option>
      <option value="github">A GitHub project's releases</option>
      <option value="hn">Hacker News search</option>
      <option value="arxiv">arXiv research papers</option>
    </select>
    <label class="form-label" for="source-id">Short name</label>
    <input id="source-id" class="form-input" type="text" placeholder="e.g. simonw-blog or langchain-ai" required autocomplete="off">
    <label class="form-label" for="source-target" id="source-target-label">Feed URL</label>
    <input id="source-target" class="form-input" type="text" placeholder="https://simonwillison.net/atom/everything/" required autocomplete="off">
    <div class="form-row-2">
      <div>
        <label class="form-label" for="source-cadence">How often to check</label>
        <select id="source-cadence" class="form-select">
          <option value="15">Every 15 min</option>
          <option value="30">Every 30 min</option>
          <option value="60" selected>Every 1 hour</option>
          <option value="360">Every 6 hours</option>
          <option value="1440">Every 24 hours</option>
        </select>
      </div>
      <div>
        <label class="form-label" for="source-trust">How much do you trust it?</label>
        <select id="source-trust" class="form-select">
          <option value="1.0">Completely (official docs)</option>
          <option value="0.8" selected>A lot (a team or expert blog)</option>
          <option value="0.6">Some (community discussion)</option>
          <option value="0.4">A little (experiments)</option>
        </select>
      </div>
    </div>
    <div class="form-actions">
      <button type="button" class="btn-plain" onclick="closeModal('add-source-modal')">Cancel</button>
      <button type="submit" class="btn-primary" id="btn-save-source">Add website</button>
    </div>
  </form>
</dialog>
<dialog id="add-question-modal" class="sheet" aria-labelledby="add-question-title">
  <header class="sheet-head"><h2 id="add-question-title">Add a question</h2><button type="button" class="icon-btn" onclick="closeModal('add-question-modal')" aria-label="Close">{ICON_CLOSE}</button></header>
  <form id="add-question-form" class="sheet-body form-sheet" onsubmit="submitAddQuestion(event)">
    <p class="foot-note" style="margin-top:0">gleaner looks for it in everything it collects, quotes the sentences that answer it, and tells you when the answer changes.</p>
    <label class="form-label" for="question-text">What do you want to keep up with?</label>
    <textarea id="question-text" class="form-textarea" rows="3" placeholder="e.g. What are the common failure modes of LangGraph in production?" required></textarea>
    <div class="form-actions">
      <button type="button" class="btn-plain" onclick="closeModal('add-question-modal')">Cancel</button>
      <button type="submit" class="btn-primary" id="btn-save-question">Add question</button>
    </div>
  </form>
</dialog>
<dialog id="feeds-modal" class="sheet" aria-labelledby="feeds-title">
  <header class="sheet-head"><h2 id="feeds-title">Websites for this question</h2><button type="button" class="icon-btn" onclick="closeModal('feeds-modal')" aria-label="Close">{ICON_CLOSE}</button></header>
  <div id="feeds-body" class="sheet-body"></div>
</dialog>
<dialog id="score-inspector-modal" class="sheet" aria-labelledby="score-inspector-title">
  <header class="sheet-head"><h2 id="score-inspector-title">Why this item is here</h2><button type="button" class="icon-btn" onclick="closeModal('score-inspector-modal')" aria-label="Close">{ICON_CLOSE}</button></header>
  <div id="score-inspector-body" class="sheet-body"></div>
</dialog>
<div id="toast" class="toast" role="status" aria-live="polite" hidden></div>
<script>{JS}</script>
</body>
</html>
"""


def _icon(path: str, size: int = 18) -> str:
    """An inline stroke SVG icon from its path markup."""
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{path}</svg>')


ICON_REFRESH = _icon('<path d="M20 12a8 8 0 1 1-2.34-5.66"/><path d="M20 4v5h-5"/>')
ICON_THEME = _icon('<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor"/>')
ICON_HELP = _icon('<circle cx="12" cy="12" r="8.5"/><path d="M9.6 9.5a2.5 2.5 0 1 1 3.6 2.2c-.8.4-1.2 1-1.2 1.8v.5"/><path d="M12 17h.01"/>')
ICON_SEARCH = _icon('<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>')
ICON_CLOSE = _icon('<path d="M6 6l12 12M18 6 6 18"/>')
ICON_MIC = _icon('<rect x="9" y="3.5" width="6" height="11" rx="3"/><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v2.5"/>')
ICON_UP = _icon('<path d="M7 10v10H4V10z"/><path d="M7 10l4-7a2 2 0 0 1 2 2v4h5.5a2 2 0 0 1 2 2.3l-1.2 7A2 2 0 0 1 17.3 20H7"/>', 16)
ICON_DOWN = _icon('<path d="M7 14V4H4v10z"/><path d="M7 14l4 7a2 2 0 0 0 2-2v-4h5.5a2 2 0 0 0 2-2.3l-1.2-7A2 2 0 0 0 17.3 4H7"/>', 16)

# Layout: a 1200px page. The Brief is a reading column with a side column of quiet context;
# the feed is a filter sidebar beside a one-column list. Apple system greys and type scale,
# one blue tint for actions, and a highlighter yellow reserved for what changed in an answer.
CSS = r"""
:root {
  color-scheme: light;
  --bg: #f5f5f7; --surface: #ffffff; --fill: rgba(118,118,128,.12); --fill-2: rgba(118,118,128,.07);
  --label: #1d1d1f; --label-2: #6e6e73; --label-3: #8e8e93; --sep: rgba(60,60,67,.14);
  --tint: #0066cc; --tint-strong: #0071e3; --tint-fill: rgba(0,113,227,.1); --on-tint: #fff;
  --green: #1d7d36; --green-fill: rgba(52,199,89,.14); --orange: #b25000; --orange-fill: rgba(255,149,0,.14);
  --red: #c4001a; --red-fill: rgba(255,59,48,.1); --mark: #fde68a; --mark-ink: inherit;
  --material: rgba(245,245,247,.8); --shadow: 0 1px 2px rgba(0,0,0,.04), 0 0 0 .5px rgba(0,0,0,.05);
  --shadow-pop: 0 24px 60px rgba(0,0,0,.18), 0 0 0 .5px rgba(0,0,0,.08);
  --font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-display: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-serif: "New York", ui-serif, Georgia, "Times New Roman", serif;
  --font-mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  --r: 16px; --r-sm: 10px; --ease: cubic-bezier(.32,.72,0,1); --page: 1200px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #000000; --surface: #1c1c1e; --fill: rgba(118,118,128,.24); --fill-2: rgba(118,118,128,.14);
    --label: #f5f5f7; --label-2: #a1a1a6; --label-3: #7c7c80; --sep: rgba(84,84,88,.55);
    --tint: #2997ff; --tint-strong: #0071e3; --tint-fill: rgba(41,151,255,.16);
    --green: #30d158; --green-fill: rgba(48,209,88,.16); --orange: #ff9f0a; --orange-fill: rgba(255,159,10,.16);
    --red: #ff453a; --red-fill: rgba(255,69,58,.16); --mark: rgba(255,214,10,.3); --mark-ink: #fff;
    --material: rgba(22,22,23,.8); --shadow: 0 0 0 .5px rgba(255,255,255,.08);
    --shadow-pop: 0 24px 60px rgba(0,0,0,.6), 0 0 0 .5px rgba(255,255,255,.1);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #000000; --surface: #1c1c1e; --fill: rgba(118,118,128,.24); --fill-2: rgba(118,118,128,.14);
  --label: #f5f5f7; --label-2: #a1a1a6; --label-3: #7c7c80; --sep: rgba(84,84,88,.55);
  --tint: #2997ff; --tint-strong: #0071e3; --tint-fill: rgba(41,151,255,.16);
  --green: #30d158; --green-fill: rgba(48,209,88,.16); --orange: #ff9f0a; --orange-fill: rgba(255,159,10,.16);
  --red: #ff453a; --red-fill: rgba(255,69,58,.16); --mark: rgba(255,214,10,.3); --mark-ink: #fff;
  --material: rgba(22,22,23,.8); --shadow: 0 0 0 .5px rgba(255,255,255,.08);
  --shadow-pop: 0 24px 60px rgba(0,0,0,.6), 0 0 0 .5px rgba(255,255,255,.1);
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--label); font: 15px/1.47 var(--font); letter-spacing: -.01em; -webkit-font-smoothing: antialiased; }
a { color: var(--tint); text-decoration: none; }
a:hover { text-decoration: underline; }
button { font: inherit; color: inherit; letter-spacing: inherit; cursor: pointer; }
h1, h2, h3 { margin: 0; font-family: var(--font-display); text-wrap: balance; }
:focus-visible { outline: 3px solid color-mix(in srgb, var(--tint) 55%, transparent); outline-offset: 2px; border-radius: 8px; }
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
[hidden] { display: none !important; }
.page { max-width: var(--page); margin-inline: auto; padding-left: max(24px, env(safe-area-inset-left)); padding-right: max(24px, env(safe-area-inset-right)); }

/* Top bar: wordmark, views, actions */
.bar { position: sticky; top: 0; z-index: 20; background: var(--material); backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px); border-bottom: .5px solid var(--sep); padding-top: env(safe-area-inset-top); }
.bar-inner { max-width: var(--page); margin-inline: auto; min-height: 56px; display: flex; align-items: center; gap: 16px;
  padding-left: max(24px, env(safe-area-inset-left)); padding-right: max(24px, env(safe-area-inset-right)); }
.wordmark { font: 700 21px/1 var(--font-display); letter-spacing: -.03em; flex: none; }
.tabs { flex: 1; min-width: 0; display: flex; align-items: center; gap: 2px; padding: 8px 0; }
.tab-btn { flex: none; display: inline-flex; align-items: center; gap: 6px; border: 0; border-radius: 980px; padding: 6px 12px;
  background: transparent; color: var(--label-2); font-size: 14px; font-weight: 500; list-style: none; cursor: pointer; transition: background .2s var(--ease), color .2s; }
.tab-btn::-webkit-details-marker { display: none; }
.tab-btn:hover { color: var(--label); background: var(--fill-2); }
.tab-btn[aria-selected="true"], .nav-more[data-active="true"] > summary { background: var(--fill); color: var(--label); font-weight: 600; }
.nav-more { position: relative; }
.nav-more-pop { position: absolute; top: calc(100% + 6px); left: 0; z-index: 30; min-width: 170px; padding: 6px;
  display: grid; gap: 2px; background: var(--surface); border-radius: 14px; box-shadow: var(--shadow-pop); }
.nav-more-pop .tab-btn { width: 100%; justify-content: space-between; border-radius: 10px; }
.tab-n { font-size: 12px; font-weight: 500; color: var(--label-2); font-variant-numeric: tabular-nums; }
.tab-n.warn { color: var(--orange); }
.bar-actions { display: flex; align-items: center; gap: 4px; flex: none; }
.icon-btn { width: 34px; height: 34px; display: inline-grid; place-items: center; border: 0; border-radius: 50%; background: transparent; color: var(--label-2); }
.icon-btn:hover { background: var(--fill); color: var(--label); }
.btn-primary { display: inline-flex; align-items: center; gap: 6px; border: 0; border-radius: 980px; background: var(--tint-strong);
  color: var(--on-tint); font-weight: 590; font-size: 14px; padding: 7px 16px; min-height: 34px; margin-left: 4px; transition: transform .15s var(--ease), filter .15s; }
.btn-primary:hover { filter: brightness(1.08); }
.btn-primary:active, .act:active, .chip:active, .btn-plain:active, .btn-secondary:active { transform: scale(.96); }
.btn-primary[disabled] { opacity: .6; cursor: progress; }
.btn-primary svg { width: 16px; height: 16px; }
.btn-primary.busy svg { animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.btn-secondary { border: 0; border-radius: 980px; background: var(--fill); color: var(--tint); font-weight: 590; font-size: 14px; padding: 8px 16px; }
.feed-pick { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 0; }
.feed-pick + .feed-pick { border-top: 1px solid var(--sep); }
.feed-pick .btn-secondary { flex: none; white-space: nowrap; }
.voice-btn { display: inline-flex; align-items: center; justify-content: center; gap: 6px; }
.voice-btn svg { width: 16px; height: 16px; }
.voice-btn.rec { background: var(--red-fill); color: var(--red); }
.voice-btn.rec svg { animation: pulse .8s ease-in-out infinite alternate; }
.voice-btn.busy { opacity: .6; cursor: progress; }
@keyframes pulse { to { opacity: .3; } }
.btn-plain { border: 0; background: transparent; color: var(--tint); font-size: 14px; font-weight: 500; padding: 6px 8px; border-radius: 8px; text-align: left; }
.btn-plain:hover { background: var(--tint-fill); }
.btn-plain.warn { color: var(--orange); }

/* Brief */
.tab-content { padding-block: 40px 24px; animation: fade .3s var(--ease); }
@keyframes fade { from { opacity: .3; transform: translateY(6px); } }
.narrow { max-width: 760px; }
.brief-grid { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 48px; align-items: start; }
.brief-main { min-width: 0; max-width: 760px; }
.eyebrow { margin: 0; color: var(--label-2); font-size: 13px; font-weight: 600; letter-spacing: .02em; text-transform: uppercase; }
.brief-main h1, .digest h1 { margin-top: 8px; font-size: clamp(34px, 5vw, 52px); line-height: 1.04; font-weight: 700; letter-spacing: -.04em; }
.lede { margin: 14px 0 0; max-width: 60ch; color: var(--label-2); font-size: 19px; line-height: 1.45; letter-spacing: -.02em; }
.status-line { margin-top: 14px; display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; border: 0; background: none;
  padding: 4px 0; color: var(--label-2); font-size: 13px; font-variant-numeric: tabular-nums; text-align: left; }
.status-line:hover { color: var(--label); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); }
.dot-warn { background: var(--orange); }
.stack { display: grid; gap: 20px; margin-top: 32px; }
.q-card { background: var(--surface); border-radius: var(--r); box-shadow: var(--shadow); padding: 24px 26px 18px; scroll-margin-top: 90px; }
.q-card:target { outline: 2px solid var(--tint); outline-offset: 2px; }
.q-remove { margin-left: auto; color: var(--label-2); }
.q-foot-empty { justify-content: flex-end; }
.q-card.flash { box-shadow: 0 0 0 2px var(--tint), var(--shadow); }
.q-head { display: flex; gap: 14px; align-items: flex-start; justify-content: space-between; }
.q-card h2 { font-size: 22px; line-height: 1.25; font-weight: 650; letter-spacing: -.025em; }
.q-meta { margin: 6px 0 0; color: var(--label-2); font-size: 13px; }
.label { margin: 20px 0 0; color: var(--label-2); font-size: 12px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }
.quotes { list-style: none; margin: 10px 0 0; padding: 0; display: grid; gap: 16px; }
.quotes blockquote { margin: 0; padding-left: 14px; border-left: 3px solid var(--fill); font: 17px/1.5 var(--font-serif); letter-spacing: -.005em; }
.q-src { margin-top: 6px; padding-left: 17px; display: flex; flex-wrap: wrap; gap: 2px 10px; font-size: 13px; }
.q-src a { font-weight: 500; }
.q-src-meta { color: var(--label-2); }
.more { margin-top: 14px; }
.more summary { cursor: pointer; color: var(--tint); font-size: 14px; font-weight: 500; list-style: none; }
.more summary::-webkit-details-marker { display: none; }
.more[open] summary { margin-bottom: 4px; }
.answer { margin: 16px 0 0; font: 19px/1.55 var(--font-serif); max-width: 62ch; }
.answer del { color: var(--label-2); text-decoration-thickness: 1.5px; }
.old-wording { margin: 8px 0 0; font-size: 14px; color: var(--label-2); }
.old-wording summary { cursor: pointer; color: var(--tint); font-weight: 500; }
.old-wording del { text-decoration-thickness: 1px; }
.answer-empty { margin: 16px 0 0; color: var(--label-2); }
.takeaway { margin: 16px 0 0; font: 19px/1.5 var(--font-serif); max-width: 62ch; text-wrap: pretty; }
.byline { margin: 4px 0 0; color: var(--label-2); font-size: 12px; }
.cite { font: 600 11px var(--font); vertical-align: super; line-height: 0; margin-left: 2px; }
.changed { margin: 14px 0 0; padding: 10px 14px; border-radius: var(--r-sm); background: var(--fill-2); font-size: 14px; }
.counter { margin: 12px 0 0; font-size: 14px; color: var(--label-2); }
.counter strong { color: var(--label); }
.q-foot { margin-top: 18px; padding-top: 12px; border-top: .5px solid var(--sep); display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; }
.q-foot .foot-note { margin: 0; }
.q-foot .q-write { margin-left: auto; font-size: 13px; }
.q-more { position: relative; }
.q-more summary { list-style: none; color: var(--label-2); font-size: 13px; }
.q-more summary::-webkit-details-marker { display: none; }
.q-more-pop { position: absolute; right: 0; bottom: calc(100% + 6px); z-index: 15; min-width: 160px; padding: 6px;
  display: grid; gap: 2px; background: var(--surface); border-radius: 12px; box-shadow: var(--shadow-pop); }
.q-more-pop .btn-plain { width: 100%; margin: 0; }
.side-more { margin-top: 10px; padding-top: 8px; border-top: .5px solid var(--sep); }
.side-more summary { font-size: 13px; color: var(--label-2); }

.brief-side { position: sticky; top: 80px; display: grid; gap: 16px; }
.side-card { background: var(--surface); border-radius: var(--r); box-shadow: var(--shadow); padding: 18px 18px 12px; }
.side-card h3 { font-size: 15px; font-weight: 650; letter-spacing: -.01em; margin-bottom: 8px; }
.side-card .btn-plain { margin: 6px 0 0 -8px; }
.reading { list-style: none; margin: 0; padding: 0; }
.reading li { padding: 10px 0; border-top: .5px solid var(--sep); display: flex; flex-direction: column; gap: 2px; }
.reading li:first-child { border-top: 0; padding-top: 2px; }
.reading a { color: var(--label); font-size: 14px; font-weight: 500; line-height: 1.35; }
.reading a:hover { color: var(--tint); text-decoration: none; }
.kv { margin: 0; display: grid; grid-template-columns: 1fr auto; }
.kv dt, .kv dd { margin: 0; padding: 7px 0; border-bottom: .5px solid var(--sep); font-size: 14px; }
.kv dt { color: var(--label-2); padding-right: 16px; }
.kv dd { text-align: right; font-variant-numeric: tabular-nums; font-weight: 500; }
.row-sub { color: var(--label-2); font-size: 12px; }
.reading .row-sub a { color: var(--tint); font-size: 12px; font-weight: 400; }
.side-note { margin: -2px 0 8px; color: var(--label-2); font-size: 12px; }
.tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
.tile { display: flex; flex-direction: column; gap: 4px; padding: 12px; border-radius: var(--r-sm); background: var(--fill-2); }
.tile-num { font: 650 26px/1.1 var(--font-display); letter-spacing: -.025em; font-variant-numeric: tabular-nums; }
.tile-cap { color: var(--label-2); font-size: 12px; line-height: 1.35; }
.tiles-foot { margin: 12px 0 6px; }
.section-title { font-size: 22px; font-weight: 650; letter-spacing: -.02em; }
.section-actions { display: flex; gap: 4px; align-items: center; }
.section-actions .btn-plain { text-decoration: none; }

/* Shared bits */
.pill { display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; padding: 3px 10px; border-radius: 980px;
  background: var(--fill); color: var(--label-2); font-size: 12px; font-weight: 600; letter-spacing: 0; }
.pill-tint { background: var(--tint-fill); color: var(--tint); }
.pill-green { background: var(--green-fill); color: var(--green); }
.pill-orange { background: var(--orange-fill); color: var(--orange); }
.pill-red { background: var(--red-fill); color: var(--red); }
.pill-move { opacity: .75; font-weight: 400; }
.foot-note { margin: 6px 0 0; color: var(--label-2); font-size: 13px; }
.foot-note:empty { display: none; }
.seg { display: inline-flex; padding: 2px; border-radius: 9px; background: var(--fill); max-width: 100%; }
.seg button { flex: none; border: 0; background: transparent; border-radius: 7px; padding: 5px 12px; font-size: 13px; font-weight: 500; color: var(--label); }
.seg button[aria-pressed="true"] { background: var(--surface); font-weight: 600; box-shadow: 0 3px 8px rgba(0,0,0,.12), 0 0 0 .5px rgba(0,0,0,.04); }
:root[data-theme="dark"] .seg button[aria-pressed="true"] { background: #636366; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .seg button[aria-pressed="true"] { background: #636366; } }
.seg-col { display: grid; width: 100%; }
.seg-col button { text-align: left; padding: 7px 12px; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { border: 0; border-radius: 980px; background: var(--fill); color: var(--label); padding: 6px 12px; font-size: 13px; font-weight: 500; text-align: left; }
.chip span { color: var(--label-2); font-variant-numeric: tabular-nums; margin-left: 2px; }
.chip[aria-pressed="true"] { background: var(--label); color: var(--bg); }
.chip[aria-pressed="true"] span { color: inherit; opacity: .7; }
.chip-sm { padding: 3px 9px; font-size: 12px; font-family: var(--font-mono); }
.search { position: relative; display: flex; align-items: center; }
.search svg { position: absolute; left: 10px; color: var(--label-2); width: 16px; height: 16px; pointer-events: none; }
.search input { width: 100%; border: 0; border-radius: var(--r-sm); background: var(--fill); color: var(--label); font: inherit; font-size: 15px; padding: 8px 12px 8px 32px; outline: none; }
.search input:focus { box-shadow: 0 0 0 3px color-mix(in srgb, var(--tint) 45%, transparent); }
.search input::placeholder { color: var(--label-2); }

/* Feed */
.feed-grid { display: grid; grid-template-columns: 240px minmax(0, 820px); gap: 40px; align-items: start; }
.feed-tools { position: sticky; top: 80px; display: grid; gap: 4px; }
.feed-tools .label { margin-top: 18px; margin-bottom: 6px; }
.exports { margin-top: 18px; }
.group { list-style: none; margin: 16px 0 0; padding: 0; background: var(--surface); border-radius: var(--r); box-shadow: var(--shadow); overflow: hidden; }
.group > li + li, .cards > .feed-card + .feed-card { border-top: .5px solid var(--sep); }
.cards { margin-top: 0; }
.feed-card { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 16px; padding: 16px 18px; transition: background .3s; }
.feed-card:hover { background: var(--fill-2); }
.feed-card.flash { background: var(--tint-fill); }
.card-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; color: var(--label-2); font-size: 12px; }
.card-meta .src { font-weight: 600; color: var(--label); }
.card-meta .pill { margin-left: 4px; padding: 1px 8px; font-size: 11px; }
.card-title { margin-top: 3px; font-size: 17px; line-height: 1.3; font-weight: 600; letter-spacing: -.02em; }
.card-title a { color: var(--label); }
.card-title a:hover { color: var(--tint); text-decoration: none; }
.card-why { margin: 4px 0 0; font-size: 14px; }
.card-summary { margin: 4px 0 0; color: var(--label-2); font-size: 14px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.perspectives { margin: 10px 0 0; display: grid; gap: 6px; padding: 10px 12px; border-radius: var(--r-sm); background: var(--fill-2); font-size: 14px; }
.perspectives dt { font-size: 12px; font-weight: 600; color: var(--label-2); }
.perspectives dd { margin: 0; }
.card-side { display: flex; flex-direction: column; align-items: flex-end; justify-content: space-between; gap: 8px; }
.score { font: 600 13px/1 var(--font); color: var(--label-2); font-variant-numeric: tabular-nums; padding: 4px 6px; border-radius: 6px; border: 0; background: var(--fill-2); cursor: pointer; transition: background .15s, color .15s; }
.score:hover { background: var(--tint-fill); color: var(--tint); }
.card-actions { display: flex; gap: 0; opacity: .6; transition: opacity .15s; }
.feed-card:hover .card-actions, .feed-card:focus-within .card-actions, .card-actions:has([aria-pressed="true"]) { opacity: 1; }
.act { width: 32px; height: 32px; display: inline-grid; place-items: center; border: 0; background: transparent; color: var(--label-2); border-radius: 8px; }
.act:hover { background: var(--fill); color: var(--label); }
.act[aria-pressed="true"] { color: var(--tint); background: var(--tint-fill); }

/* Other panels */
.panel-title { font-size: 34px; font-weight: 700; letter-spacing: -.035em; }
.panel-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; justify-content: space-between; }
.digest { background: var(--surface); border-radius: var(--r); box-shadow: var(--shadow); padding: 28px; }
.digest-list { margin: 22px 0 0; padding: 0; list-style: none; display: grid; gap: 16px; }
.digest-list li { padding-left: 16px; border-left: 3px solid var(--fill); }
.digest-list p { margin: 0; font: 17px/1.5 var(--font-serif); }
.cites { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 6px; }
.ask-form { margin-top: 18px; display: flex; gap: 10px; }
.search-lg { flex: 1; }
.search-lg input { font-size: 17px; padding: 11px 14px 11px 36px; border-radius: 12px; }
.ask-form + .chip-row { margin-top: 12px; }
.welcome { background: var(--tint-fill); border-radius: var(--r); padding: 18px 22px; margin-bottom: 24px; display: grid; gap: 10px; justify-items: start; }
.welcome-lead { margin: 0; font-size: 17px; font-weight: 600; color: var(--label); }
.welcome-steps { margin: 0; padding-left: 20px; display: grid; gap: 4px; color: var(--label); }
[data-welcome="done"] #welcome { display: none; }
.ask-sources { margin: 10px 0 0; padding-left: 20px; display: grid; gap: 6px; }
.ask-result { margin-top: 18px; background: var(--surface); border-radius: var(--r); box-shadow: var(--shadow); padding: 18px 20px; }
.answer-text { margin: 0; white-space: pre-wrap; font-size: 15px; }
.list-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; padding: 12px 16px; }
.row-text { flex: 1 1 220px; min-width: 0; display: flex; flex-direction: column; gap: 2px; }
.row-title { font-size: 15px; font-weight: 600; }
.row-actions { display: flex; flex-wrap: wrap; gap: 2px; }
.snippet { margin: 10px 0 0; padding: 10px 12px; border-radius: var(--r-sm); background: var(--fill-2); font: 12px/1.5 var(--font-mono); white-space: pre-wrap; overflow-wrap: anywhere; }
.empty { padding: 40px 20px; text-align: center; color: var(--label-2); }
.empty p { margin: 0; }
.empty p:first-child { color: var(--label); font-size: 17px; font-weight: 600; margin-bottom: 4px; }

/* Sheets, toast, footer */
.form-sheet { display: flex; flex-direction: column; gap: 12px; }
.form-label { font-size: 13px; font-weight: 600; color: var(--label); margin-top: 4px; }
.form-input, .form-select, .form-textarea {
  width: 100%; border: .5px solid var(--sep); border-radius: var(--r-sm);
  background: var(--fill-2); color: var(--label); font: inherit; font-size: 14px;
  padding: 9px 12px; outline: none; transition: border-color .15s, box-shadow .15s;
}
.form-input:focus, .form-select:focus, .form-textarea:focus {
  border-color: var(--tint); box-shadow: 0 0 0 3px color-mix(in srgb, var(--tint) 25%, transparent); background: var(--surface);
}
.form-textarea { resize: vertical; min-height: 72px; }
.form-row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.form-actions { display: flex; justify-content: flex-end; align-items: center; gap: 8px; margin-top: 6px; padding-top: 12px; border-top: .5px solid var(--sep); }
.score-badge-lg { font-size: 32px; font-weight: 700; color: var(--tint); line-height: 1; font-family: var(--font-mono); }
.score-bar-track { height: 6px; border-radius: 980px; background: var(--fill); overflow: hidden; margin-top: 5px; }
.score-bar-fill { height: 100%; border-radius: 980px; background: var(--tint); transition: width .3s var(--ease); }
mark.highlight { background: var(--mark); color: var(--mark-ink); border-radius: 3px; padding: 0 2px; }
.sheet { width: min(600px, calc(100vw - 32px)); max-height: min(82vh, 760px); padding: 0; border: 0; border-radius: 18px;
  background: var(--surface); color: var(--label); box-shadow: var(--shadow-pop); overflow: hidden; }
.sheet-wide { width: min(820px, calc(100vw - 32px)); }
.sheet[open] { display: flex; flex-direction: column; animation: sheet .35s var(--ease); }
.sheet::backdrop { background: rgba(0,0,0,.32); backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px); }
@keyframes sheet { from { opacity: 0; transform: translateY(16px) scale(.98); } }
.sheet-head { display: flex; align-items: center; justify-content: space-between; padding: 14px 14px 10px 22px; border-bottom: .5px solid var(--sep); }
.sheet-head h2 { font-size: 17px; font-weight: 650; }
.sheet-body { padding: 10px 22px 22px; overflow: auto; }
.sheet-body .kv { grid-template-columns: max-content 1fr; }
.sheet-body .kv dd { text-align: left; font-weight: 400; overflow-wrap: anywhere; }
.guide p, .guide li { font-size: 15px; }
.guide h3 { font-size: 15px; font-weight: 650; margin-top: 18px; }
.guide ol { padding-left: 20px; display: grid; gap: 6px; }
.log-table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
.log-table th { text-align: left; color: var(--label-2); font-weight: 600; padding: 8px 12px 8px 0; border-bottom: .5px solid var(--sep); }
.log-table td { padding: 9px 12px 9px 0; border-bottom: .5px solid var(--sep); vertical-align: top; }
.log-table td:last-child { color: var(--label-2); overflow-wrap: anywhere; }
.toast { position: fixed; left: 50%; bottom: calc(24px + env(safe-area-inset-bottom)); transform: translateX(-50%); z-index: 50;
  max-width: calc(100vw - 32px); padding: 10px 18px; border-radius: 980px; background: var(--label); color: var(--bg);
  font-size: 14px; font-weight: 500; box-shadow: var(--shadow-pop); animation: toast .35s var(--ease); }
@keyframes toast { from { opacity: 0; transform: translate(-50%, 12px); } }
.footer { margin-top: 48px; padding-block: 20px calc(32px + env(safe-area-inset-bottom)); border-top: .5px solid var(--sep); color: var(--label-2); font-size: 12px; }
.footer p { margin: 0; }
kbd { font: 11px var(--font-mono); padding: 1px 5px; border-radius: 4px; background: var(--fill); }
.spinner { color: var(--label-2); font-size: 14px; padding: 8px 0; }

@media (max-width: 1080px) {
  .brief-grid { grid-template-columns: minmax(0, 1fr); gap: 32px; }
  .brief-side { position: static; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
  .feed-grid { grid-template-columns: minmax(0, 1fr); gap: 20px; }
  .feed-tools { position: static; }
  .seg-col { display: inline-flex; width: auto; overflow-x: auto; }
}
@media (max-width: 720px) {
  .bar-inner { flex-wrap: wrap; gap: 0 8px; padding-top: 8px; }
  .wordmark { flex: 1; }
  .tabs { order: 3; flex-basis: 100%; margin: 0 -8px; padding: 6px 0 8px; }
  .tabs > .tab-btn, .nav-more > summary { padding: 6px 7px; font-size: 13px; gap: 4px; }
  .page { padding-left: max(16px, env(safe-area-inset-left)); padding-right: max(16px, env(safe-area-inset-right)); }
  .bar-inner { padding-left: max(16px, env(safe-area-inset-left)); padding-right: max(16px, env(safe-area-inset-right)); }
  .tab-content { padding-top: 24px; }
  .q-card { padding: 20px 18px 14px; }
  .q-head { flex-direction: column; gap: 8px; }
  .q-foot .foot-note:last-child { margin-left: 0; flex-basis: 100%; }
  .lede { font-size: 17px; }
  .ask-form { flex-direction: column; }
  .ask-form .btn-primary { justify-content: center; margin-left: 0; }
  .feed-card { grid-template-columns: 1fr; gap: 8px; }
  .card-side { flex-direction: row; align-items: center; }
  .seg-col { display: flex; width: 100%; }
  .seg-col button { flex: 1 1 auto; text-align: center; padding-inline: 6px; }
  .sheet { width: 100vw; max-width: 100vw; margin: auto 0 0; border-radius: 18px 18px 0 0; max-height: 85vh; }
}
@media (hover: none) { .card-actions { opacity: 1; } }
@media (pointer: coarse) {
  .act, .btn-plain, .chip, .tab-btn, .seg button, .icon-btn, .btn-primary, .search input, .btn-secondary, .more summary, .status-line, .score, .form-input, .form-select, .form-textarea { min-height: 44px; }
  .act, .icon-btn { width: 44px; }
  .more summary { display: flex; align-items: center; }
  .exports a { display: inline-block; padding: 13px 6px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""

_JS_ICONS = {"up": ICON_UP, "down": ICON_DOWN}

JS = "const ICONS = " + json.dumps(_JS_ICONS) + ";\nconst ALL_WEIGHTS = " + json.dumps(RANKING_MODES) + ";\n" + r"""
// Feed text and model output are untrusted: escape every value placed into HTML.
function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'})[ch]);
}

// Only http(s) links are rendered; anything else (javascript:, data:, ...) becomes '#'.
function safeUrl(u) {
  try {
    const parsed = new URL(String(u || ''));
    return (parsed.protocol === 'http:' || parsed.protocol === 'https:') ? esc(u) : '#';
  } catch (e) {
    return '#';
  }
}

const TAB_NAMES = ['brief', 'feed', 'digest', 'trends', 'opportunities', 'ask', 'sources', 'quarantine'];
const MORE_TABS = {digest: 'Top 5', trends: 'Trending', sources: 'Websites', quarantine: 'Blocked'};
const KIND_LABELS = {paper: 'Paper', release: 'Release', post: 'Discussion', story: 'Discussion', discussion: 'Discussion', article: 'Article', news: 'Article'};
const CATEGORY_OF_KIND = {paper: 'paper', release: 'release', post: 'discussion', story: 'discussion', discussion: 'discussion', article: 'article', news: 'article'};
// Wording follows the mode weights in gleaner/scoring.py
const MODE_DESCRIPTIONS = {
  balanced: 'What matches your interests best.',
  deep_research: 'Ideas unlike what you\u2019ve seen lately.',
  breaking_radar: 'What many websites are talking about.',
  actionable: 'Releases, deadlines and papers you can use.'
};
let activeCategory = 'all';

function switchTab(name, btn) {
  document.querySelectorAll('.tab-btn[data-tab]').forEach(b => {
    const on = b.dataset.tab === name;
    b.setAttribute('aria-selected', String(on));
    b.setAttribute('tabindex', on ? '0' : '-1');
    if (on && !btn) b.scrollIntoView({block: 'nearest', inline: 'nearest'});
  });
  const more = document.getElementById('nav-more');
  const moreLabel = document.getElementById('nav-more-label');
  if (more) {
    more.open = false;
    more.setAttribute('data-active', String(Boolean(MORE_TABS[name])));
  }
  if (moreLabel) moreLabel.textContent = MORE_TABS[name] || 'More';
  document.querySelectorAll('.tab-content').forEach(p => { p.hidden = p.id !== 'tab-' + name; });
  window.scrollTo({top: 0});
  try { history.replaceState(null, '', '#' + name); } catch (e) {}
}

let toastTimer = null;
function showToast(msg) {
  const t = document.getElementById('toast');
  if (!t) return;
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2800);
}

function askSuggestion(text) {
  const input = document.getElementById('ask-input');
  if (!input) return;
  input.value = text.trim();
  askQuestion({preventDefault() {}});
}

function applyFilters() {
  const input = document.getElementById('feed-search');
  const q = (input && input.value || '').trim();
  const qLower = q.toLowerCase();
  let shown = 0, total = 0;
  document.querySelectorAll('#cards-container .feed-card').forEach(card => {
    total++;
    const okCat = activeCategory === 'all' || card.dataset.category === activeCategory;
    const main = card.querySelector('.card-main');  // not the buttons: "why" would match every card
    const okText = !qLower || (main || card).textContent.toLowerCase().includes(qLower);
    card.hidden = !(okCat && okText);
    if (!card.hidden) {
      shown++;
      highlightMatchesInCard(card, q);
    }
  });
  const count = document.getElementById('filter-count');
  if (count) count.textContent = (q || activeCategory !== 'all') ? (shown + ' of ' + total + ' items') : '';
}

function highlightMatchesInCard(card, query) {
  const titleA = card.querySelector('.card-title a');
  if (titleA) {
    if (!titleA.dataset.orig) titleA.dataset.orig = titleA.textContent;
    titleA.innerHTML = highlightHtml(titleA.dataset.orig, query);
  }
  const summaryP = card.querySelector('.card-summary');
  if (summaryP) {
    if (!summaryP.dataset.orig) summaryP.dataset.orig = summaryP.textContent;
    summaryP.innerHTML = highlightHtml(summaryP.dataset.orig, query);
  }
  const whyP = card.querySelector('.card-why');
  if (whyP) {
    if (!whyP.dataset.orig) whyP.dataset.orig = whyP.textContent;
    whyP.innerHTML = highlightHtml(whyP.dataset.orig, query);
  }
}

function highlightHtml(rawText, query) {
  const text = String(rawText);
  if (!query) return esc(text);
  // match on the raw text and escape each piece, so a query never splits an HTML entity
  const reg = new RegExp('(' + query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
  return text.split(reg).map((part, i) => i % 2 ? '<mark class="highlight">' + esc(part) + '</mark>' : esc(part)).join('');
}

function filterFeed(query) { applyFilters(); }

function filterByCategory(category, btn) {
  activeCategory = (activeCategory === category) ? 'all' : category;
  document.querySelectorAll('.chip[data-category]').forEach(c => c.setAttribute('aria-pressed', String(c.dataset.category === activeCategory)));
  applyFilters();
}

function toggleTheme() {
  const root = document.documentElement;
  const current = root.getAttribute('data-theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const next = current === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  try { localStorage.setItem('gleaner-theme', next); } catch (e) {}
}

function highlightCard(cardId) {
  switchTab('feed');
  const el = document.getElementById('card-' + cardId);
  if (!el) { showToast('That item is no longer in the feed'); return; }
  el.hidden = false;
  el.scrollIntoView({behavior: 'smooth', block: 'center'});
  el.classList.add('flash');
  setTimeout(() => el.classList.remove('flash'), 1600);
}

let currentMode = 'balanced';

async function setRankingMode(mode, btn) {
  currentMode = mode;
  document.querySelectorAll('.mode-pill').forEach(p => p.setAttribute('aria-pressed', String(p === btn)));
  const desc = document.getElementById('mode-desc');
  if (desc && MODE_DESCRIPTIONS[mode]) desc.textContent = MODE_DESCRIPTIONS[mode];
  const container = document.getElementById('cards-container');
  try {
    container.setAttribute('aria-busy', 'true');
    const res = await fetch('/api/feed?mode=' + encodeURIComponent(mode));
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    if (data.cards && data.cards.length) container.innerHTML = data.cards.map(renderCardHtml).join('');
    applyFilters();
  } catch (e) {
    showToast('Couldn’t re-rank the feed. Is the server still running?');
  } finally {
    container.removeAttribute('aria-busy');
  }
}

function shortDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return isNaN(d) ? '' : d.getUTCDate() + ' ' + d.toLocaleString('en', {month: 'short', timeZone: 'UTC'});
}

// Must produce the same markup as render_card() in gleaner/dashboard.py.
function renderCardHtml(c) {
  const id = esc(c.item.id);
  const kind = String(c.item.kind || 'article').toLowerCase();
  const url = safeUrl(c.item.url);
  const p = c.components || {};
  const n = x => Number(x || 0).toFixed(2);
  const meta = ['<span class="src">' + esc(c.item.source_id || 'unknown') + '</span>', '<span>' + esc(KIND_LABELS[kind] || kind) + '</span>'];
  if (c.item.published_at) meta.push('<time datetime="' + esc(c.item.published_at) + '">' + esc(shortDate(c.item.published_at)) + '</time>');
  let badges = '';
  if (c.exploring) badges += '<span class="pill pill-tint" title="Just outside your interests">Exploring</span>';
  if (c.also_covered_by && c.also_covered_by.length) badges += '<span class="pill" title="Also covered by other sources">+' + c.also_covered_by.length + ' sources</span>';
  const why = ['Evaluated via fallback', 'Matches your active interests'].some(x => String(c.why || '').startsWith(x)) ? '' : '<p class="card-why">' + esc(c.why) + '</p>';
  const summary = (c.summary && c.summary !== c.item.title) ? '<p class="card-summary">' + esc(c.summary) + '</p>' : '';
  let persp = '';
  if (c.perspectives && Object.keys(c.perspectives).length) {
    persp = '<dl class="perspectives">' + Object.entries(c.perspectives).map(([k, val]) =>
      '<div><dt>' + esc(k.replace(/_/g, ' ')) + '</dt><dd>' + esc(val) + '</dd></div>').join('') + '</dl>';
  }
  const breakdown = 'Score ' + n(c.score) + ': relevance ' + n(p.relevance) + ', novelty ' + n(p.novelty) + ', quality ' + n(p.quality) + ', momentum ' + n(p.momentum) + ', actionability ' + n(p.actionability);
  const fb = (action, label, icon) => '<button type="button" class="act" data-id="' + id + '" data-action="' + action + '" aria-pressed="false" aria-label="' + label + '" title="' + label + '" onclick="submitFeedback(this.dataset.id, this.dataset.action, this)">' + icon + '</button>';
  return '<article class="feed-card" id="card-' + id + '" data-id="' + id + '" data-kind="' + esc(kind) + '" data-category="' + (CATEGORY_OF_KIND[kind] || 'article') + '">' +
    '<div class="card-main"><div class="card-meta">' + meta.join('<span aria-hidden="true">·</span>') + badges + '</div>' +
    '<h3 class="card-title"><a href="' + url + '" target="_blank" rel="noopener noreferrer">' + esc(c.item.title) + '</a></h3>' +
    why + summary + persp + '</div>' +
    '<div class="card-side"><button type="button" class="score" title="' + breakdown + '" data-score="' + n(c.score) + '" data-rel="' + n(p.relevance) + '" data-nov="' + n(p.novelty) + '" data-qual="' + n(p.quality) + '" data-mom="' + n(p.momentum) + '" data-act="' + n(p.actionability) + '" data-title="' + esc(c.item.title) + '" data-why="' + esc(c.why) + '" onclick="inspectScore(this)">Why this?</button><div class="card-actions">' +
    fb('like', 'More like this', ICONS.up) + fb('dislike', 'Less like this', ICONS.down) +
    '</div></div></article>';
}

const FEEDBACK_MESSAGES = {like: 'You’ll see more like this.', dislike: 'You’ll see less like this.', save: 'Saved.'};

async function submitFeedback(itemId, action, btn) {
  try {
    const res = await fetch('/api/feedback', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({item_id: itemId, action: action})
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const card = btn.closest('.feed-card');
    if (card && action !== 'save') card.querySelectorAll('[data-action="like"], [data-action="dislike"]').forEach(b => b.setAttribute('aria-pressed', 'false'));
    btn.setAttribute('aria-pressed', 'true');
    showToast(String(data.message || '').includes('already') ? 'Already noted.' : (FEEDBACK_MESSAGES[action] || 'Noted.'));
  } catch (e) {
    showToast('Couldn’t save your feedback. Try again.');
  }
}

async function submitQuestionFeedback(questionId, rating, btn) {
  try {
    const res = await fetch('/api/questions/' + encodeURIComponent(questionId) + '/feedback', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({rating: rating})
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    btn.parentElement.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b === btn)));
    const hr = data.hit_rate || {};
    const el = document.getElementById('hitrate-' + questionId);
    const parts = [['useful', 'useful'], ['wrong', 'wrong'], ['already_knew', 'knew it']].filter(([k]) => hr[k]).map(([k, label]) => hr[k] + ' ' + label);
    if (el && parts.length) el.textContent = 'Your ratings: ' + parts.join(' · ');
    showToast('Thanks. Your rating is recorded.');
  } catch (e) {
    showToast('Couldn’t save your rating. Try again.');
  }
}

async function removeQuestion(questionId, btn) {
  // the first click asks, the second removes: a stray click can't lose a question
  if (btn.dataset.confirming !== 'yes') {
    btn.dataset.confirming = 'yes';
    btn.textContent = 'Click again to remove';
    setTimeout(() => { btn.dataset.confirming = ''; btn.textContent = 'Remove'; }, 4000);
    return;
  }
  try {
    const res = await fetch('/api/questions/' + encodeURIComponent(questionId), {method: 'DELETE'});
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const card = document.getElementById('question-' + questionId);
    if (card) card.remove();
    showToast('Removed. Its quotes are kept if you add it again.');
  } catch (e) {
    showToast('Couldn’t remove the question. Try again.');
  }
}

async function simulateAttack(btn) {
  btn.disabled = true;
  try {
    const res = await fetch('/api/security/simulate-poison', {method: 'POST'});
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    showToast('Blocked and quarantined (' + data.label + ').');
    setTimeout(() => location.reload(), 900);
  } catch (e) {
    showToast('The test attack didn’t run. Check the server log.');
    btn.disabled = false;
  }
}

async function synthesizeLivingAnswer(qid, btn) {
  const orig = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Synthesizing…';
  try {
    const res = await fetch('/api/questions/' + encodeURIComponent(qid) + '/synthesize?force=true', {method: 'POST'});
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showToast('Synthesis failed: ' + (err.detail || res.statusText));
      return;
    }
    const data = await res.json().catch(() => ({}));
    // name the model that actually wrote it: the answer role is configurable
    showToast(data.synthesized_by ? 'Answer written by ' + data.synthesized_by : (data.reason || 'No answer was written: showing the quotes.'));
    setTimeout(() => location.reload(), 800);
  } catch (err) {
    showToast('Synthesis error: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

async function triggerCollect(btn) {
  const label = btn.querySelector('span');
  btn.disabled = true;
  btn.classList.add('busy');
  if (label) label.textContent = 'Collecting…';

  // progress comes only from the server's real stages (SSE); nothing is shown on a timer
  showToast('Collecting from due sources…');

  let es = null;
  try {
    es = new EventSource('/api/collect/events');
    es.onmessage = (e) => {
      try {
        const ev = JSON.parse(e.data);
        if (ev && ev.message && ev.stage !== 'idle') {
          showToast(ev.message);
          if (label && ev.stage) label.textContent = ev.stage + '…';
        }
        if (ev && (ev.stage === 'complete' || ev.stage === 'finished')) {
          if (es) es.close();
        }
      } catch (err) {}
    };
    es.onerror = () => { if (es) es.close(); };
  } catch (err) {}

  try {
    const res = await fetch('/api/collect?force=true', {method: 'POST'});
    if (es) es.close();
    if (res.status === 409) { showToast('A collect is already running.'); return; }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) { showToast('Collect failed: ' + (data.detail || 'HTTP ' + res.status)); return; }
    const s = data.stats || {};
    if (s.answers_changed > 0) showToast('New evidence for ' + s.answers_changed + ' question' + (s.answers_changed > 1 ? 's' : '') + '.');
    else showToast(s.items_stored > 0 ? 'Collected ' + s.items_stored + ' new items.' : 'Everything is up to date.');
    setTimeout(() => location.reload(), 1200);
  } catch (e) {
    if (es) es.close();
    showToast('Couldn’t reach the server.');
  } finally {
    btn.disabled = false;
    btn.classList.remove('busy');
    if (label) label.textContent = 'Collect';
  }
}

function inspectScore(btn) {
  const d = btn.dataset;
  const score = d.score || '0.00';
  const rel = parseFloat(d.rel || 0);
  const nov = parseFloat(d.nov || 0);
  const qual = parseFloat(d.qual || 0);
  const mom = parseFloat(d.mom || 0);
  const act = parseFloat(d.act || 0);
  const title = d.title || 'Item';
  const why = d.why || '';

  const body = document.getElementById('score-inspector-body');
  if (!body) return;

  function bar(label, val, weight, hint) {
    const pct = Math.min(100, Math.max(0, Math.round(val * 100)));
    return '<div style="margin-top: 14px;">' +
      '<div style="display:flex; justify-content:space-between; font-size:13px; font-weight:500;">' +
        '<span>' + esc(label) + ' <span style="color:var(--label-2); font-weight:400;">(' + esc(weight) + ')</span></span>' +
        '<span style="font-family:var(--font-mono); font-weight:600;">' + val.toFixed(2) + '</span>' +
      '</div>' +
      '<div class="score-bar-track">' +
        '<div class="score-bar-fill" style="width:' + pct + '%;"></div>' +
      '</div>' +
      '<div style="font-size:12px; color:var(--label-2); margin-top:3px;">' + esc(hint) + '</div>' +
    '</div>';
  }

  // the chosen ranking's weights, from scoring.RANKING_MODES: hard-coded numbers drift from the real ones
  const WEIGHTS = ALL_WEIGHTS[currentMode] || ALL_WEIGHTS.balanced;
  const pct = (w) => Math.round(w * 100) + '% of the score';
  const whyHtml = why ? '<p style="font-size: 13px; color: var(--tint); margin: 6px 0 0;">' + esc(why) + '</p>' : '';
  body.innerHTML = '<div style="margin-bottom: 16px;">' +
    '<div style="font-size:12px; color:var(--label-2); text-transform:uppercase; font-weight:600; letter-spacing:.02em; margin-bottom:4px;">Item</div>' +
    '<h3 style="font-size: 16px; font-weight: 600; line-height: 1.35; margin: 0;">' + esc(title) + '</h3>' +
    whyHtml + '</div>' +
    '<div style="display:flex; align-items:baseline; gap:10px; padding: 12px 16px; border-radius: var(--r-sm); background: var(--fill-2);">' +
      '<span class="score-badge-lg">' + esc(score) + '</span>' +
      '<span style="font-size: 13px; color: var(--label-2);">The higher, the better it fits you. Newer items count a little more.</span>' +
    '</div>' +
    '<div style="margin-top: 16px;">' +
      bar('Match', rel, pct(WEIGHTS.relevance), 'How well it matches your interests') +
      bar('New idea', nov, pct(WEIGHTS.novelty), 'How different it is from what you saw lately') +
      bar('Trust', qual, pct(WEIGHTS.quality), 'How much you trust the website, and your likes and dislikes of it') +
      bar('Popular', mom, pct(WEIGHTS.momentum), 'How many websites, Hacker News votes and GitHub stars') +
      bar('Useful', act, pct(WEIGHTS.actionability), 'Releases, deadlines and tools come before opinions') +
    '</div>';
  openSheet('score-inspector-modal');
}

async function seedSampleOpportunities(btn) {
  const old = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Loading…';
  try {
    const res = await fetch('/api/opportunities/seed-sample', {method: 'POST'});
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    showToast('Loaded ' + (data.count || 4) + ' sample research deadlines.');
    setTimeout(() => location.reload(), 800);
  } catch (e) {
    showToast('Couldn’t load sample deadlines: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
}

function openAddSourceModal() {
  openSheet('add-source-modal');
  const inp = document.getElementById('source-id');
  if (inp) inp.focus();
}

function onSourceTypeChange(type) {
  const lbl = document.getElementById('source-target-label');
  const inp = document.getElementById('source-target');
  if (!lbl || !inp) return;
  if (type === 'rss') {
    lbl.textContent = 'Feed URL';
    inp.placeholder = 'https://simonwillison.net/atom/everything/';
  } else if (type === 'github') {
    lbl.textContent = 'Repository (owner/repo)';
    inp.placeholder = 'langchain-ai/langgraph';
  } else if (type === 'arxiv') {
    lbl.textContent = 'Search Query';
    inp.placeholder = 'cat:cs.AI OR cat:cs.CL';
  } else if (type === 'hn') {
    lbl.textContent = 'Search Query';
    inp.placeholder = 'e.g. LLM or LangGraph';
  }
}

async function submitAddSource(e) {
  e.preventDefault();
  const type = document.getElementById('source-type').value;
  const id = document.getElementById('source-id').value.trim();
  const target = document.getElementById('source-target').value.trim();
  const cadence = parseInt(document.getElementById('source-cadence').value, 10);
  const trust = parseFloat(document.getElementById('source-trust').value);
  const btn = document.getElementById('btn-save-source');

  const params = {};
  if (type === 'rss') params.url = target;
  else if (type === 'github') params.repo = target;
  else if (type === 'arxiv') params.query = target;
  else if (type === 'hn') { if (target) params.query = target; }

  btn.disabled = true;
  btn.textContent = 'Adding…';
  try {
    const res = await fetch('/api/sources', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id: id, type: type, params: params, cadence_s: cadence * 60, trust: trust})
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'HTTP ' + res.status);
    showToast('Source "' + id + '" added.');
    closeModal('add-source-modal');
    setTimeout(() => location.reload(), 800);
  } catch (err) {
    showToast('Failed to add source: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Add website';
  }
}

function openAddQuestionModal() {
  openSheet('add-question-modal');
  const inp = document.getElementById('question-text');
  if (inp) inp.focus();
}

async function submitAddQuestion(e) {
  e.preventDefault();
  const text = document.getElementById('question-text').value.trim();
  const btn = document.getElementById('btn-save-question');

  btn.disabled = true;
  btn.textContent = 'Searching your sources…';
  try {
    const res = await fetch('/api/questions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: text})
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'HTTP ' + res.status);
    const found = (data.evidence && data.evidence.claims_added) || 0;
    showToast(found ? `Added. Found ${found} quote${found === 1 ? '' : 's'} in what you've collected.`
                    : "Added. Nothing in what you've collected yet. The next collect looks again.");
    closeModal('add-question-modal');
    // open the page at the new question: it can sort far down the Brief
    setTimeout(() => location.assign('/?added=' + encodeURIComponent(data.id) + '#question-' + data.id), 1600);
  } catch (err) {
    showToast('Failed to track question: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Add question';
  }
}

async function askQuestion(e) {
  e.preventDefault();
  const input = document.getElementById('ask-input');
  const box = document.getElementById('ask-result');
  const q = String(input.value || '').trim();
  if (!q) return;
  box.hidden = false;
  box.innerHTML = '<p class="spinner">Searching your feed…</p>';
  try {
    const res = await fetch('/api/ask', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query: q})});
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    // number the items the answer used and list them by title, never by their internal ids
    const sources = data.sources || [];
    let answer = esc(data.answer);
    sources.forEach((s, i) => { answer = answer.split('[' + esc(s.id) + ']').join('[' + (i + 1) + ']'); });
    // Llama writes **bold** and [text](url): shown as such, on the already-escaped text, http(s) links only
    // Muse writes Markdown: headings, bullets and quote lines become plain lines, on the escaped text
    answer = answer.split('\n').map(line => line
        .replace(/^#{1,6}\s+(.*)$/, '<strong>$1</strong>')
        .replace(/^&gt;\s?/, '')
        .replace(/^\s*[-*]\s+/, '\u2022 ')).join('\n')
      .replace(/`([^`\n]+)`/g, '$1')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    const list = sources.length ? '<ol class="ask-sources">' + sources.map(s =>
      '<li><a href="' + safeUrl(s.url) + '" target="_blank" rel="noopener noreferrer">' + esc(s.title) + '</a> <span class="foot-note">' + esc(s.source || '') + '</span></li>').join('') + '</ol>' : '';
    const by = data.answered_by ? '<p class="foot-note">Answered by ' + esc(data.answered_by) + '</p>' : '';
    box.innerHTML = '<p class="answer-text">' + answer + '</p>' + list + by;
    return data;
  } catch (err) {
    box.innerHTML = '<p class="answer-text">Couldn’t get an answer. Is the server still running?</p>';
    return null;
  }
}

async function findFeeds(qid, btn) {
  const body = document.getElementById('feeds-body');
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Searching the web, about half a minute…';
  try {
    const res = await fetch('/api/questions/' + encodeURIComponent(qid) + '/find-feeds', {method: 'POST'});
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'HTTP ' + res.status);
    const feeds = data.feeds || [];
    body.innerHTML = '<p class="foot-note" style="margin-top:0">' + esc(data.searched_by || 'The model') + ' searched the web. gleaner opened each feed to check it works.</p>' +
      (feeds.length ? '<ul class="group">' + feeds.map((f, i) =>
        '<li class="feed-pick"><div><strong>' + esc(f.title) + '</strong><div class="foot-note">' + esc(f.why) + '</div>' +
        '<div class="foot-note">Newest: ' + esc(f.latest) + '</div></div>' +
        '<button type="button" class="btn-secondary" data-i="' + i + '" onclick="addFoundFeed(this)">+ Add</button></li>').join('') + '</ul>'
        : '<p>No working feeds found this time. Try again later, or add a website yourself.</p>');
    body.dataset.feeds = JSON.stringify(feeds);
    openSheet('feeds-modal');
  } catch (err) {
    showToast(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

async function addFoundFeed(btn) {
  const f = JSON.parse(document.getElementById('feeds-body').dataset.feeds || '[]')[Number(btn.dataset.i)];
  if (!f) return;
  const host = (() => { try { return new URL(f.url).hostname.replace(/^www\./, ''); } catch (e) { return 'site'; } })();
  const id = (host + '-' + Math.random().toString(36).slice(2, 6)).replace(/[^a-zA-Z0-9_-]/g, '-').slice(0, 64);
  btn.disabled = true;
  try {
    const res = await fetch('/api/sources', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id: id, type: 'rss', params: {url: f.url}, cadence_s: 3600, trust: 0.6})});
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'HTTP ' + res.status);
    btn.textContent = 'Added';
    showToast('Added ' + f.title + '. The next collect reads it.');
  } catch (err) {
    btn.disabled = false;
    showToast('Couldn’t add it: ' + err.message);
  }
}

async function suggestNextSteps(btn) {
  const box = document.getElementById('next-steps-body');
  const label = btn ? btn.textContent : 'Suggest next steps';
  if (btn) { btn.disabled = true; btn.textContent = 'Thinking…'; }
  try {
    const res = await fetch('/api/next-steps', {method: 'POST'});
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'HTTP ' + res.status);
    const steps = data.steps || [];
    if (!steps.length) {
      box.innerHTML = '<p class="side-note">Nothing to suggest yet. Collect from your websites first.</p>';
      return data;
    }
    const items = steps.map(s => {
      const links = (s.sources || []).filter(src => src.url).map(src =>
        '<a href="' + safeUrl(src.url) + '" target="_blank" rel="noopener noreferrer">' + esc(src.title) + '</a>'
      ).join(' · ');
      const findBtn = s.question_id
        ? ' <button type="button" class="btn-plain" data-qid="' + esc(s.question_id) + '" onclick="findFeeds(this.dataset.qid, this)">Find websites</button>'
        : '';
      const sub = [esc(s.why || ''), links].filter(Boolean).join(' · ');
      return '<li><strong>' + esc(s.action) + '</strong><span class="row-sub">' + sub + findBtn + '</span></li>';
    }).join('');
    box.innerHTML = '<ol class="reading">' + items + '</ol><p class="side-note">Suggested by ' + esc(data.by || 'rules') + '</p>';
    return data;
  } catch (err) {
    showToast('Couldn’t suggest next steps: ' + err.message);
    return null;
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = label; }
  }
}

// Voice: the page records 16 kHz mono WAV, Meta's Muse Voice Transcribe turns it into text on
// the server (/api/voice), and the browser's own speech synthesis reads replies (Meta has no TTS).
const VOICE_HELP = 'Try saying: open deadlines, collect, read the top 5, add a question, then your question, or just ask anything.';
const VOICE_MAX_MS = 30000;
let rec = null;
let voiceBusy = false;  // asking for the mic, or sending a clip: a second press waits

function setVoiceState(state) {
  document.querySelectorAll('.voice-btn').forEach(b => {
    b.classList.toggle('rec', state === 'listening');
    b.classList.toggle('busy', state === 'thinking');
    b.setAttribute('aria-pressed', String(state === 'listening'));
    const label = b.querySelector('span');
    if (label) label.textContent = state === 'listening' ? 'Stop' : state === 'thinking' ? 'Working…' : 'Talk';
  });
}

function say(text) {
  if (!('speechSynthesis' in window) || !text) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(String(text).slice(0, 900));
  u.lang = 'en-US';
  speechSynthesis.speak(u);
}

async function toggleVoice() {
  if ('speechSynthesis' in window) speechSynthesis.cancel();
  if (rec) { finishRecording(); return; }
  if (voiceBusy) return;
  if (!navigator.mediaDevices || !window.AudioContext) { showToast('This browser can’t record audio. Type instead.'); return; }
  let stream;
  voiceBusy = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}});
  } catch (err) {
    showToast('The microphone is blocked. Allow it next to the address bar, then press Talk again.');
    return;
  } finally {
    voiceBusy = false;
  }
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(stream);
  const node = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  // stop by itself after 1.5 s of quiet that follows speech, so a press of Talk is enough
  let spoke = false, quietSince = 0;
  node.onaudioprocess = e => {
    const data = new Float32Array(e.inputBuffer.getChannelData(0));
    chunks.push(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    const loud = Math.sqrt(sum / data.length) > 0.02;
    if (loud) { spoke = true; quietSince = 0; }
    else if (spoke) {
      quietSince = quietSince || performance.now();
      if (performance.now() - quietSince > 1500) finishRecording();
    }
  };
  src.connect(node);
  node.connect(ctx.destination);
  rec = {stream, ctx, node, src, chunks, timer: setTimeout(finishRecording, VOICE_MAX_MS)};
  setVoiceState('listening');
  showToast('Listening… it stops when you stop talking.');
}

function encodeWav16k(chunks, rate) {
  const len = chunks.reduce((n, c) => n + c.length, 0);
  const pcm = new Float32Array(len);
  let o = 0;
  for (const c of chunks) { pcm.set(c, o); o += c.length; }
  const ratio = rate / 16000;
  const outLen = Math.floor(len / ratio);
  const buf = new ArrayBuffer(44 + outLen * 2);
  const v = new DataView(buf);
  const str = (p, s) => { for (let i = 0; i < s.length; i++) v.setUint8(p + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + outLen * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 16000, true); v.setUint32(28, 32000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, outLen * 2, true);
  for (let i = 0; i < outLen; i++) {
    // average each window of input samples: a cheap low-pass before dropping to 16 kHz
    const a = Math.floor(i * ratio), b = Math.max(a + 1, Math.min(len, Math.floor((i + 1) * ratio)));
    let s = 0;
    for (let j = a; j < b; j++) s += pcm[j];
    s = Math.max(-1, Math.min(1, s / (b - a)));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return {blob: new Blob([buf], {type: 'audio/wav'}), seconds: outLen / 16000};
}

async function finishRecording() {
  if (!rec) return;
  const r = rec;
  rec = null;
  clearTimeout(r.timer);
  r.src.disconnect(); r.node.disconnect();
  r.stream.getTracks().forEach(t => t.stop());
  const rate = r.ctx.sampleRate;
  r.ctx.close();
  const wav = encodeWav16k(r.chunks, rate);
  if (wav.seconds < 0.4) { setVoiceState('idle'); showToast('I didn’t hear anything. Press Talk and speak.'); return; }
  setVoiceState('thinking');
  voiceBusy = true;
  try {
    const res = await fetch('/api/voice', {method: 'POST', headers: {'Content-Type': 'audio/wav'}, body: wav.blob});
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'HTTP ' + res.status);
    if (!data.text) { showToast('I didn’t catch that. Try again a little louder.'); return; }
    showToast('Heard: “' + data.text + '”');
    await runVoiceCommand(data.command || {action: 'ask', text: data.text});
  } catch (err) {
    showToast(err.message);
  } finally {
    voiceBusy = false;
    setVoiceState('idle');
  }
}

function spokenAnswer(data) {
  if (!data) return 'I couldn’t get an answer.';
  let text = String(data.answer || '')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '$1')
    .replace(/\[[a-zA-Z0-9_-]{1,64}\]/g, '')
    .replace(/\*\*/g, '').replace(/https?:\/\/\S+/g, '')
    .replace(/^#{1,6}\s+/gm, '').replace(/^>\s?/gm, '').replace(/^\s*[-*]\s+/gm, '').replace(/`/g, '');
  const sources = data.sources || [];
  if (sources.length && text.length < 140) text += ' First: ' + sources[0].title + '.';
  return text;
}

async function runVoiceCommand(cmd) {
  if (cmd.action === 'tab') {
    switchTab(cmd.tab);
    say('Opening ' + cmd.label + '.');
  } else if (cmd.action === 'collect') {
    const btn = document.getElementById('btn-collect');
    say(btn.disabled ? 'A collect is already running.' : 'Collecting now.');
    if (!btn.disabled) btn.click();
  } else if (cmd.action === 'read_digest') {
    switchTab('digest');
    const panel = document.getElementById('tab-digest');
    const head = panel.querySelector('h1');
    const points = Array.from(panel.querySelectorAll('.digest-list li')).map(li => li.textContent.trim());
    say(points.length ? (head ? head.textContent + '. ' : '') + points.join('. ') : 'There is no top 5 yet. Say collect first.');
  } else if (cmd.action === 'next_steps') {
    switchTab('brief');
    const data = await suggestNextSteps(document.getElementById('btn-next-steps'));
    const steps = (data && data.steps) || [];
    say(steps.length ? steps[0].action : 'Nothing to suggest yet.');
  } else if (cmd.action === 'add_question') {
    openAddQuestionModal();
    document.getElementById('question-text').value = cmd.text;
    say('Check the question, then press Add question.');
  } else if (cmd.action === 'help') {
    say(VOICE_HELP);
    setTimeout(() => showToast(VOICE_HELP), 1500);
  } else {
    switchTab('ask');
    document.getElementById('ask-input').value = cmd.text;
    say('Let me look.');
    say(spokenAnswer(await askQuestion({preventDefault() {}})));
  }
}

function dismissWelcome() {
  document.documentElement.setAttribute('data-welcome', 'done');
  try { localStorage.setItem('gleaner-welcome-done', '1'); } catch (e) {}
}

function closeModal(id) {
  const m = document.getElementById(id);
  if (m && m.open) m.close();
}

function openSheet(id) {
  const m = document.getElementById(id);
  if (m && !m.open) m.showModal();
  return m;
}

function openGuideModal() { openSheet('guide-modal'); }

function kv(rows) {
  return '<dl class="kv">' + rows.map(([k, val]) => '<dt>' + esc(k) + '</dt><dd>' + esc(val) + '</dd>').join('') + '</dl>';
}

async function openEnvDiagnostics() {
  openSheet('env-modal');
  const body = document.getElementById('env-modal-body');
  body.innerHTML = '<p class="spinner">Checking…</p>';
  try {
    const res = await fetch('/api/env/check');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const d = await res.json();
    const m = d.models || {};
    body.innerHTML = kv([
      ['Status', d.status === 'healthy' ? 'Everything is working' : d.failing_sources + ' website' + (d.failing_sources === 1 ? '' : 's') + ' failing'],
      ['Websites', d.active_sources + ' of ' + d.enabled_sources + ' working' + (d.total_sources > d.enabled_sources ? ' (' + (d.total_sources - d.enabled_sources) + ' paused)' : '')],
      ['Items', d.items_read + ' read, ' + d.items_kept + ' kept for you'],
      ['Checks quotes', m.judge || ''],
      ['Writes answers', (m.answer || '') + (d.answer_key_set ? '' : ' (MODEL_API_KEY not set)')],
      ['Answers Ask', m.ask || ''],
      ['Vector search', d.sqlite_vec ? 'sqlite-vec' : 'Built-in fallback'],
      ['Python', d.python_version],
      ['Database', d.database_path],
      ['Checked', new Date(d.timestamp).toLocaleString([], {dateStyle: 'medium', timeStyle: 'short'})]
    ]);
  } catch (err) {
    body.innerHTML = '<p>Couldn’t load the status: ' + esc(err.message) + '</p>';
  }
}

async function fetchSourceNow(sourceId, btn) {
  const old = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Fetching…';
  try {
    const res = await fetch('/api/sources/' + encodeURIComponent(sourceId) + '/fetch', {method: 'POST'});
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'HTTP ' + res.status);
    showToast(sourceId + ': ' + (data.items_stored || 0) + ' new items.');
    setTimeout(() => location.reload(), 1200);
  } catch (e) {
    showToast(sourceId + ' failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
}

async function toggleSource(sourceId, enable, btn) {
  btn.disabled = true;
  try {
    const res = await fetch('/api/sources/' + encodeURIComponent(sourceId), {
      method: 'PATCH', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: enable})
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'HTTP ' + res.status);
    showToast(sourceId + (enable ? ' resumed.' : ' paused.'));
    setTimeout(() => location.reload(), 700);
  } catch (e) {
    showToast('Couldn’t update ' + sourceId + ': ' + e.message);
    btn.disabled = false;
  }
}

async function viewSourceLogs(sourceId) {
  openSheet('logs-modal');
  document.getElementById('logs-modal-title').textContent = 'Fetch log · ' + sourceId;
  const body = document.getElementById('logs-modal-body');
  body.innerHTML = '<p class="spinner">Loading…</p>';
  try {
    const res = await fetch('/api/sources/' + encodeURIComponent(sourceId) + '/logs');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const logs = Array.isArray(data) ? data : (data.logs || []);
    if (!logs.length) { body.innerHTML = '<div class="empty"><p>No fetches yet.</p></div>'; return; }
    body.innerHTML = '<table class="log-table"><thead><tr><th>When</th><th>Result</th><th>Items</th><th>Details</th></tr></thead><tbody>' +
      logs.map(l => {
        const ok = !l.error && l.status_code >= 200 && l.status_code < 400;
        const result = ok ? '<span class="pill pill-green">OK</span>' : (l.error ? '<span class="pill pill-red">Failed</span>' : '<span class="pill pill-orange">HTTP ' + esc(l.status_code) + '</span>');
        const when = l.fetched_at ? new Date(String(l.fetched_at).replace(/(\.\d{3})\d+/, '$1')).toLocaleString([], {dateStyle: 'medium', timeStyle: 'short'}) : '';
        return '<tr><td>' + esc(when) + '</td><td>' + result + '</td><td>' + esc(l.item_count ?? 0) + '</td><td>' + esc(l.error || '') + '</td></tr>';
      }).join('') + '</tbody></table>';
  } catch (err) {
    body.innerHTML = '<p>Couldn’t load the log: ' + esc(err.message) + '</p>';
  }
}

function init() {
  const tabList = document.querySelector('.tabs');
  if (tabList) tabList.addEventListener('keydown', e => {
    const tabs = Array.from(tabList.querySelectorAll('.tab-btn[data-tab]'));
    const i = tabs.indexOf(document.activeElement);
    if (i === -1) return;
    const next = {ArrowRight: (i + 1) % tabs.length, ArrowLeft: (i - 1 + tabs.length) % tabs.length, Home: 0, End: tabs.length - 1}[e.key];
    if (next === undefined) return;
    e.preventDefault();
    tabs[next].focus();
    switchTab(tabs[next].dataset.tab, tabs[next]);
  });
  document.addEventListener('click', e => {
    document.querySelectorAll('details.nav-more[open], details.q-more[open]').forEach(d => {
      if (!d.contains(e.target)) d.open = false;
    });
  });
  document.addEventListener('keydown', e => {
    const t = e.target;
    if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA') {
      if (e.key === 'Escape' && t.id === 'feed-search') { t.value = ''; applyFilters(); t.blur(); }
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey || document.querySelector('dialog[open]')) return;
    if (e.key === '/') {
      e.preventDefault();
      switchTab('feed');
      const s = document.getElementById('feed-search');
      if (s) { s.focus(); s.select(); }
    } else if (e.key === '?') {
      openGuideModal();
    } else if ((e.key === 'v' || e.key === 'V') && !e.repeat) {
      toggleVoice();
    } else if (e.key >= '1' && e.key <= '8') {
      switchTab(TAB_NAMES[Number(e.key) - 1]);
    }
  });
  document.querySelectorAll('dialog').forEach(d => d.addEventListener('click', e => { if (e.target === d) d.close(); }));
  const start = (location.hash || '').slice(1);
  if (TAB_NAMES.includes(start)) switchTab(start);
}
document.addEventListener('DOMContentLoaded', init);
"""
