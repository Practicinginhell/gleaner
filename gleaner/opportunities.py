"""Opportunity detection and deadline reminders for grants, CFPs, and jobs."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gleaner.db import Database
from gleaner.models import Item, Opportunity
from gleaner.security import screen_prompt_guard

logger = logging.getLogger(__name__)


class ExtractedOpportunity(BaseModel):
    """What the model must return for an opportunity; anything else is rejected.

    Attributes:
        opp_type: One of cfp, grant, fellowship, job or bounty.
        deadline: The deadline, if any.
        eligibility: Short eligibility summary, at most 500 characters.
        link: An http(s) URL without whitespace, at most 2000 characters.
        amount: Funding or salary, at most 200 characters.
    """

    model_config = ConfigDict(extra="ignore")

    opp_type: Literal["cfp", "grant", "fellowship", "job", "bounty"]
    deadline: datetime | None = None
    eligibility: str | None = Field(default=None, max_length=500)
    link: str | None = Field(default=None, max_length=2000)
    amount: str | None = Field(default=None, max_length=200)

    @field_validator("link")
    @classmethod
    def _http_link(cls, value: str | None) -> str | None:
        """Reject a link that isn't an http(s) URL without whitespace."""
        if value is not None and not re.prefixmatch(r"https?://[^\s]+\Z", value):
            raise ValueError("link must be an http(s) URL without whitespace")
        return value


def opportunity_id(item_id: str) -> str:
    """Return a stable opportunity id from the full item id.

    The full id is hashed because a prefix would collide (e.g. arXiv ids).
    """
    return f"opp-{hashlib.sha256(item_id.encode('utf-8')).hexdigest()[:16]}"


def extract_opportunity(
    item: Item,
    db: Database,
    deadline: datetime | None = None,
    opp_type: str = "cfp",
    eligibility: str | None = None,
    link: str | None = None,
    amount: str | None = None,
    now: datetime | None = None,
    model: Any | None = None,
) -> Opportunity | None:
    """Extract and persist structured opportunity with deadline tracking.

    With a ``model`` and no explicit deadline/eligibility, the item text is screened with Prompt
    Guard and the model's JSON is validated against ``ExtractedOpportunity``. Returns None (and
    stores nothing) when the text is flagged, the model call fails, or its output is invalid.
    Re-extracting an item updates its row but keeps a reminder or expired status. Without explicit
    fields it also returns None and changes nothing when the item already has a row that a model
    filled, or when no model is given: a collect with the judge down mustn't erase a deadline.

    Args:
        item: The item describing the opportunity.
        db: Database to store the opportunity in.
        deadline: Explicit deadline; naive datetimes are taken as UTC.
        opp_type: Opportunity type when no model extraction runs.
        eligibility: Explicit eligibility summary.
        link: Link to the opportunity; defaults to the item's URL.
        amount: Funding or salary.
        now: Reference time for expiry; defaults to the current UTC time.
        model: Optional model with `generate`, `invoke` or a plain call.

    Returns:
        The stored opportunity, with its status as stored, or None as described above.
    """
    if now is None:
        now = datetime.now(UTC)

    if deadline is None and eligibility is None:
        with db.connect() as conn:
            stored = conn.execute(
                "SELECT deadline, eligibility FROM opportunities WHERE item_id = ? LIMIT 1;", (item.id,)
            ).fetchone()
        # keep what was extracted: no model can't add to it, and a model has already read this item
        if stored is not None and (model is None or stored["deadline"] or stored["eligibility"]):
            return None

    # If model is provided and explicit fields aren't, use model for structured extraction
    if model is not None and deadline is None and eligibility is None:
        guard = screen_prompt_guard(f"{item.title}\n{item.text[:1000]}")
        if not guard.is_safe:
            logger.warning("Opportunity extraction skipped for %s: Prompt Guard %s (%s)", item.id, guard.label, guard.reason)
            return None
        prompt = (
            "System: You are an information analyst extracting actionable opportunity data.\n"
            f"--- BEGIN OPPORTUNITY TEXT ---\nTitle: {item.title}\n{item.text[:1000]}\n--- END OPPORTUNITY TEXT ---\n"
            "Extract structured opportunity information.\n"
            "Output JSON with keys: opp_type ('cfp'|'grant'|'fellowship'|'job'|'bounty'), "
            "deadline (ISO 8601 string or null), eligibility (short summary or null), "
            "link (url or null), amount (funding/salary or null)."
        )
        try:
            if hasattr(model, "generate"):
                resp = model.generate([{"role": "user", "content": prompt}])
            elif hasattr(model, "invoke"):
                resp = model.invoke(prompt)
            else:
                resp = model(prompt)
        except Exception as e:  # noqa: BLE001 - model boundary: any failure is logged and degrades (no opportunity stored)
            logger.warning("Opportunity extraction model call failed for %s: %s", item.id, e)
            return None
        clean = str(getattr(resp, "content", resp)).strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            extracted = ExtractedOpportunity.model_validate_json(clean)
        except ValidationError as e:
            logger.warning("Opportunity extraction output rejected for %s: %s", item.id, e.errors()[:3])
            return None
        opp_type = extracted.opp_type
        eligibility = extracted.eligibility
        amount = extracted.amount or amount
        link = extracted.link or link
        deadline = extracted.deadline

    if deadline is not None and deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)

    status = "open"
    if deadline is not None and deadline < now:
        status = "expired"

    with db.connect() as conn:
        # One opportunity per item: reuse an existing row's id (including ids from older versions)
        existing = conn.execute("SELECT id FROM opportunities WHERE item_id = ? LIMIT 1;", (item.id,)).fetchone()
    opp_id = existing["id"] if existing else opportunity_id(item.id)
    opp = Opportunity(
        id=opp_id,
        item_id=item.id,
        type=opp_type,
        deadline=deadline,
        eligibility=eligibility,
        link=link or item.url,
        amount=amount,
        status=status,
    )

    with db.connect() as conn:
        conn.execute(
            """
        INSERT INTO opportunities (id, item_id, type, deadline, eligibility, link, amount, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            type = excluded.type,
            deadline = excluded.deadline,
            eligibility = excluded.eligibility,
            link = excluded.link,
            amount = excluded.amount,
            -- Keep reminder/expired state; a passed deadline still expires the row
            status = CASE
                WHEN excluded.status = 'expired' OR opportunities.status IS NULL THEN excluded.status
                ELSE opportunities.status
            END;
        """,
            (
                opp.id,
                opp.item_id,
                opp.type,
                opp.deadline.isoformat() if opp.deadline else None,
                opp.eligibility,
                opp.link,
                opp.amount,
                opp.status,
            ),
        )
        opp.status = conn.execute("SELECT status FROM opportunities WHERE id = ?;", (opp.id,)).fetchone()["status"]

    return opp


def check_opportunity_reminders(
    db: Database,
    now: datetime | None = None,
) -> list[Opportunity]:
    """Fire a reminder, once each, for opportunities due in about 7 days.

    Open opportunities with a deadline 6 to 8 days away are marked `reminded_7d`, so each fires
    only once. A row with an unparseable deadline is logged and skipped.

    Args:
        db: Database holding opportunities.
        now: Reference time; defaults to the current UTC time; naive is taken as UTC.

    Returns:
        The opportunities reminded on this call.
    """
    if now is None:
        now = datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    remind_window_start = now + timedelta(days=6)
    remind_window_end = now + timedelta(days=8)

    fired_reminders: list[Opportunity] = []

    with db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM opportunities WHERE status = 'open' AND deadline IS NOT NULL;"
        ).fetchall()

        for r in rows:
            deadline_str = r["deadline"]
            try:
                deadline = datetime.fromisoformat(deadline_str)
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=UTC)
            except (TypeError, ValueError) as e:
                logger.warning("Skipping opportunity %s with unparseable deadline %r: %s", r["id"], deadline_str, e)
                continue

            if remind_window_start <= deadline <= remind_window_end:
                opp = Opportunity(
                    id=r["id"],
                    item_id=r["item_id"],
                    type=r["type"],
                    deadline=deadline,
                    eligibility=r["eligibility"],
                    link=r["link"],
                    amount=r["amount"],
                    status="reminded_7d",
                )
                fired_reminders.append(opp)
                # Mark as reminded so it fires only once
                conn.execute(
                    "UPDATE opportunities SET status = 'reminded_7d' WHERE id = ?;",
                    (opp.id,),
                )

    return fired_reminders


def _ics_strip_controls(value: str) -> str:
    """Drop control characters (CR, LF, ...) from a value.

    A value can then never start a new iCalendar line.
    """
    return "".join(ch for ch in value if ch >= " " and ch != "\x7f")


def _ics_text(value: str) -> str:
    r"""Escape an RFC 5545 TEXT value.

    Escapes backslash, semicolon, comma, and newlines as \n; CR is stripped.
    """
    value = value.replace("\r", "")
    value = value.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")
    return _ics_strip_controls(value)


def generate_opportunity_ics(opp: Opportunity, item_title: str = "Opportunity Deadline") -> str:
    """Generate an RFC 5545 iCalendar (.ics) string for an opportunity deadline.

    Item and model text is escaped as TEXT, and control characters are stripped from every
    value, so untrusted titles cannot inject calendar lines. Without a deadline the event starts
    and ends now.

    Args:
        opp: The opportunity.
        item_title: Title of the opportunity's item, used in the event summary.

    Returns:
        The iCalendar text, with CRLF line endings.
    """
    now_dt = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if opp.deadline:
        dt_val = opp.deadline
        if dt_val.tzinfo is None:
            dt_val = dt_val.replace(tzinfo=UTC)
        start_str = dt_val.strftime("%Y%m%dT%H%M%SZ")
        end_dt = dt_val + timedelta(hours=1)
        end_str = end_dt.strftime("%Y%m%dT%H%M%SZ")
    else:
        start_str = now_dt
        end_str = now_dt

    summary = _ics_text(f"{opp.type.upper()}: {item_title}")
    desc_parts = [f"Type: {opp.type}"]
    if opp.amount:
        desc_parts.append(f"Amount: {opp.amount}")
    if opp.eligibility:
        desc_parts.append(f"Eligibility: {opp.eligibility}")
    if opp.link:
        desc_parts.append(f"Link: {opp.link}")
    description = "\\n".join(_ics_text(part) for part in desc_parts)
    uid = _ics_strip_controls(opp.id)
    url = _ics_strip_controls(opp.link) if opp.link else None

    ics = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//gleaner//Personal Information Discovery//EN\r\n"
        "CALSCALE:GREGORIAN\r\n"
        "METHOD:PUBLISH\r\n"
        "BEGIN:VEVENT\r\n"
        f"UID:{uid}@gleaner.local\r\n"
        f"DTSTAMP:{now_dt}\r\n"
        f"DTSTART:{start_str}\r\n"
        f"DTEND:{end_str}\r\n"
        f"SUMMARY:{summary}\r\n"
        f"DESCRIPTION:{description}\r\n"
    )
    if url:
        ics += f"URL:{url}\r\n"
    ics += "STATUS:CONFIRMED\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n"
    return ics

