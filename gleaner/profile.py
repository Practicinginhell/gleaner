"""Interest profile bootstrap workflow with approval gate."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from graphflow import Application, InMemoryCheckpointStore, Workflow
from graphflow.persistence import BaseCheckpointStore
from graphflow.persistence.sqlite import SqliteCheckpointStore
from pydantic import BaseModel, Field

from gleaner.config import parse_interests_yaml
from gleaner.db import Database
from gleaner.embed import Embedder, FakeEmbedder

logger = logging.getLogger(__name__)


class ProfileBootstrapState(BaseModel):
    """State of one profile bootstrap run.

    Attributes:
        raw_yaml: The interests YAML submitted for review.
        draft_interests: The parsed interests awaiting approval.
        status: ``draft``, ``pending_approval``, ``approved`` or ``rejected``.
    """

    raw_yaml: str = ""
    draft_interests: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "draft"


def profile_checkpointer(db: Database) -> BaseCheckpointStore:
    """Return the store where profile runs are checkpointed.

    A SQLite file beside a file DB, so a pending approval survives a restart; memory for an
    in-memory DB, or when graphflow's sqlite extra is missing.

    Args:
        db: The app database.

    Returns:
        The checkpoint store.
    """
    if db.path == ":memory:":
        return InMemoryCheckpointStore()
    path = Path(db.path)
    try:
        return SqliteCheckpointStore(str(path.with_name(f"{path.stem}.checkpoints.db")))
    except ImportError as e:
        logger.warning("%s; a pending profile approval will not survive a restart.", e)
        return InMemoryCheckpointStore()


def build_profile_workflow(db: Database, embedder: Embedder | None = None) -> Application:
    """Build the graphflow workflow for the interest profile.

    It parses interests, pauses for human approval, and persists them.

    Args:
        db: The database the approved interests and their vectors are saved to.
        embedder: Embeds each interest's centroid; a ``FakeEmbedder`` of the DB's dimension when None.

    Returns:
        The application with the ``profile_bootstrap`` workflow registered.
    """
    if embedder is None:
        embedder = FakeEmbedder(dim=db.vec_dim)

    wf = Workflow("profile_bootstrap", state_schema=ProfileBootstrapState)

    def draft_node(state: dict[str, Any] | ProfileBootstrapState) -> dict[str, Any]:
        """Parse the YAML into draft interests awaiting approval."""
        raw_yaml = state.raw_yaml if isinstance(state, ProfileBootstrapState) else state.get("raw_yaml", "")
        # Same validation as interests.yaml: a bad draft fails here, before anyone is asked to approve it.
        drafts = [interest.model_dump() for interest in parse_interests_yaml(raw_yaml, "interests YAML")]
        return {"draft_interests": drafts, "status": "pending_approval"}

    def save_node(state: dict[str, Any] | ProfileBootstrapState) -> dict[str, Any]:
        """Upsert the approved interests and embed each one's centroid."""
        draft_interests = (
            state.draft_interests if isinstance(state, ProfileBootstrapState) else state.get("draft_interests", [])
        )
        # New and changed interests get the next profile version, so judgments cached
        # against the old profile are redone; unchanged interests keep theirs.
        with db.connect() as conn:
            next_version = conn.execute("SELECT COALESCE(MAX(profile_version), 0) + 1 FROM interests;").fetchone()[0]
        for d in draft_interests:
            with db.connect() as conn:
                conn.execute(
                    """
                INSERT INTO interests (id, label, description, keywords, exclusions, weight, kind, active, profile_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label = excluded.label,
                    description = excluded.description,
                    keywords = excluded.keywords,
                    exclusions = excluded.exclusions,
                    weight = excluded.weight,
                    kind = excluded.kind,
                    active = excluded.active,
                    profile_version = excluded.profile_version
                WHERE (label, description, keywords, exclusions, weight, kind, active)
                    IS NOT (excluded.label, excluded.description, excluded.keywords, excluded.exclusions,
                            excluded.weight, excluded.kind, excluded.active);
                """,
                    (
                        d["id"],
                        d["label"],
                        d["description"],
                        json.dumps(d.get("keywords", [])),
                        json.dumps(d.get("exclusions", [])),
                        d.get("weight", 1.0),
                        d.get("kind", "topic"),
                        1 if d.get("active", True) else 0,
                        next_version,
                    ),
                )

            # Embed interest centroid: search_query prefix + label + description + keywords
            text_to_embed = f"search_query: {d['label']} {d['description']} {' '.join(d.get('keywords', []))}"
            vec = embedder.embed([text_to_embed])[0]
            db.upsert_interest_vector(d["id"], vec)

        return {"status": "approved"}

    def reject_node(state: dict[str, Any] | ProfileBootstrapState) -> dict[str, Any]:
        return {"status": "rejected"}

    wf.then(draft_node, name="draft")
    wf.require_approval(
        before=save_node,
        prompt="Review interest profile before saving to database:",
        on_reject=reject_node,
    )
    wf.then(save_node, name="save")

    app = Application("profile_app", checkpointer=profile_checkpointer(db))
    app.register(wf)
    return app
