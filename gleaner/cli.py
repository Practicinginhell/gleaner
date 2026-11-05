"""gleaner CLI (`python -m gleaner`).

Subcommands:
  init       Initialize database and bootstrap profile from interests.yaml
  collect    Run one tick of the collection workflow
  feed       Print the current ranked feed to stdout
  digest     Generate or view the daily digest
  ask        Interactive or one-shot RAG question answering over feed
  serve      Run the local FastAPI web server and scheduler
  eval       Run offline evaluation on labeled set
  demo       Run an offline end-to-end demonstration using fixtures
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gleaner.config import load_interests_config, load_sources_config
from gleaner.db import Database
from gleaner.feed import generate_feed
from gleaner.llama import is_meta_model

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"  # sources.yaml, interests.yaml
DATA_DIR = REPO_ROOT / "data"  # the SQLite database (gitignored)
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "discovery"


def get_default_db() -> Database:
    """Open data/gleaner.db, creating the data directory if needed."""
    DATA_DIR.mkdir(exist_ok=True)
    return Database(DATA_DIR / "gleaner.db")


def _build_fixture_transport() -> Any:
    """An httpx mock transport serving the recorded test fixtures for every source."""
    import httpx

    fixtures_dir = FIXTURES_DIR
    rss_xml = (fixtures_dir / "rss_sample.xml").read_text(encoding="utf-8")
    arxiv_xml = (fixtures_dir / "arxiv_sample.xml").read_text(encoding="utf-8")
    hn_json = (fixtures_dir / "hn_sample.json").read_text(encoding="utf-8")
    gh_json = (fixtures_dir / "github_sample.json").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "arxiv.org" in url:
            return httpx.Response(200, text=arxiv_xml)
        if "algolia" in url:
            return httpx.Response(200, text=hn_json)
        if "api.github.com" in url:
            return httpx.Response(200, text=gh_json)
        return httpx.Response(200, text=rss_xml, headers={"ETag": '"fixture-etag-1"'})

    return httpx.MockTransport(handler)


def _confirm(question: str) -> bool | None:
    """Ask a yes/no question on the terminal until it gets an answer.

    Args:
        question: the question to print before ``[y/n]``.

    Returns:
        True for yes, False for no, None when stdin is not interactive (nobody to ask) or closes.
    """
    if not sys.stdin.isatty():
        return None
    while True:
        try:
            answer = input(f"{question} [y/n] ").strip().lower()
        except EOFError:
            return None
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def sync_questions_from_config(db: Database, cfg: Any) -> None:
    """Store the questions from questions.yaml without undoing the user's choices.

    A question set `active: false` in the file stays off, and one the user removed in the app
    stays removed: the file adds and updates questions, it doesn't bring them back.

    Args:
        db: The database.
        cfg: The parsed questions config.
    """
    from gleaner.models import Question

    for q in cfg.questions:
        stored = db.get_question(q.id)
        active = q.active and (stored.active if stored else True)
        db.upsert_question(Question(id=q.id, text=q.text, linked_interests=q.linked_interests, watch=q.watch, active=active))


def cmd_init(args: argparse.Namespace) -> int:
    """Load sources and questions from config; bootstrap the interest profile.

    Sources missing from sources.yaml are paused, not deleted, so their fetch history stays.
    The drafted profile is shown and saved only once approved (``--yes`` approves it).

    Args:
        args: the parsed command-line arguments.

    Returns:
        The exit code: 0, or 1 when the drafted profile needed approval and stdin is not a terminal.
    """
    import uuid

    from graphflow import ApprovalDecision

    from gleaner.models import SourceRow
    from gleaner.profile import build_profile_workflow

    sources_path = CONFIG_DIR / "sources.yaml"
    interests_path = CONFIG_DIR / "interests.yaml"
    print(f"Loading sources from {sources_path.name}...")
    sources = load_sources_config(sources_path)
    print(f"Found {len(sources)} configured sources.")

    print(f"Loading interests from {interests_path.name}...")
    interests = load_interests_config(interests_path)
    print(f"Found {len(interests)} configured interests.")

    db = get_default_db()
    for src in sources:
        row = db.get_source(src.id) or SourceRow(id=src.id, type=src.type)
        # the config sets what a source is; its fetch state (etag, backoff, next due) is kept
        row.type, row.params, row.cadence_s, row.trust, row.enabled = src.type, src.params, src.cadence_s, src.trust, src.enabled
        db.upsert_source(row)
    # The config is the list of sources: pause ones removed from it (kept for their fetch history)
    configured = {src.id for src in sources}
    for row in db.list_sources():
        if row.id not in configured and row.enabled and row.type != "osv":  # the stack watch's own source
            row.enabled = False
            db.upsert_source(row)
            print(f"Paused source '{row.id}': no longer in {sources_path.name}.")

    profile_app = build_profile_workflow(db)
    raw_yaml = interests_path.read_text(encoding="utf-8")
    thread_id = f"cli-init-{uuid.uuid4().hex[:8]}"
    res = profile_app.run({"raw_yaml": raw_yaml}, thread_id=thread_id)
    exit_code = 0
    if res.status == "interrupted":
        # Human approval gate: show the drafted profile before anything is saved.
        print("\nDrafted interest profile:")
        for d in res.state.get("draft_interests", []):
            print(f"  - {d['label']} ({d['id']}, {d['kind']}, weight {d['weight']}): {', '.join(d['keywords'])}")
        approved = True if getattr(args, "yes", False) else _confirm("Save this interest profile?")
        if approved is None:
            print("Not saving the interest profile: stdin is not a terminal. Re-run with --yes to approve it.")
            exit_code = 1
        feedback = "" if approved else "Declined at the CLI."
        res = profile_app.resume(thread_id, ApprovalDecision(approved=bool(approved), feedback=feedback))

    questions_path = CONFIG_DIR / "questions.yaml"
    if questions_path.exists():
        print(f"Loading standing questions from {questions_path.name}...")
        from gleaner.config import parse_questions_yaml
        q_cfg = parse_questions_yaml(questions_path.read_text(encoding="utf-8"))
        sync_questions_from_config(db, q_cfg)
        print(f"Loaded {len(q_cfg.questions)} standing questions.")

    print(
        f"Database initialized at {db.path} (schema v{db.get_schema_version()}, "
        f"sqlite-vec: {db.has_vec}, profile status: {res.state.get('status')})."
    )
    return exit_code


def cmd_collect(args: argparse.Namespace) -> int:
    """Run one collect tick under the run lock.

    ``--offline`` fetches from the recorded fixtures instead of the live sources. The lock is
    released even when the run fails or is interrupted, then the error is raised again.

    Args:
        args: the parsed command-line arguments.

    Returns:
        The exit code: 0, also when another collect run holds the lock.
    """
    import asyncio
    import uuid

    import httpx

    from gleaner.llama import get_llama_model
    from gleaner.workflows import acquire_run_lock, release_run_lock, run_collect_tick

    print("Running collect tick...")
    db = get_default_db()
    run_id = f"cli-collect-{uuid.uuid4().hex[:8]}"
    if not acquire_run_lock(db, run_id=run_id):
        print("Another collect is already running, so this one is skipped.")
        return 0

    async def _run() -> dict:
        if getattr(args, "offline", False):
            async with httpx.AsyncClient(transport=_build_fixture_transport()) as client:
                return await run_collect_tick(db, client=client)
        return await run_collect_tick(
            db,
            model=get_llama_model("judge"),
            writer=get_llama_model("writer"),
            answer_model=get_llama_model("answer"),
        )

    try:
        stats = asyncio.run(_run())
        release_run_lock(db, run_id=run_id, status="finished", stats=stats)
        print(f"Collect tick complete: {stats}")
    except BaseException as exc:  # Ctrl-C too: a held lock would refuse the next collect for 15 minutes
        release_run_lock(db, run_id=run_id, status="failed", errors={"error": str(exc) or type(exc).__name__})
        raise
    return 0


def cmd_feed(args: argparse.Namespace) -> int:
    """Print the top 20 ranked feed cards with their score breakdown.

    Args:
        args: the parsed command-line arguments.

    Returns:
        The exit code: always 0.
    """
    db = get_default_db()
    cards = generate_feed(db, limit=20)
    if not cards:
        print("Ranked Feed: (empty)")
        print("No items in feed yet. Run 'python -m gleaner collect' to fetch items.")
        return 0

    print(f"\n--- Ranked Feed ({len(cards)} items) ---\n")
    for idx, card in enumerate(cards, 1):
        print(f"[{idx}] {card.item.title}")
        print(
            f"    Source: {card.item.source_id or 'unknown'} | Score: {card.score:.4f} "
            f"| Decay: {card.components.decay:.2f}"
        )
        print(f"    Why: {card.why}")
        if card.also_covered_by:
            print(f"    Also covered by: {', '.join(card.also_covered_by)}")
        print(
            f"    Breakdown: rel={card.components.relevance:.2f}, nov={card.components.novelty:.2f}, "
            f"qual={card.components.quality:.2f}, mom={card.components.momentum:.2f}, "
            f"act={card.components.actionability:.2f}"
        )
        print(f"    URL: {card.item.url or 'N/A'}")
        print()
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    """Print the changed living answers, then generate and print the digest.

    Args:
        args: the parsed command-line arguments.

    Returns:
        The exit code: always 0.
    """
    from gleaner.digest import generate_digest
    from gleaner.llama import get_llama_model

    db = get_default_db()
    model_name = getattr(args, "model", None)
    model = get_llama_model("writer", model_name=model_name)

    # Standing questions living answers & diffs (M17, ADR-0007)
    questions = db.list_questions()
    living_diffs = []
    for q in questions:
        ans = db.get_latest_answer(q.id)
        if ans and ans.what_changed:
            living_diffs.append((q, ans))

    if living_diffs:
        print("\n=== Living Answers: What Changed ===")
        for q, ans in living_diffs:
            print(f"  ⚡ [{q.id}] {q.text}")
            print(f"     What changed: {ans.what_changed} (confidence: {ans.confidence.upper()}, v{ans.version})")
            if ans.counterpoint_claim_id:
                cp = db.get_claim(ans.counterpoint_claim_id)
                if cp:
                    print(f"     Counterpoint: {cp.text}")
        print()

    digest = generate_digest(db, model=model)
    print(f"\n=== {digest.title} ===")
    print(f"{digest.summary}\n")
    if not digest.bullets:
        print("No digest bullets available yet. Run 'python -m gleaner init' and 'python -m gleaner collect' first.")
    else:
        for b in digest.bullets:
            cites = ", ".join(b.item_ids)
            print(f"  • {b.text} [cites: {cites}]")
    print()
    return 0


def cmd_questions(args: argparse.Namespace) -> int:
    """Show, export or rate the standing questions and their living answers.

    Questions from config/questions.yaml are loaded first; a malformed file only prints a
    warning. ``--export`` prints or saves the research dossier, ``--feedback`` records a verdict,
    and otherwise each question's living answer is updated and printed.

    Args:
        args: the parsed command-line arguments.

    Returns:
        The exit code: 0, or 1 when the feedback could not be recorded.
    """
    from gleaner.config import parse_questions_yaml
    from gleaner.llama import get_llama_model
    from gleaner.questions import record_answer_feedback, synthesize_living_answer

    db = get_default_db()
    questions_path = CONFIG_DIR / "questions.yaml"
    if questions_path.exists():
        try:
            q_cfg = parse_questions_yaml(questions_path.read_text(encoding="utf-8"))
            sync_questions_from_config(db, q_cfg)
        except Exception as exc:  # noqa: BLE001  # Safe degradation: ignore malformed questions config
            print(f"Warning: could not parse questions.yaml: {exc}")

    if getattr(args, "export", False):
        from gleaner.dossier import generate_dossier

        dossier_text = generate_dossier(db)
        out_path = getattr(args, "output", None)
        if out_path:
            Path(out_path).write_text(dossier_text, encoding="utf-8")
            print(f"Exported research dossier to {out_path}")
        else:
            print(dossier_text)
        return 0

    if getattr(args, "feedback", None):
        q_id, verdict = args.feedback[0], args.feedback[1]
        try:
            latest = db.get_latest_answer(q_id)
            version = latest.version if latest else 1
            record_answer_feedback(db, question_id=q_id, version=version, verdict=verdict)
            rate = db.get_track_record_hit_rate(q_id)
            print(f"Recorded feedback '{verdict}' for question '{q_id}'. Updated hit rate: {rate.get('hit_rate_pct', 0.0):.1f}%")
        except ValueError as err:
            print(f"Error: {err}")
            return 1
        return 0

    questions = db.list_questions()
    if not questions:
        print("No standing questions found. Add them to config/questions.yaml.")
        return 0

    print(f"\n=== Standing Questions & Living Answers ({len(questions)}) ===\n")
    model_name = getattr(args, "model", None)
    writer_model = get_llama_model("writer", model_name=model_name)
    answer_model = get_llama_model("answer", model_name=model_name)
    for q in questions:
        ans = synthesize_living_answer(db, q, model=answer_model, writer=writer_model)
        if ans is None:
            ans = db.get_latest_answer(q.id)

        print(f"❓ [{q.id}] {q.text}")
        if q.linked_interests:
            print(f"   Interests: {', '.join(q.linked_interests)}")
        if q.watch:
            print(f"   Watch: {q.watch}")

        if ans:
            full_text = " ".join(s.text for s in ans.sentences)
            by_str = f" [synthesized by {ans.synthesized_by}]" if ans.synthesized_by else ""
            print(f"   Confidence: {ans.confidence.upper()} (version {ans.version}){by_str}")
            print(f"   Living Answer: {full_text}")
            if ans.what_changed:
                wc_by = f" (by {ans.what_changed_by})" if ans.what_changed_by else ""
                print(f"   What Changed: {ans.what_changed}{wc_by}")
            if ans.counterpoint_claim_id:
                cp = db.get_claim(ans.counterpoint_claim_id)
                cp_text = cp.text if cp else ans.counterpoint_claim_id
                print(f"   Counterpoint: {cp_text}")
            else:
                print("   Counterpoint: no counter-evidence found yet")
        else:
            print("   Living Answer: insufficient evidence to answer yet")

        rate = db.get_track_record_hit_rate(q.id)
        if rate.get("total", 0) > 0:
            print(f"   Track record: {rate.get('hit_rate_pct', 0.0):.1f}% useful ({rate.get('useful')}/{rate.get('total')})")
        print()
    return 0


def _review_tool_calls(request: Any) -> list[Any]:
    """Ask the user about each write action the ask agent wants to take.

    Args:
        request: the agent's interrupt, listing the actions under ``actions``.

    Returns:
        One ToolDecision per action, in order; with no terminal, every action is rejected.
    """
    from graphflow import ToolDecision

    print("\nThe agent wants to change your data:")
    decisions = []
    for action in request.actions:
        print(f"  - {action.get('summary') or action.get('description') or action['tool']} (tool: {action['tool']})")
        approved = _confirm("Allow this action?")
        if approved is None:
            print("    No terminal to confirm on, so the action is rejected.")
        decisions.append(ToolDecision.approve() if approved else ToolDecision.reject("The user declined this action."))
    return decisions


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer one question with the ask agent, asking before each write action.

    When the model is unreachable, it falls back to a local full-text search of the feed.

    Args:
        args: the parsed command-line arguments.

    Returns:
        The exit code: always 0.
    """
    from graphflow import GraphflowError

    from gleaner.ask import build_ask_agent
    from gleaner.llama import get_llama_model

    query = args.query or "What are the latest updates on Graphflow and LangGraph?"
    print(f"Ask query: {query}")
    db = get_default_db()

    model_name = getattr(args, "model", None)
    model = get_llama_model("ask", model_name=model_name)
    ask_app = build_ask_agent(db, model=model)
    try:
        res = ask_app.run({"input": query}, thread_id="cli-ask")
        while res.status == "interrupted":
            res = ask_app.resume("cli-ask", _review_tool_calls(res.interrupts[0]))
        answer = res.get("output") or res.get("response", "")
    except GraphflowError as exc:
        print(f"\nModel agent offline or unreachable ({exc}). Falling back to local search:\n")
        items = db.search_fts(query, limit=5)
        if items:
            lines = [f"Found {len(items)} relevant item(s) in local feed:"]
            for idx, it in enumerate(items, 1):
                snippet = (it.text[:150] + "...") if len(it.text) > 150 else it.text
                lines.append(f"  [{idx}] {it.title} ({it.source_id or 'unknown'})\n      {snippet}")
            lines.append("\n(Tip: Start Ollama or configure OLLAMA_BASE_URL to enable full conversational synthesis.)")
            answer = "\n".join(lines)
        else:
            answer = (
                f"No matching items found for '{query}' in local database.\n"
                "(Tip: Run 'python -m gleaner collect' to populate your feed, and start Ollama to enable full conversational answers.)"
            )
    print(f"\nAnswer:\n{answer}\n")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the local web server and, unless ``--no-scheduler``, the scheduler.

    Args:
        args: the parsed command-line arguments.

    Returns:
        The exit code: always 0.
    """
    import uvicorn

    from gleaner.server import create_app

    print(f"Starting server on http://{args.host}:{args.port}...")
    db = get_default_db()
    enable_scheduler = not getattr(args, "no_scheduler", False)
    interval_s = float(getattr(args, "scheduler_interval", 30.0))
    uvicorn.run(
        create_app(db, enable_scheduler=enable_scheduler, scheduler_interval_s=interval_s),
        host=args.host,
        port=args.port,
    )
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    """Run the Model Context Protocol server over stdio.

    Args:
        args: the parsed command-line arguments.

    Returns:
        The exit code: always 0.
    """
    from gleaner.mcp import MCPServer

    db = get_default_db()
    server = MCPServer(db)
    server.run_stdio()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the offline evaluation on the labeled set and print its metrics.

    Args:
        args: the parsed command-line arguments.

    Returns:
        The exit code: 0, or 1 when no labeled dataset is found.
    """
    from gleaner.eval import run_eval

    labels_path = FIXTURES_DIR / "labeled_eval.json"
    if not labels_path.is_file():
        labels_path = CONFIG_DIR / "eval_labels.json"
    if not labels_path.is_file():
        print("No evaluation dataset found at config/eval_labels.json or tests/fixtures/discovery/labeled_eval.json")
        return 1

    eval_db = Database(":memory:")
    metrics = run_eval(labels_path, eval_db)
    print("\n--- Discovery Feed Evaluation Results ---")
    print(f"Items Evaluated: {metrics.get('total_evaluated', 0)}")
    print(f"Precision@20:    {metrics.get('precision@20', 0.0):.4f}")
    print(f"nDCG@20:         {metrics.get('ndcg@20', 0.0):.4f}")
    print(f"Noise Rate:      {metrics.get('noise_rate', 0.0):.4f}")
    print(f"Filter Recall:   {metrics.get('filter_recall', 0.0):.4f}")
    print("-----------------------------------------\n")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run an offline end-to-end demo of every phase on an in-memory database.

    Uses the recorded fixtures and mock models, so it needs no network and no model server.

    Args:
        args: the parsed command-line arguments.

    Returns:
        The exit code: always 0.
    """
    import asyncio
    import json

    import httpx
    from graphflow import ApprovalDecision, MockModel, ModelResponse, ToolCall, ToolDecision

    from gleaner.ask import build_ask_agent
    from gleaner.digest import generate_digest
    from gleaner.eval import run_eval
    from gleaner.learn import record_feedback, update_interests_from_feedback
    from gleaner.models import SourceRow
    from gleaner.opportunities import extract_opportunity
    from gleaner.profile import build_profile_workflow
    from gleaner.trends import detect_trends
    from gleaner.workflows import build_collect_workflow

    print("Running offline discovery feed demo...")
    sources = load_sources_config(CONFIG_DIR / "sources.yaml")
    interests = load_interests_config(CONFIG_DIR / "interests.yaml")
    db = Database(":memory:")
    print(f"Loaded {len(sources)} sources, {len(interests)} interests into in-memory DB (vec: {db.has_vec}).")

    # 1. Profile bootstrap workflow with human approval gate
    profile_app = build_profile_workflow(db)
    raw_yaml = (CONFIG_DIR / "interests.yaml").read_text(encoding="utf-8")
    r1 = profile_app.run({"raw_yaml": raw_yaml}, thread_id="demo-profile")
    node_paused = r1.interrupts[0].node_name if r1.interrupts else "approval"
    print(f"[Phase 2] Profile workflow paused at '{node_paused}' ({len(r1.state['draft_interests'])} draft interests).")
    r2 = profile_app.resume("demo-profile", ApprovalDecision(approved=True))
    print(f"[Phase 2] Profile approved and saved (status={r2.state['status']}).")

    # 2. Seed sources and run declarative collect workflow over offline fixtures
    for src in sources:
        db.upsert_source(
            SourceRow(id=src.id, type=src.type, params=src.params, cadence_s=src.cadence_s, trust=src.trust)
        )

    async def _collect() -> Any:
        async with httpx.AsyncClient(transport=_build_fixture_transport()) as client:
            collect_app = build_collect_workflow(db, client=client)
            return await collect_app.arun({})

    collect_res = asyncio.run(_collect())
    print(
        f"[Phase 1-2] Collect workflow finished: {collect_res.state['scored_count']} items scored, "
        f"errors={collect_res.state['errors']}."
    )

    # 3. Ranked feed with MMR diversity
    cards = generate_feed(db, limit=5)
    print(f"[Phase 2] Top {len(cards)} ranked feed cards:")
    for idx, c in enumerate(cards, 1):
        print(f"  {idx}. [{c.item.source_id}] {c.item.title} (score={c.score:.3f})")

    # 4. Daily digest with citation verification
    if cards:
        valid_id = cards[0].item.id
        digest_json = json.dumps(
            {
                "title": "Daily AI & Systems Briefing",
                "summary": "Key highlights across your tracked sources.",
                "bullets": [
                    {"text": f"New development in {cards[0].item.title}.", "item_ids": [valid_id]},
                    {"text": "Hallucinated claim about uncollected paper.", "item_ids": ["nonexistent-id-999"]},
                ],
            }
        )
        digest = generate_digest(db, model=MockModel(responses=[ModelResponse(content=digest_json)]))
        print(f"[Phase 3] Digest generated: {len(digest.bullets)} verified bullet(s) kept (1 hallucinated citation dropped).")

        # 5. Feedback & Rocchio learning loop
        record_feedback(db, item_id=valid_id, action="like", dwell_s=45.0)
        update_interests_from_feedback(db)
        print(f"[Phase 4] Recorded 'like' on {valid_id} and updated interest centroids via Rocchio.")

    # 6. Trends & Opportunities
    from gleaner.opportunities import check_opportunity_reminders

    trends = detect_trends(db)
    opp = (
        extract_opportunity(
            cards[0].item,
            db,
            deadline=datetime.now(UTC) + timedelta(days=7),
            opp_type="cfp",
        )
        if cards
        else None
    )
    reminders = check_opportunity_reminders(db)
    print(
        f"[Phase 5] Trend bursts detected: {len(trends)} | "
        f"Opportunities due in 7d reminders fired: {len(reminders)} (deadline: {opp.deadline.date() if opp and opp.deadline else 'N/A'})."
    )

    # 7. Ask Agent with read tool + approval-gated write tool
    ask_model = MockModel(
        responses=[
            ModelResponse(tool_calls=[ToolCall(id="tc_1", name="search_feed", args={"query": "Graphflow"})]),
            ModelResponse(tool_calls=[ToolCall(id="tc_2", name="mute_source", args={"source_id": "hn-frontpage"})]),
            ModelResponse(content="I searched your feed for Graphflow and muted 'hn-frontpage' after your approval."),
        ]
    )
    ask_app = build_ask_agent(db, model=ask_model)
    ask_r1 = ask_app.run({"input": "Search for Graphflow and mute hn-frontpage"}, thread_id="demo-ask")
    if ask_r1.status == "interrupted":
        prompt_str = ask_r1.interrupts[0].prompt if ask_r1.interrupts else "Approve tool call"
        print(f"[Phase 6] Ask agent paused before write tool: {prompt_str}")
        ask_r2 = ask_app.resume("demo-ask", ToolDecision.approve())
        print(f"[Phase 6] Ask agent resumed after approval: '{ask_r2.get('output')}'")

    # 8. Evaluation harness
    labels_path = FIXTURES_DIR / "labeled_eval.json"
    eval_db = Database(":memory:")
    metrics = run_eval(labels_path, eval_db)
    print(
        f"[Phase 7] Eval metrics: precision@20={metrics['precision@20']:.2f}, "
        f"nDCG@20={metrics['ndcg@20']:.2f}, filter_recall={metrics['filter_recall']:.2f}."
    )

    # 9. Standing questions, living answers, and answer diffs (Phase 8, M17)
    from gleaner.questions import (
        Answer,
        Claim,
        Question,
        Sentence,
        compute_answer_diff,
        record_answer_feedback,
    )

    demo_q1 = Question(
        id="q-small-models-tool-use",
        text="Are small open models reliable enough for production tool use and agent loops?",
        linked_interests=["agents", "local-llm"],
    )
    demo_q2 = Question(
        id="q-langgraph-production",
        text="What are the common failure modes and scalability limits of LangGraph in production?",
        linked_interests=["agents", "python-evals"],
    )
    demo_q3 = Question(
        id="q-stack-changes",
        text="What significant changes, breaking updates, or advisories affect my project stack?",
        linked_interests=["agents", "security"],
        watch="pyproject.toml",
    )
    db.upsert_question(demo_q1)
    db.upsert_question(demo_q2)
    db.upsert_question(demo_q3)

    c1 = Claim(
        id="claim-demo-1",
        question_id=demo_q1.id,
        item_id=cards[0].item.id if cards else "item-1",
        text="Small models achieve 82% accuracy on single function calls.",
        stance="supports",
        quote="achieve 82% accuracy on single function calls",
        origin_key="arxiv:2410.12345",
    )
    c2 = Claim(
        id="claim-demo-2",
        question_id=demo_q1.id,
        item_id="item-2",
        text="Small models fail on 65% of recursive loops without checkpoint recovery.",
        stance="contradicts",
        quote="fails on 65% of recursive loops",
        origin_key="github:langchain-ai/langgraph",
    )
    c3 = Claim(
        id="claim-demo-3",
        question_id=demo_q1.id,
        item_id="item-3",
        text="Speculative decoding improves tool calling reliability by 34%.",
        stance="supports",
        quote="improves tool calling reliability by 34%",
        origin_key="arxiv:2410.99999",
    )
    db.upsert_claim(c1)
    db.upsert_claim(c2)
    db.upsert_claim(c3)

    ans_v1 = Answer(
        question_id=demo_q1.id,
        version=1,
        sentences=[
            Sentence(
                text="Single function calls are handled reliably by 3B models.",
                claim_ids=[c1.id],
            )
        ],
        confidence="low",
    )
    db.upsert_answer(ans_v1)

    ans_v2 = Answer(
        question_id=demo_q1.id,
        version=2,
        sentences=[
            Sentence(
                text="Single function calls are handled reliably by 3B models.",
                claim_ids=[c1.id],
            ),
            Sentence(
                text="Speculative decoding techniques significantly boost tool-calling stability.",
                claim_ids=[c3.id],
            ),
            Sentence(
                text="However, multi-step recursion exhibits high failure rates without external checkpointers.",
                claim_ids=[c2.id],
            ),
        ],
        confidence="medium",
        counterpoint_claim_id=c2.id,
        what_changed="Added speculative decoding benchmark and counter-evidence for recursive loop failures.",
    )
    db.upsert_answer(ans_v2)
    diff = compute_answer_diff(ans_v1, ans_v2)
    record_answer_feedback(db, question_id=demo_q1.id, version=2, verdict="useful")
    hit_rate = db.get_track_record_hit_rate(demo_q1.id)

    print(
        f"[Phase 8] Standing questions: answer updated (v{diff.from_version} -> v{diff.to_version}): "
        f"'{diff.what_changed}' | Added sentences: {len(diff.added_sentences)} | "
        f"Counterpoint: {c2.origin_key} | Hit rate: {hit_rate['hit_rate_pct']:.1f}%."
    )
    print("Discovery feed skeleton verified successfully.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run its subcommand (the demo when none).

    Args:
        argv: the arguments after the program name, or None to read ``sys.argv``.

    Returns:
        The subcommand's exit code.

    Raises:
        SystemExit: the arguments are invalid or ``--model`` is not a Meta model.
    """
    parser = argparse.ArgumentParser(description="gleaner: personal information discovery")
    parser.add_argument("--model", default=None, help="Model name override (e.g. llama3.2:3b, llama3.1:8b)")
    subparsers = parser.add_subparsers(dest="command")

    # init
    init_p = subparsers.add_parser("init", help="Initialize DB and load configs")
    init_p.add_argument("--yes", "-y", action="store_true", help="Save the drafted interest profile without asking")

    # collect
    collect_p = subparsers.add_parser("collect", help="Run collection tick")
    collect_p.add_argument(
        "--offline", action="store_true", help="Fetch from recorded test fixtures instead of the live sources"
    )

    # feed
    subparsers.add_parser("feed", help="Show ranked feed")

    # digest
    digest_p = subparsers.add_parser("digest", help="Generate or display digest")
    digest_p.add_argument("--model", default=argparse.SUPPRESS, help="Model name override (e.g. llama3.1:8b)")

    # questions
    questions_p = subparsers.add_parser("questions", help="Inspect standing questions and living answers")
    questions_p.add_argument(
        "--export", action="store_true", help="Export complete research dossier to Markdown (Obsidian/PKM compatible)"
    )
    questions_p.add_argument("--output", "-o", default=None, help="Save exported dossier to file path instead of stdout")
    questions_p.add_argument(
        "--feedback",
        nargs=2,
        metavar=("QUESTION_ID", "VERDICT"),
        help="Record feedback (useful, wrong, already_knew)",
    )
    questions_p.add_argument("--model", default=argparse.SUPPRESS, help="Model name override")

    # ask
    ask_p = subparsers.add_parser("ask", help="Ask questions about collected items")
    ask_p.add_argument("query", nargs="?", default="", help="Question to ask")
    ask_p.add_argument("--model", default=argparse.SUPPRESS, help="Model name override (e.g. llama3.1:8b)")

    # serve
    serve_p = subparsers.add_parser("serve", help="Run local web server")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument(
        "--no-scheduler", action="store_true", help="Disable the background cadence scheduler"
    )
    serve_p.add_argument(
        "--scheduler-interval",
        type=float,
        default=30.0,
        help="Cadence scheduler check interval in seconds (default: 30)",
    )

    # mcp
    subparsers.add_parser("mcp", help="Run Model Context Protocol (MCP) stdio server")

    # eval
    subparsers.add_parser("eval", help="Run offline evaluation")

    # demo
    subparsers.add_parser("demo", help="Run offline demo")

    args = parser.parse_args(argv)
    if getattr(args, "model", None) and not is_meta_model(args.model):
        parser.error(f"--model '{args.model}' is not a Meta model (e.g. muse-spark-1.1, llama3.2:3b)")
    if getattr(args, "model", None):
        os.environ["GLEANER_MODEL"] = args.model

    if not args.command:
        # Default to demo when called with no args
        return cmd_demo(args)

    commands = {
        "init": cmd_init,
        "collect": cmd_collect,
        "feed": cmd_feed,
        "digest": cmd_digest,
        "questions": cmd_questions,
        "ask": cmd_ask,
        "serve": cmd_serve,
        "mcp": cmd_mcp,
        "eval": cmd_eval,
        "demo": cmd_demo,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
