"""Tests for discovery CLI commands (init, collect, ask, eval)."""

import argparse
import sys
from datetime import UTC, datetime

from gleaner import cli
from gleaner.db import Database
from gleaner.models import Item, SourceRow


def test_cmd_collect_multiple_runs(tmp_path, monkeypatch):
    """Running collect sequentially must not fail with UNIQUE constraint on runs.id."""
    db_path = tmp_path / "collect_multi.db"
    db = Database(db_path)
    src = SourceRow(
        id="src-1",
        type="rss",
        params={"url": "http://example.com/feed.xml"},
        cadence_s=0,
    )
    db.upsert_source(src)

    # Point get_default_db to this test database
    monkeypatch.setattr(cli, "get_default_db", lambda: Database(db_path))

    args = argparse.Namespace(offline=True)
    # First collect run
    ret1 = cli.cmd_collect(args)
    assert ret1 == 0

    # Second collect run should succeed and not crash on UNIQUE constraint
    ret2 = cli.cmd_collect(args)
    assert ret2 == 0


def test_cmd_ask_offline_model_fallback(tmp_path, monkeypatch, capsys):
    """When LLM endpoint is unreachable, cmd_ask should fall back gracefully without unhandled exception."""
    db_path = tmp_path / "ask_fallback.db"
    db = Database(db_path)
    src = SourceRow(id="src-1", type="rss", params={}, cadence_s=3600)
    db.upsert_source(src)
    item = Item(
        id="item-offline-1",
        source_id="src-1",
        title="Offline Resilience in LangGraph",
        text="How to handle network partitions and offline models.",
    )
    db.upsert_item(item)

    monkeypatch.setattr(cli, "get_default_db", lambda: Database(db_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:19999/v1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:19999/v1")

    args = argparse.Namespace(query="How to handle offline models?")
    ret = cli.cmd_ask(args)
    assert ret == 0

    out = capsys.readouterr().out
    assert "Offline Resilience in LangGraph" in out or "Offline" in out or "fallback" in out.lower()


def test_cmd_eval_runs_and_reports_metrics(tmp_path, monkeypatch, capsys):
    """cmd_eval should evaluate benchmark cleanly even if gleaner.db is already populated."""
    db_path = tmp_path / "populated.db"
    db = Database(db_path)
    src = SourceRow(id="src-1", type="rss", params={}, cadence_s=3600)
    db.upsert_source(src)
    db.upsert_item(Item(id="other-id-1", source_id="src-1", title="Unrelated item", text="Unrelated text"))

    monkeypatch.setattr(cli, "get_default_db", lambda: Database(db_path))

    args = argparse.Namespace()
    ret = cli.cmd_eval(args)
    assert ret == 0

    out = capsys.readouterr().out
    assert "Precision@20:" in out
    assert "nDCG@20:" in out
    assert "Precision@20:    0.0000" not in out


def test_cmd_feed_empty_and_populated(tmp_path, monkeypatch, capsys):
    """cmd_feed should cleanly render empty and populated feeds."""
    db_path = tmp_path / "feed.db"
    db = Database(db_path)
    monkeypatch.setattr(cli, "get_default_db", lambda: Database(db_path))

    # Empty feed
    args = argparse.Namespace()
    assert cli.cmd_feed(args) == 0
    assert "empty" in capsys.readouterr().out.lower()

    # Populated feed (item length >= 20 characters to pass Stage A filter)
    src = SourceRow(id="src-1", type="rss", params={}, cadence_s=3600)
    db.upsert_source(src)
    item = Item(id="it-1", source_id="src-1", title="Feed Item 1", text="Substantial text for feed 1 with more than twenty characters.")
    db.upsert_item(item)
    from gleaner.workflows import process_and_score_items
    process_and_score_items(db, [item])

    assert cli.cmd_feed(args) == 0
    out = capsys.readouterr().out
    assert "Ranked Feed" in out
    assert "Feed Item 1" in out


def test_cmd_digest_fallback(tmp_path, monkeypatch, capsys):
    """cmd_digest runs and prints daily briefing with fallback citations."""
    db_path = tmp_path / "digest.db"
    db = Database(db_path)
    monkeypatch.setattr(cli, "get_default_db", lambda: Database(db_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:19999/v1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:19999/v1")

    src = SourceRow(id="src-1", type="rss", params={}, cadence_s=3600)
    db.upsert_source(src)
    item = Item(id="it-1", source_id="src-1", title="Digest Item 1", text="Text for digest 1")
    db.upsert_item(item)

    args = argparse.Namespace()
    assert cli.cmd_digest(args) == 0
    out = capsys.readouterr().out
    assert "Daily Briefing" in out


def test_cli_main_dispatch():
    """Verify main() argument parsing and dispatch."""
    import pytest
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0



import io  # noqa: E402

import pytest  # noqa: E402
from graphflow import MockModel, ModelResponse, OpenAIModel, ToolCall  # noqa: E402


class _TTY(io.StringIO):
    """Stdin that answers prompts from a script and claims to be a terminal."""

    def isatty(self) -> bool:
        return True


def _capture_collect(monkeypatch, tmp_path):
    """Point collect at a temp DB and record how run_collect_tick is called."""
    import gleaner.workflows

    db_path = tmp_path / "collect_mode.db"
    Database(db_path)
    monkeypatch.setattr(cli, "get_default_db", lambda: Database(db_path))
    calls = []

    async def fake_tick(
        db, client=None, model=None, embedder=None, writer=None, answer_model=None, progress_callback=None
    ):
        calls.append({"client": client, "model": model, "writer": writer, "answer_model": answer_model})
        return {"sources_run": 0, "items_stored": 0, "sources_failed": 0}

    monkeypatch.setattr(gleaner.workflows, "run_collect_tick", fake_tick)
    return calls


def test_collect_runs_live_with_the_llama_judge_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("GLEANER_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    calls = _capture_collect(monkeypatch, tmp_path)
    assert cli.main(["collect"]) == 0
    assert calls[0]["client"] is None  # real network client, owned by the tick
    assert isinstance(calls[0]["model"], OpenAIModel)
    assert calls[0]["model"].model_name == "llama3.2:3b"
    assert calls[0]["writer"].model_name == "llama3.2:3b"
    assert calls[0]["answer_model"].model_name == "muse-spark-1.3"


def test_collect_offline_uses_recorded_fixtures(tmp_path, monkeypatch):
    import httpx

    calls = _capture_collect(monkeypatch, tmp_path)
    assert cli.main(["collect", "--offline"]) == 0
    client = calls[0]["client"]
    assert isinstance(client, httpx.AsyncClient)
    assert isinstance(client._transport, httpx.MockTransport)


@pytest.mark.parametrize("command", ["digest", "ask"])
def test_root_model_flag_survives_the_subcommand(command, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, f"cmd_{command}", lambda args: seen.update(model=args.model) or 0)
    monkeypatch.setenv("GLEANER_MODEL", "restored-after-test")  # main() writes it; monkeypatch restores it
    assert cli.main(["--model", "llama3.1:70b", command]) == 0
    assert seen["model"] == "llama3.1:70b"

    assert cli.main([command, "--model", "llama3.2:1b"]) == 0
    assert seen["model"] == "llama3.2:1b"

    assert cli.main([command]) == 0
    assert seen["model"] is None


def _init_env(tmp_path, monkeypatch):
    (tmp_path / "sources.yaml").write_text(
        """sources:
  - id: "on-src"
    type: "rss"
    params: { url: "https://example.com/on.xml" }
    cadence: "1h"
  - id: "off-src"
    type: "rss"
    params: { url: "https://example.com/off.xml" }
    cadence: "1h"
    enabled: false
""",
        encoding="utf-8",
    )
    (tmp_path / "interests.yaml").write_text(
        """interests:
  - id: "agents"
    label: "Agent Frameworks"
    description: "Agent orchestration frameworks."
    keywords: ["agent"]
""",
        encoding="utf-8",
    )
    db_path = tmp_path / "init.db"
    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli, "get_default_db", lambda: Database(db_path))
    return Database(db_path)


def _interest_count(db):
    with db.connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM interests;").fetchone()[0]


def test_init_keeps_disabled_sources_disabled(tmp_path, monkeypatch):
    db = _init_env(tmp_path, monkeypatch)
    assert cli.main(["init", "--yes"]) == 0
    assert db.get_source("on-src").enabled is True
    assert db.get_source("off-src").enabled is False
    assert _interest_count(db) == 1


def test_init_pauses_sources_removed_from_the_config(tmp_path, monkeypatch):
    db = _init_env(tmp_path, monkeypatch)
    db.upsert_source(SourceRow(id="removed-src", type="rss", params={"url": "https://example.com/feed.xml"}, failures=3))

    assert cli.main(["init", "--yes"]) == 0

    removed = db.get_source("removed-src")
    assert removed is not None  # kept for its fetch history
    assert removed.enabled is False
    assert db.get_source("on-src").enabled is True


def test_init_keeps_the_stack_watch_source(tmp_path, monkeypatch):
    db = _init_env(tmp_path, monkeypatch)
    db.upsert_source(SourceRow(id="osv", type="osv", params={}))  # made by the stack watch, not sources.yaml

    assert cli.main(["init", "--yes"]) == 0

    assert db.get_source("osv").enabled is True


def test_init_shows_the_draft_profile_and_saves_only_on_confirmation(tmp_path, monkeypatch, capsys):
    db = _init_env(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "stdin", _TTY("n\n"))
    cli.main(["init"])
    out = capsys.readouterr().out
    assert "Agent Frameworks" in out
    assert "rejected" in out
    assert _interest_count(db) == 0

    monkeypatch.setattr(sys, "stdin", _TTY("y\n"))
    assert cli.main(["init"]) == 0
    assert _interest_count(db) == 1


def test_init_without_a_terminal_needs_yes(tmp_path, monkeypatch, capsys):
    db = _init_env(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert cli.main(["init"]) == 1
    assert "--yes" in capsys.readouterr().out
    assert _interest_count(db) == 0


def _ask_env(tmp_path, monkeypatch):
    import gleaner.llama

    db_path = tmp_path / "ask_cli.db"
    db = Database(db_path)
    db.upsert_source(SourceRow(id="noisy-blog", type="rss", params={}, cadence_s=3600))
    monkeypatch.setattr(cli, "get_default_db", lambda: Database(db_path))
    model = MockModel(
        responses=[
            ModelResponse(tool_calls=[ToolCall(id="c1", name="mute_source", args={"source_id": "noisy-blog"})]),
            ModelResponse(content="Done."),
        ]
    )
    monkeypatch.setattr(gleaner.llama, "get_llama_model", lambda *a, **k: model)
    return db


def test_ask_asks_before_a_write_tool_and_respects_no(tmp_path, monkeypatch, capsys):
    db = _ask_env(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "stdin", _TTY("n\n"))
    assert cli.main(["ask", "mute noisy-blog"]) == 0
    out = capsys.readouterr().out
    assert "Mute source 'noisy-blog'" in out  # the pending action, from the interrupt
    assert "Done." in out
    assert db.get_source("noisy-blog").enabled is True


def test_ask_runs_the_write_tool_on_yes(tmp_path, monkeypatch):
    db = _ask_env(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "stdin", _TTY("y\n"))
    assert cli.main(["ask", "mute noisy-blog"]) == 0
    assert db.get_source("noisy-blog").enabled is False


def test_non_meta_model_flag_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--model", "Qwen/Qwen3.5-0.8B:latest", "feed"])
    assert exc.value.code == 2
    assert "not a Meta model" in capsys.readouterr().err


def test_init_keeps_each_sources_fetch_state(tmp_path, monkeypatch):
    db = _init_env(tmp_path, monkeypatch)
    due = datetime(2026, 11, 1, tzinfo=UTC)
    db.upsert_source(
        SourceRow(id="on-src", type="rss", params={"url": "https://example.com/on.xml"}, etag='"v7"', failures=2, next_due_at=due)
    )

    assert cli.main(["init", "--yes"]) == 0

    src = db.get_source("on-src")
    # a reset would refetch everything at once and forget the backoff of a failing feed
    assert (src.etag, src.failures, src.next_due_at.replace(tzinfo=UTC)) == ('"v7"', 2, due)


def test_loading_questions_yaml_keeps_questions_removed_in_the_app(tmp_path):
    """`init` and `questions` used to bring back every question the user had removed."""
    from gleaner.cli import sync_questions_from_config
    from gleaner.config import parse_questions_yaml
    from gleaner.db import Database
    from gleaner.models import Question

    db = Database(tmp_path / "q.db")
    db.upsert_question(Question(id="q-removed", text="Old?", active=False))
    cfg = parse_questions_yaml("questions:\n  - id: q-removed\n    text: Old?\n  - id: q-off\n    text: Off?\n    active: false\n"
                               "  - id: q-new\n    text: New?\n")

    sync_questions_from_config(db, cfg)

    assert {q.id for q in db.list_questions()} == {"q-new"}
