"""The package runs, and graphflow at the pinned commit offers what gleaner relies on."""

import inspect
import subprocess
import sys

import gleaner


def test_version_from_cli():
    out = subprocess.run(
        [sys.executable, "-m", "gleaner", "--version"], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == f"gleaner {gleaner.__version__}"


def test_graphflow_reaches_llama_through_openai_compatible_endpoints():
    from graphflow import MockModel, OpenAIModel

    assert "base_url" in inspect.signature(OpenAIModel.__init__).parameters
    assert MockModel is not None


def test_import_and_version_do_not_load_the_app_stack():
    """`import gleaner` and `gleaner --version` stay light: the app modules load on first use."""
    code = (
        "import sys, gleaner\n"
        "from gleaner.__main__ import main\n"
        "main(['--version'])\n"
        "heavy = sorted(m for m in sys.modules if m == 'graphflow' or (m.startswith('gleaner.') and m != 'gleaner.__main__'))\n"
        "print('HEAVY', heavy)\n"
        "assert gleaner.Database.__name__ == 'Database'\n"
        "print('LOADED', 'gleaner.db' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "HEAVY []" in out.stdout
    assert "LOADED True" in out.stdout


def test_version_flag_is_only_read_before_the_command(monkeypatch, capsys):
    """An argument after the subcommand that happens to be `-V` is the app's, not a version request."""
    import gleaner.__main__ as entry
    from gleaner import cli

    seen = []
    monkeypatch.setattr(cli, "main", lambda argv: seen.append(argv) or 0)
    assert entry.main(["ask", "--", "-V"]) == 0
    assert seen == [["ask", "--", "-V"]]
    assert "gleaner " not in capsys.readouterr().out

    assert entry.main(["-V"]) == 0
    assert capsys.readouterr().out.strip() == f"gleaner {gleaner.__version__}"
