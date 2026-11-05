"""Literate sources (docs/literate/*.md) must tangle to the committed modules.

The documents are the source of truth (ADR-0006). If someone edits a ``.py`` file
without its document, or the document without re-tangling, these tests fail.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LITERATE_DIR = ROOT / "docs" / "literate"

_spec = importlib.util.spec_from_file_location("tangle", ROOT / "scripts" / "tangle.py")
assert _spec and _spec.loader
tangle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tangle)

# The modules ADR-0006 puts under literate control.
LITERATE_MODULES = {
    "gleaner/chunking.py",
    "gleaner/dossier.py",
    "gleaner/filters.py",
    "gleaner/judge.py",
    "gleaner/learn.py",
    "gleaner/mcp.py",
    "gleaner/questions.py",
    "gleaner/scoring.py",
    "gleaner/security.py",
    "gleaner/workflows.py",
}


# --- the tangler itself -------------------------------------------------------


def _doc(body: str) -> str:
    return body.replace("~~~", "```")


def test_expands_references_with_indentation():
    doc = _doc(
        "Intro prose.\n\n"
        "~~~python\n<<out/a.py>>=\ndef f():\n    <<body of f>>\n~~~\n\n"
        "Why the body looks like this.\n\n"
        "~~~python\n<<body of f>>=\nx = 1\n\nreturn x\n~~~\n"
    )
    files = tangle.tangle_text(doc)
    # Blank lines inside an indented chunk stay empty (no trailing spaces).
    assert files == {"out/a.py": "def f():\n    x = 1\n\n    return x\n"}


def test_append_definition_concatenates_in_document_order():
    doc = _doc(
        "~~~python\n<<out/a.py>>=\n<<imports>>\n~~~\n"
        "~~~python\n<<imports>>=\nimport os\n~~~\n"
        "~~~python\n<<imports>>+=\nimport re\n~~~\n"
    )
    assert tangle.tangle_text(doc) == {"out/a.py": "import os\nimport re\n"}


def test_untagged_code_blocks_are_illustration_only():
    doc = _doc(
        "~~~python\nprint('example, not part of the program')\n~~~\n"
        "~~~python\n<<out/a.py>>=\nx = 1\n~~~\n"
    )
    assert tangle.tangle_text(doc) == {"out/a.py": "x = 1\n"}


def test_code_starting_with_backticks_does_not_close_the_fence():
    doc = _doc('~~~python\n<<out/a.py>>=\n<<strip>>\n~~~\n~~~python\n<<strip>>=\n~~~json" in s\nx = 1\n~~~\n')
    assert tangle.tangle_text(doc) == {"out/a.py": '```json" in s\nx = 1\n'}


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("~~~python\n<<out/a.py>>=\n<<missing>>\n~~~\n", "undefined chunk"),
        (
            "~~~python\n<<out/a.py>>=\nx = 1\n~~~\n~~~python\n<<orphan>>=\ny = 2\n~~~\n",
            "never used",
        ),
        (
            "~~~python\n<<out/a.py>>=\n<<a>>\n~~~\n~~~python\n<<a>>=\n<<a>>\n~~~\n",
            "cycle",
        ),
        (
            "~~~python\n<<out/a.py>>=\nx\n~~~\n~~~python\n<<out/a.py>>=\ny\n~~~\n",
            "defined twice",
        ),
        ("~~~python\n<<extra>>+=\nx\n~~~\n", "before it is defined"),
    ],
)
def test_malformed_documents_are_rejected(body, message):
    with pytest.raises(tangle.TangleError, match=message):
        tangle.tangle_text(_doc(body))


def test_check_reports_drift_without_writing(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(_doc("~~~python\n<<mod.py>>=\nx = 1\n~~~\n"))
    target = tmp_path / "mod.py"
    target.write_text("x = 2\n")

    assert tangle.main(["--check", "--root", str(tmp_path), str(doc)]) == 1
    assert target.read_text() == "x = 2\n"  # --check never writes

    assert tangle.main(["--root", str(tmp_path), str(doc)]) == 0
    assert target.read_text() == "x = 1\n"
    assert tangle.main(["--check", "--root", str(tmp_path), str(doc)]) == 0


# --- the project's literate sources -------------------------------------------


def _project_outputs() -> dict[str, tuple[Path, str]]:
    outputs: dict[str, tuple[Path, str]] = {}
    for doc in sorted(LITERATE_DIR.glob("*.md")):
        for path, text in tangle.tangle_text(doc.read_text(encoding="utf-8")).items():
            assert path not in outputs, f"{path} is tangled by two documents"
            outputs[path] = (doc, text)
    return outputs


@pytest.mark.skipif(not any(LITERATE_DIR.glob("*.md")), reason="docs/literate documents not yet populated")
def test_every_literate_module_has_a_document():
    assert set(_project_outputs()) == LITERATE_MODULES


@pytest.mark.skipif(not any(LITERATE_DIR.glob("*.md")), reason="docs/literate documents not yet populated")
@pytest.mark.parametrize("module", sorted(LITERATE_MODULES))
def test_document_tangles_to_the_committed_module(module):
    doc, text = _project_outputs()[module]
    committed = (ROOT / module).read_text(encoding="utf-8")
    assert text == committed, (
        f"{module} differs from {doc.relative_to(ROOT)}; edit the document and run "
        "./scripts/tangle (ADR-0006)"
    )


def test_tests_the_documents_cite_exist():
    """A document's Verification list names real tests: a reader runs them to check its claims."""
    import re

    real = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        real |= set(re.findall(r"^(?:async )?def (test_\w+)", path.read_text(encoding="utf-8"), re.M))
    cited = {
        (doc.name, name)
        for doc in LITERATE_DIR.glob("*.md")
        for name in re.findall(r"`(test_\w+)`", doc.read_text(encoding="utf-8"))
    }
    assert sorted(c for c in cited if c[1] not in real) == []
