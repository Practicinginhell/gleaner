"""Tangle literate Markdown documents into source files (ADR-0006).

A document is ordinary Markdown. A fenced code block whose first line is
``<<chunk name>>=`` defines a chunk; ``<<chunk name>>+=`` appends to one defined
earlier. Inside a chunk, a line holding only ``<<other chunk>>`` (any indent) is
replaced by that chunk, every line re-indented to match. Chunks named like a path
(containing ``/`` or ending in ``.py``) are roots: each is written to that path,
relative to ``--root`` (the repo root by default). Code blocks without a header are
illustrations and are ignored.

    ./scripts/tangle                 # rewrite every module from docs/literate/*.md
    ./scripts/tangle --check         # exit 1 if any module differs from its document
    ./scripts/tangle docs/literate/filters.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FENCE = re.compile(r"^```")
HEADER = re.compile(r"^<<(?P<name>[^<>]+)>>(?P<op>\+?=)\s*$")
REFERENCE = re.compile(r"^(?P<indent>[ \t]*)<<(?P<name>[^<>]+)>>\s*$")
REPO_ROOT = Path(__file__).resolve().parents[1]


class TangleError(ValueError):
    """The document is malformed: an undefined, unused, cyclic or duplicate chunk."""


def parse_chunks(text: str) -> dict[str, list[str]]:
    """Collect chunk definitions, in document order, from a Markdown document.

    Args:
        text: The Markdown document.

    Returns:
        Each chunk's name mapped to its lines, ``+=`` continuations appended.

    Raises:
        TangleError: A chunk is defined twice, or continued before it is defined.
    """
    chunks: dict[str, list[str]] = {}
    current: list[str] | None = None
    in_fence = first_line = False
    for line in text.splitlines():
        # Open on any ``` line; close only on a bare ``` line, so code that merely
        # starts with backticks (security.py strips ``` from model output) is safe.
        if (not in_fence and FENCE.match(line)) or (in_fence and line.rstrip() == "```"):
            in_fence, first_line, current = not in_fence, not in_fence, None
            continue
        if not in_fence:
            continue
        if first_line:
            first_line = False
            if header := HEADER.match(line):
                name, op = header["name"].strip(), header["op"]
                if op == "=" and name in chunks:
                    raise TangleError(f"chunk <<{name}>> defined twice (use +=)")
                if op == "+=" and name not in chunks:
                    raise TangleError(f"<<{name}>>+= appears before it is defined")
                current = chunks.setdefault(name, [])
                continue
        if current is not None:
            current.append(line)
    return chunks


def is_root(name: str) -> bool:
    """Whether the chunk is written to a file: its name contains ``/`` or ends in ``.py``."""
    return "/" in name or name.endswith(".py")


def expand(name: str, chunks: dict[str, list[str]], used: set[str], stack: tuple[str, ...] = ()) -> list[str]:
    """Return chunk ``name`` with every reference replaced, recursively.

    Args:
        name: The chunk to expand.
        chunks: All chunk definitions.
        used: Collects every chunk expanded; updated in place.
        stack: The chunks being expanded around this one, to catch cycles.

    Returns:
        The expanded lines.

    Raises:
        TangleError: A reference is undefined or cyclic.
    """
    if name in stack:
        raise TangleError(f"cycle: {' -> '.join((*stack, name))}")
    if name not in chunks:
        raise TangleError(f"undefined chunk <<{name}>> (used in <<{stack[-1]}>>)")
    used.add(name)
    out: list[str] = []
    for line in chunks[name]:
        if ref := REFERENCE.match(line):
            indent = ref["indent"]
            sub_lines = expand(ref["name"].strip(), chunks, used, (*stack, name))
            out.extend(indent + sub if sub else sub for sub in sub_lines)
        else:
            out.append(line)
    return out


def tangle_text(text: str) -> dict[str, str]:
    """Map each root chunk's output path to its fully expanded text.

    Args:
        text: The Markdown document.

    Returns:
        Each root chunk's path mapped to its text.

    Raises:
        TangleError: The document is malformed, or a chunk is never used.
    """
    chunks = parse_chunks(text)
    used: set[str] = set()
    files = {name: "\n".join(expand(name, chunks, used)) + "\n" for name in chunks if is_root(name)}
    if unused := sorted(set(chunks) - used):
        raise TangleError(f"chunk(s) never used: {', '.join(unused)}")
    return files


def main(argv: list[str] | None = None) -> int:
    """Tangle the documents named on the command line, or all of docs/literate/.

    Args:
        argv: The command-line arguments; ``sys.argv[1:]`` when None.

    Returns:
        0 on success; 1 when a document is malformed or, with ``--check``, a file differs.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("docs", nargs="*", type=Path, help="default: docs/literate/*.md")
    parser.add_argument("--check", action="store_true", help="report drift, write nothing")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="output paths are relative to this")
    args = parser.parse_args(argv)

    docs = args.docs or sorted((REPO_ROOT / "docs" / "literate").glob("*.md"))
    status = 0
    for doc in docs:
        try:
            files = tangle_text(doc.read_text(encoding="utf-8"))
        except TangleError as exc:
            print(f"{doc}: {exc}", file=sys.stderr)
            status = 1
            continue
        for rel, text in files.items():
            target = args.root / rel
            current = target.read_text(encoding="utf-8") if target.exists() else None
            if current == text:
                continue
            if args.check:
                print(f"{rel} differs from {doc}", file=sys.stderr)
                status = 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                print(f"wrote {rel}")
    return status


if __name__ == "__main__":
    sys.exit(main())
