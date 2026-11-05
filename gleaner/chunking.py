"""Quotable prose sentences from structured text, for standing-question evidence.

Segmentation is graphflow's (``graphflow.text``, the author's pre-existing port of Jina AI's
Semantic Chunking Regex v1); this module only decides which text is quotable prose.
"""

from __future__ import annotations

import re

from graphflow.text.chunking import CHUNK_RE


def _blank_fenced_code(text: str) -> str:
    """Blank out fenced code blocks, whatever their length.

    Args:
        text: The document text.

    Returns:
        ``text`` with every character of a fenced code block (fence lines included) turned to a
        space, newlines kept, so offsets still line up with the original.
    """
    out: list[str] = []
    marker: str | None = None  # the open fence's ``` or ~~~
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip(" ")
        fence = stripped[:3] if len(line) - len(stripped) <= 3 and stripped[:3] in ("```", "~~~") else None
        if marker is None and fence is None:
            out.append(line)
            continue
        out.append(re.sub(r"[^\r\n]", " ", line))
        if marker is None:
            marker = fence
        elif fence == marker:
            marker = None
    return "".join(out)


def _is_structural_block(chunk: str) -> bool:
    """Return True if a chunk is blank, code, a table, math, a heading or a rule."""
    s = chunk.strip()
    if not s:
        return True
    # Fenced code blocks or pre tags
    if s.startswith(("```", "~~~", "<pre")):
        return True
    # Tables
    if (s.startswith("|") and "|" in s[1:]) or s.startswith("<table"):
        return True
    # LaTeX display math
    if s.startswith("$$"):
        return True
    # Headings and rules
    if s.startswith(("#", "<h1", "<h2", "<h3", "<h4", "<h5", "<h6", "---", "***", "<hr")):
        return True
    return False


def _is_prose_sentence(sentence: str) -> bool:
    """Return True for quotable prose: six words or more, no markdown or code fences."""
    words = sentence.split()
    if len(words) < 6:
        return False
    # Avoid markdown links, code blocks, raw URLs, or table delimiters
    return not bool(re.search(r"\]\(|https?://|```|`|^[#*>|\[-]", sentence))


def extract_prose_sentences(text: str) -> list[str]:
    """Extract clean, quotable prose sentences from a document.

    Uses semantic chunking first to filter out code blocks, tables, math, and
    headings, preventing non-prose lines from being butchered into fake claims.

    Args:
        text: The document text.

    Returns:
        The prose sentences in document order; each is an exact substring of ``text``.
        Empty when ``text`` is empty.
    """
    if not text:
        return []

    blanked = _blank_fenced_code(text)
    runs: list[str] = []
    run_start: int | None = None
    run_end = 0
    for m in CHUNK_RE.finditer(blanked):
        chunk = m.group(0)
        start = m.start() + len(chunk) - len(chunk.lstrip())
        end = m.start() + len(chunk.rstrip())
        structural = _is_structural_block(chunk)
        # a sentence chunk's trailing \s+ can swallow a blanked code block: never join across one
        if run_start is not None and (structural or text[run_end:start] != blanked[run_end:start]):
            runs.append(text[run_start:run_end])
            run_start = None
        if structural:
            continue
        if run_start is None:
            run_start = start
        run_end = end
    if run_start is not None:
        runs.append(text[run_start:run_end])

    sentences: list[str] = []
    for run in runs:
        sub_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", run)]
        sentences.extend(s for s in sub_sentences if s and _is_prose_sentence(s))
    return sentences
