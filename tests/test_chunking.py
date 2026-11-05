"""Tests for semantic chunking and structure-preserving text segmentation."""

from graphflow.text import chunk_text

from gleaner.chunking import extract_prose_sentences
from gleaner.db import Database
from gleaner.models import Item, Question
from gleaner.questions import extract_claims_for_item


def test_chunk_text_preserves_code_blocks_intact():
    text = (
        "Here is how to configure LangGraph checkpoints in production:\n\n"
        "```python\n"
        "from langgraph.checkpoint.sqlite import SqliteSaver\n"
        "memory = SqliteSaver.from_conn_string(':memory:')\n"
        "```\n\n"
        "Checkpoints grow with every superstep unless delta compression is enabled."
    )
    chunks = chunk_text(text)

    # Code block must be captured as one intact chunk, not split on newlines or periods
    code_chunks = [c for c in chunks if "```" in c]
    assert len(code_chunks) == 1
    assert "from langgraph.checkpoint.sqlite import SqliteSaver" in code_chunks[0]
    assert "memory = SqliteSaver.from_conn_string(':memory:')" in code_chunks[0]


def test_chunk_text_preserves_markdown_tables():
    text = (
        "Benchmark results across reasoning architectures:\n\n"
        "| Architecture | Pass@1 | Latency |\n"
        "| :--- | :--- | :--- |\n"
        "| MCTS + 3B | 84.2% | 1.2s |\n"
        "| Beam Search | 76.5% | 0.8s |\n\n"
        "MCTS consistently outperforms beam search on multi-step reasoning."
    )
    chunks = chunk_text(text)

    table_chunks = [c for c in chunks if "|" in c]
    assert len(table_chunks) == 1
    assert "MCTS + 3B" in table_chunks[0]
    assert "Beam Search" in table_chunks[0]


def test_chunk_text_preserves_latex_math():
    text = "The scaling law is given by $$L(N, D) = \\left(\\frac{N_c}{N}\\right)^{\\alpha_N} + \\left(\\frac{D_c}{D}\\right)^{\\alpha_D}$$ as shown."
    chunks = chunk_text(text)

    math_chunks = [c for c in chunks if "$$" in c]
    assert len(math_chunks) == 1
    assert "\\alpha_N" in math_chunks[0]


def test_extract_prose_sentences_isolates_quotable_prose():
    document = (
        "# Engineering Incident Report\n\n"
        "```python\n"
        "def broken_memory_loop():\n"
        "    while True:\n"
        "        yield checkpoint.load()\n"
        "```\n\n"
        "Checkpoints grew without bound in production during high-volume agent runs.\n\n"
        "| Run | Memory | Status |\n"
        "| 1 | 8GB | OOM |\n\n"
        "Adding delta compression resolved memory growth across all clusters."
    )
    sentences = extract_prose_sentences(document)

    # Sentences should only contain the real prose sentences
    assert "Checkpoints grew without bound in production during high-volume agent runs." in sentences
    assert "Adding delta compression resolved memory growth across all clusters." in sentences

    # Neither code block lines nor table cells should appear as standalone sentences
    assert not any("broken_memory_loop" in s for s in sentences)
    assert not any("yield checkpoint" in s for s in sentences)
    assert not any("OOM" in s for s in sentences)


def test_extract_claims_for_item_ignores_code_blocks_and_extracts_prose_claim():
    db = Database(":memory:")
    q = Question(id="q-lg", text="What are the common failure modes of LangGraph in production?")

    text = (
        "Title: LangGraph Postmortem\n\n"
        "```python\n"
        "# LangGraph checkpoint failure simulation in production environments\n"
        "run_simulation(steps=100)\n"
        "```\n\n"
        "LangGraph checkpoints grew without bound in production until delta compaction was deployed."
    )
    item = Item(id="item-test-chunk", source_id="src-1", title="Postmortem", text=text, url="https://example.com/p", status="scored")

    claims = extract_claims_for_item(db, item, [q], model=None)
    assert len(claims) == 1
    # The claim quote must be the real prose sentence, NOT the comment inside the code block!
    assert claims[0].quote == "LangGraph checkpoints grew without bound in production until delta compaction was deployed."


def test_a_code_block_longer_than_the_chunkers_limit_is_still_not_prose():
    # graphflow's fence rule spans at most 1000 chars; past that, code lines would read as sentences
    code = "".join(f"// step {n}: LangGraph checkpoints grew without bound in production run {n}\n" for n in range(20))
    text = f"Intro sentence about the incident in our production cluster.\n\n```js\n{code}```\n\nThe fix shipped in version 0.4 and memory stayed flat afterwards."
    assert len(code) > 1000

    assert extract_prose_sentences(text) == [
        "Intro sentence about the incident in our production cluster.",
        "The fix shipped in version 0.4 and memory stayed flat afterwards.",
    ]


def test_a_long_sentence_is_quoted_whole_not_in_fragments():
    long = "The checkpoint store grew " + "with every superstep of the graph state " * 8 + "until delta compression shipped."
    text = f"Short intro sentence for this post here. {long}"
    assert len(long) > 300  # the chunker cuts runs this long into 200-char pieces

    sentences = extract_prose_sentences(text)
    assert long in sentences and all(s in text for s in sentences)
    assert all(s[0].isupper() for s in sentences)  # no piece starting mid-word
