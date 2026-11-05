"""Dashboard XSS: server-rendered values are escaped, URLs scheme-checked, and the inline JS escapes too."""

import json
import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

import gleaner.server
from gleaner.db import Database
from gleaner.digest import Digest
from gleaner.models import Item, SourceRow
from gleaner.server import create_app

PAYLOAD = "<img src=x onerror=alert(1)>"
EVIL_ID = 'x"><img src=x onerror=alert(1)>'


def _seed(db: Database, item: Item) -> None:
    db.upsert_source(SourceRow(id="src-x", type="rss", params={"url": "http://feed.com"}))
    db.upsert_item(item)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO scores (item_id, profile_version, components, final, computed_at) VALUES (?, 1, '{}', 0.9, ?);",
            (item.id, "2026-10-27T00:00:00+00:00"),
        )


def _page(db: Database) -> str:
    res = TestClient(create_app(db)).get("/")
    assert res.status_code == 200
    return res.text


def test_index_escapes_item_id_title_and_rejects_javascript_urls():
    db = Database(":memory:")
    _seed(db, Item(id=EVIL_ID, source_id="src-x", title=PAYLOAD, text="t", url="javascript:alert(1)", status="scored"))

    page = _page(db)

    assert "<img src=x" not in page
    assert 'data-id="x">' not in page
    assert 'data-id="x&quot;&gt;&lt;img src=x onerror=alert(1)&gt;"' in page
    assert "javascript:alert" not in page
    assert 'href="#"' in page


def test_index_keeps_http_urls():
    db = Database(":memory:")
    _seed(db, Item(id="item-ok", source_id="src-x", title="Ok", text="t", url="https://example.com/a?b=1&c=2", status="scored"))

    assert 'href="https://example.com/a?b=1&amp;c=2"' in _page(db)


def test_index_escapes_opportunity_id_and_digest_date(monkeypatch):
    db = Database(":memory:")
    _seed(db, Item(id="item-opp", source_id="src-x", title="Grant", text="t", status="scored"))
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO opportunities (id, item_id, type, deadline, status) VALUES (?, 'item-opp', 'grant', '2026-12-01', 'open');",
            ('o"><script>alert(1)</script>',),
        )
    monkeypatch.setattr(gleaner.server, "generate_digest", lambda *a, **k: Digest(date="<script>alert(2)</script>"))

    page = _page(db)

    assert "<script>alert" not in page
    assert "/api/opportunities/o%22%3E%3Cscript%3Ealert%281%29%3C%2Fscript%3E/ics" in page


def test_index_shows_no_data_yet_without_telemetry():
    db = Database(":memory:")
    _seed(db, Item(id="item-a", source_id="src-x", title="A", text="t", status="scored"))

    page = _page(db)

    assert "no data yet" in page.lower()
    assert "0.0%" not in page
    telem = TestClient(create_app(db)).get("/api/telemetry").json()
    # Only stages that were really measured get numbers; the rest are unknown, not guessed
    assert telem["deduped"] is None
    assert telem["passed_vectors"] is None
    assert telem["compute_saved_pct"] is None


def _inline_script(page: str) -> str:
    scripts = re.findall(r"<script>(.*?)</script>", page, flags=re.DOTALL)
    assert scripts
    return scripts[-1]


def test_inline_js_escapes_interpolated_fields():
    js = _inline_script(_page(Database(":memory:")))

    assert "function esc(" in js
    for field in ("c.item.title", "c.why", "c.summary", "c.item.id", "data.answer"):
        assert f"' + {field} + '" not in js, field
        assert f"esc({field})" in js, field
    assert "+ v +" not in js
    assert "'\" + c + \"'" not in js


NODE_HARNESS = """
const results = {};
const elements = {};
function el(id) {
  if (!elements[id]) elements[id] = {id, value: 'q', style: {}, innerHTML: '', innerText: '', focus() {}, classList: {add() {}, remove() {}}};
  return elements[id];
}
globalThis.document = {
  addEventListener() {}, querySelectorAll() { return []; }, querySelector() { return null; },
  getElementById: el, documentElement: {classList: {add() {}, toggle() {}}},
};
globalThis.window = {matchMedia() { return {matches: false}; }, location: {}};
globalThis.localStorage = {getItem() { return null; }, setItem() {}};
const CARD = __CARD__;
const ANSWER = __ANSWER__;
globalThis.fetch = async () => ({ok: true, json: async () => ANSWER});
__SCRIPT__
(async () => {
  results.card = renderCardHtml(CARD);
  await askQuestion({preventDefault() {}});
  results.answer = el('ask-result').innerHTML;
  console.log(JSON.stringify(results));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_inline_js_renders_hostile_values_as_text(tmp_path):
    js = _inline_script(_page(Database(":memory:")))
    card = {
        "item": {"id": EVIL_ID, "title": PAYLOAD, "url": "javascript:alert(1)", "source_id": PAYLOAD, "kind": PAYLOAD},
        "score": 0.5,
        "why": PAYLOAD,
        "summary": PAYLOAD,
        "perspectives": {PAYLOAD: PAYLOAD},
        "components": {"relevance": 0, "novelty": 0, "quality": 0, "momentum": 0, "actionability": 0},
    }
    answer = {"answer": f"{PAYLOAD} [{EVIL_ID}] **{PAYLOAD}** [x](javascript:alert(1)) [ok](https://example.com/a?b=1&c=2)",
              "citations": [EVIL_ID],
              "sources": [{"id": EVIL_ID, "title": PAYLOAD, "url": "javascript:alert(1)", "source": PAYLOAD}]}
    harness = (
        NODE_HARNESS.replace("__CARD__", json.dumps(card))
        .replace("__ANSWER__", json.dumps(answer))
        .replace("__SCRIPT__", js)
    )
    script = tmp_path / "harness.js"
    script.write_text(harness, encoding="utf-8")

    out = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=30, check=True)
    rendered = json.loads(out.stdout.strip().splitlines()[-1])

    for html_out in (rendered["card"], rendered["answer"]):
        assert "<img" not in html_out
        assert '="x">' not in html_out
        assert "&lt;img src=x onerror=alert(1)&gt;" in html_out
    for html_out in (rendered["card"], rendered["answer"]):
        assert 'href="javascript' not in html_out
        assert 'href="#"' in html_out
    assert "javascript:" not in rendered["card"]
    assert 'data-id="x&quot;&gt;&lt;img src=x onerror=alert(1)&gt;"' in rendered["card"]
    # the answer's [item id] becomes the source's number; the hostile id never reaches the page raw
    assert "[1]" in rendered["answer"] and "x&quot;&gt;&lt;img" not in rendered["answer"].split("<ol")[0]
    # the answer's markdown bold and http(s) links render; a javascript: link stays text
    assert "<strong>&lt;img src=x onerror=alert(1)&gt;</strong>" in rendered["answer"]
    assert '<a href="https://example.com/a?b=1&amp;c=2" target="_blank" rel="noopener noreferrer">ok</a>' in rendered["answer"]
    assert "[x](javascript:alert(1))" in rendered["answer"]
