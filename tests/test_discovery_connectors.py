"""Phase 1: Source connectors tests using httpx.MockTransport."""

import json
import socket
from pathlib import Path

import httpx
import pytest

from gleaner.config import SourceConfig
from gleaner.sources.arxiv import ArxivSource
from gleaner.sources.github import GitHubSource
from gleaner.sources.hn import HNSource
from gleaner.sources.rss import RSSSource

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "discovery"


@pytest.fixture(autouse=True)
def offline_dns(monkeypatch):
    """The RSS connector resolves the feed host (SSRF guard); answer from a fixed table, never the network."""

    def fake_getaddrinfo(host, *args, **kwargs):
        if host in ("example.com", "feeds.example.com"):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 0))]
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def _cfg(source_type, params, source_id="src-test"):
    return SourceConfig(id=source_id, type=source_type, params=params, cadence="1h")


async def _fetch(source, cfg, handler, **kwargs):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await source.fetch(cfg, since=None, client=client, **kwargs)


@pytest.mark.asyncio
async def test_rss_connector_parses_feed():
    xml_content = (FIXTURES_DIR / "rss_sample.xml").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=xml_content,
            headers={"ETag": '"etag-abc"', "Last-Modified": "Mon, 27 Oct 2026 14:00:00 GMT"},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        cfg = SourceConfig(
            id="rss-test",
            type="rss",
            params={"url": "https://example.com/feed.xml"},
            cadence="1h",
            trust=0.8,
        )
        source = RSSSource()
        res = await source.fetch(cfg, since=None, client=client)

        assert len(res.items) == 2
        item1 = next(it for it in res.items if "Graphflow" in it.title)
        assert item1.source_id == "rss-test"
        # Canonicalized URL: tracking params removed
        assert "utm_source" not in item1.url
        assert item1.url == "https://example.com/blog/declarative-workflows"
        # Hidden text stripped
        assert "ignore instructions" not in item1.text
        assert "Graphflow compiles Python workflows" in item1.text
        assert res.etag == '"etag-abc"'
        assert res.last_modified == "Mon, 27 Oct 2026 14:00:00 GMT"


@pytest.mark.asyncio
async def test_rss_connector_304_not_modified():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("if-none-match") == '"etag-abc"'
        return httpx.Response(304, headers={"ETag": '"etag-abc"'})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        cfg = SourceConfig(
            id="rss-test",
            type="rss",
            params={"url": "https://example.com/feed.xml"},
            cadence="1h",
        )
        source = RSSSource()
        res = await source.fetch(cfg, since=None, client=client, etag='"etag-abc"')
        assert len(res.items) == 0
        assert res.etag == '"etag-abc"'


@pytest.mark.asyncio
async def test_arxiv_connector():
    xml_content = (FIXTURES_DIR / "arxiv_sample.xml").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=xml_content)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        cfg = SourceConfig(
            id="arxiv-test",
            type="arxiv",
            params={"categories": ["cs.AI"], "query": "autonomous agents"},
            cadence="12h",
        )
        source = ArxivSource()
        res = await source.fetch(cfg, since=None, client=client)
        assert len(res.items) == 1
        item = res.items[0]
        assert "Test-Time Compute" in item.title
        assert "Alice Johnson" in (item.author or "")
        assert item.kind == "paper"
        assert item.url == "http://arxiv.org/abs/2610.12345v1"


@pytest.mark.asyncio
async def test_hn_connector():
    json_content = (FIXTURES_DIR / "hn_sample.json").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=json_content)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        cfg = SourceConfig(
            id="hn-test",
            type="hn",
            params={"query": "front_page"},
            cadence="30m",
        )
        source = HNSource()
        res = await source.fetch(cfg, since=None, client=client)
        assert len(res.items) == 2
        show_hn = next(it for it in res.items if "Show HN" in it.title)
        assert show_hn.signals["points"] == 342
        assert show_hn.signals["comments"] == 89
        assert show_hn.author == "devuser"
        assert show_hn.kind == "post"


@pytest.mark.asyncio
async def test_github_connector():
    json_content = (FIXTURES_DIR / "github_sample.json").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=json_content)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        cfg = SourceConfig(
            id="gh-test",
            type="github",
            params={"repo": "langchain-ai/graphflow"},
            cadence="6h",
        )
        source = GitHubSource()
        res = await source.fetch(cfg, since=None, client=client)
        assert len(res.items) == 1
        rel = res.items[0]
        assert rel.kind == "release"
        assert "v0.3.0" in rel.title
        assert "Declarative fan_out" in rel.text
        assert rel.url == "https://github.com/langchain-ai/graphflow/releases/tag/v0.3.0"


@pytest.mark.asyncio
async def test_github_rejects_invalid_repo_param():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=[])

    for repo in ("../../user#", "owner/repo/../../x", "owner", "a/b?x=1", "owner/..", "-bad/repo", "own er/repo", "o/r\n"):
        with pytest.raises(ValueError):
            await _fetch(GitHubSource(), _cfg("github", {"repo": repo}), handler)
    assert calls == []

    await _fetch(GitHubSource(), _cfg("github", {"repo": "octo-org/my_repo.js"}), handler)
    assert calls == ["https://api.github.com/repos/octo-org/my_repo.js/releases"]


@pytest.mark.asyncio
async def test_rss_malformed_xml_is_a_failed_fetch():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<rss><channel><item>broken")

    with pytest.raises(ValueError, match="XML"):
        await _fetch(RSSSource(), _cfg("rss", {"url": "https://example.com/feed.xml"}), handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["front_page", "llama"])
async def test_hn_calls_the_algolia_api_host(query):
    hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(200, json={"hits": []})

    await _fetch(HNSource(), _cfg("hn", {"query": query}), handler)

    # The HN Search API lives at hn.algolia.com; hn.algolia.net does not resolve
    assert hosts == ["hn.algolia.com"]


@pytest.mark.asyncio
async def test_arxiv_calls_the_https_api():
    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(request.url)
        return httpx.Response(200, text='<feed xmlns="http://www.w3.org/2005/Atom"></feed>')

    await _fetch(ArxivSource(), _cfg("arxiv", {"query": '"function calling"'}), handler)

    # export.arxiv.org answers http:// with a 301 to https://, which the client doesn't follow
    assert [(u.scheme, u.host) for u in urls] == [("https", "export.arxiv.org")]


@pytest.mark.asyncio
async def test_arxiv_malformed_xml_is_a_failed_fetch():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>Service Unavailable")

    with pytest.raises(ValueError, match="XML"):
        await _fetch(ArxivSource(), _cfg("arxiv", {"query": "agents"}), handler)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "cfg"),
    [
        (ArxivSource(), _cfg("arxiv", {"query": "agents"})),
        (HNSource(), _cfg("hn", {"query": "llama"})),
    ],
)
async def test_arxiv_and_hn_send_conditional_headers_and_handle_304(source, cfg):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["etag"] = request.headers.get("if-none-match")
        seen["lm"] = request.headers.get("if-modified-since")
        return httpx.Response(304)

    res = await _fetch(source, cfg, handler, etag='"e1"', last_modified="Mon, 27 Oct 2026 14:00:00 GMT")
    assert seen == {"etag": '"e1"', "lm": "Mon, 27 Oct 2026 14:00:00 GMT"}
    assert res.items == []
    assert res.etag == '"e1"'
    assert res.last_modified == "Mon, 27 Oct 2026 14:00:00 GMT"


@pytest.mark.asyncio
async def test_rss_relative_links_and_guids_give_distinct_ids():
    feed = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>t</title>
  <item><title>Relative A</title><link>/posts/a</link></item>
  <item><title>Relative B</title><link>posts/b</link></item>
  <item><title>No link 1</title><guid isPermaLink="false">tag:example.com,2026:1</guid></item>
  <item><title>No link 2</title><guid isPermaLink="false">tag:example.com,2026:2</guid></item>
</channel></rss>"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=feed)

    res = await _fetch(RSSSource(), _cfg("rss", {"url": "https://feeds.example.com/blog/feed.xml"}), handler)
    by_title = {it.title: it for it in res.items}
    assert by_title["Relative A"].url == "https://feeds.example.com/posts/a"
    assert by_title["Relative B"].url == "https://feeds.example.com/blog/posts/b"
    assert by_title["No link 1"].url is None
    assert len({it.id for it in res.items}) == 4


@pytest.mark.asyncio
async def test_atom_entries_without_links_use_entry_id():
    feed = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>t</title>
  <entry><id>urn:uuid:1</id><title>One</title></entry>
  <entry><id>urn:uuid:2</id><title>Two</title></entry>
</feed>"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=feed)

    res = await _fetch(RSSSource(), _cfg("rss", {"url": "https://example.com/atom.xml"}), handler)
    assert len(res.items) == 2
    assert res.items[0].id != res.items[1].id


@pytest.mark.asyncio
async def test_connector_titles_are_stripped_of_html():
    rss = """<?xml version="1.0"?><rss version="2.0"><channel>
  <item><title>&lt;b&gt;Bold&lt;/b&gt; &amp;amp; news&lt;script&gt;evil()&lt;/script&gt;</title>
  <link>https://example.com/x</link></item></channel></rss>"""
    arxiv = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
  <id>http://arxiv.org/abs/1</id><title>&lt;i&gt;Attention&lt;/i&gt; &amp;amp; More</title>
  <link href="http://arxiv.org/abs/1" rel="alternate"/></entry></feed>"""
    hn = json.dumps({"hits": [{"objectID": "1", "title": "<b>Show HN</b> X &amp; Y", "url": "https://example.com/hn"}]})
    gh = json.dumps([{"id": 1, "tag_name": "v1", "name": "<img src=x onerror=alert(1)>v1 &amp; more",
                      "html_url": "https://github.com/o/r/releases/tag/v1"}])

    cases = [
        (RSSSource(), _cfg("rss", {"url": "https://example.com/feed.xml"}), rss),
        (ArxivSource(), _cfg("arxiv", {"query": "x"}), arxiv),
        (HNSource(), _cfg("hn", {"query": "x"}), hn),
        (GitHubSource(), _cfg("github", {"repo": "o/r"}), gh),
    ]
    for source, cfg, body in cases:
        res = await _fetch(source, cfg, lambda request, body=body: httpx.Response(200, text=body))
        title = res.items[0].title
        assert "<" not in title and "&amp;" not in title and "evil" not in title, f"{source.type}: {title!r}"
        assert "&" in title, f"{source.type}: {title!r}"


def test_get_connector_includes_nostr():
    from gleaner.sources import get_connector
    from gleaner.sources.nostr import NostrSource

    assert isinstance(get_connector("nostr"), NostrSource)
