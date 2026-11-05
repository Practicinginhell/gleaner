"""Tests for RSS and Atom parsing details: link choice, RSS 1.0 and date timezones."""

from datetime import UTC

from gleaner.sources.rss import RSSSource

ATOM_WITH_REPLIES = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>b</title>
<entry><title>Post</title><id>1</id><updated>2026-11-01T00:00:00Z</updated>
<link rel="replies" href="https://blog.example/post/comments"/>
<link rel="alternate" href="https://blog.example/post"/><summary>Text.</summary></entry></feed>"""

RSS1 = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns="http://purl.org/rss/1.0/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
<channel rdf:about="https://lab.example/"><title>Lab</title><link>https://lab.example/</link></channel>
<item rdf:about="https://lab.example/p1"><title>Paper one</title><link>https://lab.example/p1</link>
<description>About agents.</description><dc:date>2026-10-30</dc:date></item></rdf:RDF>"""


def test_atom_entry_links_to_the_article_not_its_comments():
    items = RSSSource()._parse_feed_xml(ATOM_WITH_REPLIES, "b")
    assert items[0].url == "https://blog.example/post"


def test_rss_1_0_feeds_have_items_with_utc_dates():
    items = RSSSource()._parse_feed_xml(RSS1, "lab")
    assert [(i.title, i.url) for i in items] == [("Paper one", "https://lab.example/p1")]
    assert items[0].published_at.tzinfo is not None
    assert items[0].published_at.utcoffset() == UTC.utcoffset(None)
