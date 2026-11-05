"""URL canonicalization and HTML sanitization at the ingestion trust boundary."""


import pytest

from gleaner.normalize import canonicalize_url, strip_html


@pytest.mark.parametrize(
    "url",
    [
        "javascript://[::1",  # urlparse raises ValueError: must not fail open
        "https:///etc/passwd",  # empty host
        "http://",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "//example.com/no-scheme",
        "https://example.com:99999/bad-port",
    ],
)
def test_canonicalize_url_rejects_unsafe_or_hostless(url):
    assert canonicalize_url(url) is None


def test_canonicalize_url_root_trailing_slash_is_consistent():
    assert canonicalize_url("https://a.com") == canonicalize_url("https://a.com/")
    assert canonicalize_url("https://A.com/?utm_source=x") == canonicalize_url("https://a.com")


def test_strip_html_drops_hidden_subtree_with_nested_tags():
    cleaned = strip_html('<div style="display:none"><span>x</span>IGNORE</div><p>visible</p>')
    assert "IGNORE" not in cleaned
    assert "x" not in cleaned.split()
    assert "visible" in cleaned


@pytest.mark.parametrize(
    "markup",
    [
        '<div style="visibility: hidden"><b>a</b> SECRET</div>',
        "<div hidden><i>a</i> SECRET</div>",
        '<section aria-hidden="true"><p>a</p> SECRET</section>',
        '<div STYLE="Display : None"><span>a</span> SECRET</div>',
        "<script>var s = '<b>x</b>'; SECRET</script>",
        "<style>.a{} /* SECRET */</style>",
        "<!-- SECRET -->",
    ],
)
def test_strip_html_drops_all_hidden_kinds(markup):
    cleaned = strip_html(f"<p>before</p>{markup}<p>after</p>")
    assert "SECRET" not in cleaned
    assert "before" in cleaned
    assert "after" in cleaned


def test_strip_html_keeps_text_after_hidden_void_and_self_closed_elements():
    cleaned = strip_html('<p>one<br>two</p><img hidden src="x"><input type="hidden" hidden/>three')
    assert cleaned.split() == ["one", "two", "three"]


def test_strip_html_unescapes_entities_and_separates_blocks():
    assert strip_html("<p>Tom &amp; Jerry</p><p>&lt;b&gt;</p>") == "Tom & Jerry\n<b>"
