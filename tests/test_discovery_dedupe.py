"""dedupe_items edge cases: empty text, URL-less items, unsafe URLs."""



from gleaner.dedupe import dedupe_items
from gleaner.models import Item


def test_items_with_empty_text_are_not_near_duplicates():
    items = [
        Item(id="a", source_id="s", url="https://example.com/a", title="Release v1.0", text=""),
        Item(id="b", source_id="s", url="https://example.com/b", title="Totally different post", text=""),
    ]
    deduped = dedupe_items(items)
    assert len(deduped) == 2


def test_urlless_items_keep_distinct_ids():
    items = [
        Item(id="native-1", source_id="s", url=None, title="First", text="alpha beta gamma delta epsilon"),
        Item(id="native-2", source_id="s", url=None, title="Second", text="one two three four five six"),
    ]
    deduped = dedupe_items(items)
    assert len(deduped) == 2
    assert deduped[0].id != deduped[1].id


def test_unsafe_url_is_cleared_not_kept():
    items = [
        Item(id="x", source_id="s", url="javascript:alert(1)", title="Hostile", text="some text body here"),
        Item(id="y", source_id="s", url="https:///etc/passwd", title="Hostless", text="another different body"),
    ]
    deduped = dedupe_items(items)
    assert len(deduped) == 2
    assert all(it.url is None for it in deduped)
    assert deduped[0].id != deduped[1].id
