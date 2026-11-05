"""Dashboard page: the Brief leads with evidence for the user's questions, status is honest,
categories are right, and judged answers show their diff and quotes."""

import json
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from gleaner.dashboard import QuestionView, render_dashboard
from gleaner.db import Database
from gleaner.digest import Digest
from gleaner.feed import FeedCard
from gleaner.models import Answer, Claim, Item, Question, Sentence, SourceRow
from gleaner.server import create_app

TODAY = datetime.now(UTC).astimezone().date()
LG_QUESTION = Question(id="q-lg", text="What are the common failure modes of LangGraph in production?")


def _page(**overrides) -> str:
    args = {
        "cards": [],
        "digest": Digest(date=TODAY.isoformat()),
        "trends": [],
        "sources": [],
        "opportunities": [],
        "quarantine": [],
        "questions": [],
        "telemetry": None,
        "last_collect": "2026-10-29T14:41:00",
        "judge_model": "llama3.2:3b",
        "today": TODAY,
        "hn_leads": [],
    } | overrides
    return render_dashboard(**args)


def _card(item_id: str, kind: str) -> FeedCard:
    item = Item(id=item_id, source_id="hn-frontpage", kind=kind, title=f"Show HN: {item_id}", text="t", url="https://news.ycombinator.com/item?id=1")
    return FeedCard(item=item, score=0.4, why="Matches local-llm", summary="")


def _keyword_claim(n: int, created_at: datetime) -> tuple[Claim, Item]:
    item = Item(id=f"i{n}", source_id="blog-simon-willison", title=f"Post {n}", text="t", url=f"https://simonwillison.net/{n}")
    quote = f"LangGraph checkpoints grew without bound in production, case {n}."
    claim = Claim(id=f"c{n}", question_id="q-lg", item_id=item.id, text=quote, quote=quote, stance="context",
                  origin_key=f"url:{item.url}", model="heuristic", created_at=created_at)
    return claim, item


def test_hacker_news_posts_count_as_discussions():
    page = _page(cards=[_card("a", "post"), _card("b", "article")])

    assert 'id="card-a" data-id="a" data-kind="post" data-category="discussion"' in page
    assert 'data-category="discussion" aria-pressed="false" onclick="filterByCategory(this.dataset.category, this)">Discussions <span>1</span>' in page


def test_brief_opens_on_quoted_evidence_when_llama_is_off():
    now = datetime.now(UTC)
    pairs = [_keyword_claim(n, now - timedelta(minutes=n)) for n in range(5)]
    view = QuestionView(question=LG_QUESTION, claims=[c for c, _ in pairs], items={i.id: i for _, i in pairs})

    page = _page(questions=[view, QuestionView(question=Question(id="q2", text="Other?"))], judge_model="heuristic",
                 telemetry={"fetched": 1305, "scored": 258, "judged_llama": 0})

    assert "New quotes for 1 of your 2 questions." in page
    assert "258 of the 1,305 items in the last collect matched your interests." in page
    assert "Llama is off, so gleaner quotes matching sources instead of writing answers." in page
    assert '<span class="pill pill-tint">5 new today</span>' in page
    # Newest first, quoted word for word, linked to the item it came from
    assert page.index("case 0.") < page.index("case 1.") < page.index("case 2.")
    assert '<a href="https://simonwillison.net/0" target="_blank" rel="noopener noreferrer">Post 0</a>' in page
    assert "<summary>Show 2 more</summary>" in page
    # Keyword matches are not presented as judged answers
    assert "found by word match, not checked by Llama yet" in page
    assert "confidence" not in page.split('id="question-q-lg"')[1].split("</article>")[0].lower()
    assert "<dt>Checked by Llama</dt><dd>Off</dd>" in page


def test_status_is_one_quiet_line_naming_failing_sources_and_the_judge():
    sources = [
        SourceRow(id="hn-frontpage", type="hn", params={}, failures=2),
        SourceRow(id="gh-langgraph", type="github", params={}),
    ]

    page = _page(sources=sources, judge_model="heuristic")

    assert "<span>Checked by heuristic</span>" in page
    assert "<span>1 source failing</span>" in page
    assert '<span class="tab-n warn">1 failing</span>' in page
    assert "1 of 2 websites working" in page
    assert 'class="notice"' not in page  # no warning box above the content


def test_changed_answer_marks_new_and_removed_sentences_and_quotes_its_evidence():
    claim = Claim(id="c-new", question_id="q-lg", item_id="i1", text="8B model completes short plans",
                  quote="completes short multi-step plans", origin_key="arxiv:2610.1", model="llama3.1:8b")
    old = Answer(question_id="q-lg", version=1, confidence="low", sentences=[Sentence(text="They fail past two steps.")])
    new = Answer(
        question_id="q-lg",
        version=2,
        confidence="medium",
        what_changed="A paper contradicts the two-step limit.",
        sentences=[Sentence(text="An 8B model completes short plans.", claim_ids=["c-new"])],
        created_at=datetime.now(UTC),
    )

    page = _page(questions=[QuestionView(question=LG_QUESTION, latest=new, previous=old, claims=[claim])])

    card = page.split('id="question-q-lg"')[1].split("</article>")[0]
    # the current answer first, in plain text; the old wording one tap away, never in the way
    assert '<p class="answer">An 8B model completes short plans.<a class="cite" href="#claim-c-new">1</a></p>' in card
    assert "<ins>" not in card
    assert '<details class="old-wording"><summary>Show the old wording</summary>' in card
    assert "<del>They fail past two steps.</del>" in card
    assert card.index('class="answer"') < card.index("What changed:") < card.index("old-wording")
    assert "“completes short multi-step plans”" in page
    assert "1 answer changed today." in page
    assert "Medium confidence" in page


def test_unanswered_questions_say_so_and_never_print_zero_percent():
    hit_rate = {"useful": 0, "wrong": 1, "already_knew": 0, "total": 1, "hit_rate_pct": 0.0}

    page = _page(questions=[QuestionView(question=LG_QUESTION, hit_rate=hit_rate)])

    assert "No answers yet." in page
    assert "No evidence yet" in page
    assert "No evidence yet" in page and "Your ratings" not in page.split("<script>")[0]  # nothing to rate without quotes
    assert "0.0%" not in page


def test_served_page_has_no_invented_activity_log():
    db = Database(":memory:")
    db.upsert_source(SourceRow(id="src-x", type="rss", params={"url": "http://feed.com"}))

    page = TestClient(create_app(db)).get("/").text

    # The old "pipeline console" printed events no code had measured
    assert "reasoning pipeline online" not in page
    assert "Centroids shifted" not in page


def test_layout_is_a_reading_column_with_side_context_and_a_one_column_feed():
    page = _page()

    assert ".brief-grid { display: grid; grid-template-columns: minmax(0, 1fr) 320px;" in page
    assert ".feed-grid { display: grid; grid-template-columns: 240px minmax(0, 820px);" in page
    assert ".brief-grid { grid-template-columns: minmax(0, 1fr); gap: 32px; }" in page  # stacks below 1080px


def test_guide_explains_scores_as_scoring_computes_them():
    page = _page()

    assert 'id="guide-modal"' in page
    assert "openGuideModal()" in page
    assert "<dt>Novelty</dt><dd>How different the item is from recent items.</dd>" in page
    assert 'id="quickstart-banner"' not in page  # the Brief explains itself; no onboarding box


def test_dashboard_follows_apple_human_interface_guidelines():
    """Apple HIG tokens: SF fonts, translucent materials, 44px touch targets, press feedback."""
    page = _page()
    assert "-apple-system" in page
    assert "BlinkMacSystemFont" in page
    assert "SF Pro Text" in page
    assert "SF Pro Display" in page
    assert "backdrop-filter: blur" in page
    assert "-webkit-backdrop-filter: blur" in page
    assert "min-height: 44px" in page
    assert "transform: scale(" in page


def test_status_shows_the_collect_time_in_local_time():
    just_now = datetime.now(UTC).replace(tzinfo=None).isoformat()  # stored as naive UTC
    local = datetime.now(UTC).astimezone()

    page = _page(last_collect=just_now, today=local.date())

    assert f"<span>Collected today at {local:%H:%M}</span>" in page


def test_llama_summary_leads_the_quotes_and_names_its_writer():
    now = datetime.now(UTC)
    pairs = [_keyword_claim(n, now - timedelta(minutes=n)) for n in range(2)]
    latest = Answer(question_id="q-lg", version=2, what_changed="Checkpoints keep growing in production LangGraph runs.",
                    what_changed_by="llama3.2:1b", sentences=[Sentence(text="x", claim_ids=["c0"])], created_at=now)
    view = QuestionView(question=LG_QUESTION, latest=latest, claims=[c for c, _ in pairs], items={i.id: i for _, i in pairs})

    page = _page(questions=[view], judge_model="heuristic")
    card = page.split('id="question-q-lg"')[1].split("</article>")[0]

    assert '<p class="takeaway">Checkpoints keep growing in production LangGraph runs.</p>' in card
    assert "Summary by llama3.2:1b from the quotes below" in card
    assert card.index("takeaway") < card.index("<blockquote>")
    assert "Llama is off" not in page
    assert "llama3.2:1b sums up what\u2019s new. Sources are still matched by keyword." in page


def test_stack_question_lists_advisories_without_claiming_a_judge():
    item = Item(id="advisory:GHSA-1", source_id="osv", kind="advisory", title="Advisory GHSA-1: Header smuggling (httpx 0.28.1 installed)",
                text="Header smuggling\n\nDetails.", url="https://osv.dev/vulnerability/GHSA-1")
    claim = Claim(id="c-adv", question_id="q-stack", item_id=item.id, text="Header smuggling", quote="Header smuggling",
                  stance="context", origin_key="url:https://osv.dev/vulnerability/GHSA-1", model="osv")
    view = QuestionView(question=Question(id="q-stack", text="What changed in my stack?"), claims=[claim], items={item.id: item})

    card = _page(questions=[view]).split('id="question-q-stack"')[1].split("</article>")[0]

    assert "1 advisory for the versions installed here · from osv.dev" in card
    assert "“Header smuggling”" in card
    assert "confidence" not in card.lower()


def test_brief_lists_items_your_sources_had_before_hacker_news():
    from gleaner.hn_lead import HNLead

    item = Item(id="p", source_id="blog-sebastian-raschka", title="Building Reasoning Models", text="t", url="https://magazine.sebastianraschka.com/p/x")
    posted = datetime(2026, 10, 26, 15, tzinfo=UTC)
    leads = [HNLead(item=item, posted_at=posted, published_at=posted - timedelta(days=2, hours=4), points=855, hn_id="9"),
             HNLead(item=item, posted_at=posted, published_at=posted - timedelta(hours=9), points=40, hn_id="10")]

    page = _page(hn_leads=leads)

    assert '<h3 id="hn-h">Ahead of Hacker News</h3>' in page
    assert "2 days before it was posted · 855 points" in page
    assert "9 hours before it was posted · 40 points" in page
    assert '<a href="https://news.ycombinator.com/item?id=9" target="_blank" rel="noopener noreferrer">HN thread</a>' in page


def test_no_hn_card_until_there_is_a_lead():
    assert "Ahead of Hacker News" not in _page()


def test_stack_question_with_no_advisories_says_the_installed_versions_are_clear():
    view = QuestionView(question=Question(id="q-stack", text="What changed in my stack?"))

    card = _page(questions=[view]).split('id="question-q-stack"')[1].split("</article>")[0]

    assert "No known advisories affect the versions installed here. gleaner checks OSV once a day." in card


def test_quotes_llama_confirmed_show_as_checked_not_as_a_written_answer():
    now = datetime.now(UTC)
    pairs = [_keyword_claim(n, now - timedelta(minutes=n)) for n in range(2)]
    claims = [c.model_copy(update={"model": "llama3.2:3b"}) for c, _ in pairs]
    view = QuestionView(question=LG_QUESTION, claims=claims, items={i.id: i for _, i in pairs})

    card = _page(questions=[view], judge_model="llama3.2:3b").split('id="question-q-lg"')[1].split("</article>")[0]

    assert "2 quotes from 1 website · checked by llama3.2:3b" in card
    assert "<blockquote>" in card
    assert "confidence" not in card.lower()

    mixed = QuestionView(question=LG_QUESTION, claims=[claims[0], pairs[1][0]], items={i.id: i for _, i in pairs})
    assert "1 checked by llama3.2:3b, the rest found by word match" in _page(questions=[mixed])


def test_dashboard_has_interactive_configuration_modals_and_buttons():
    page = _page()

    # Sources modal and trigger
    assert 'id="add-source-modal"' in page
    assert 'id="btn-add-source"' in page
    assert "openAddSourceModal()" in page
    assert "submitAddSource(event)" in page

    # Question modal and trigger
    assert 'id="add-question-modal"' in page
    assert 'id="btn-add-question"' in page
    assert "openAddQuestionModal()" in page
    assert "submitAddQuestion(event)" in page

    # Deadlines sample seeder
    assert 'id="btn-seed-deadlines"' in page
    assert "seedSampleOpportunities(this)" in page


def test_feed_card_renders_clickable_score_inspector_button():
    page = _page(cards=[_card("c-test", "article")])

    assert 'id="score-inspector-modal"' in page
    assert '<button type="button" class="score"' in page
    assert 'onclick="inspectScore(this)"' in page
    assert 'data-score="0.40"' in page
    assert 'data-rel=' in page
    assert 'data-nov=' in page
    assert 'data-qual=' in page
    assert 'data-mom=' in page
    assert 'data-act=' in page


def test_search_keyword_highlighting_and_real_collect_progress_present():
    page = _page()

    assert "highlightMatchesInCard" in page
    assert "highlightHtml" in page
    assert "mark.highlight" in page
    # collect progress is the server's real stages, never steps rotated on a timer
    assert "new EventSource('/api/collect/events')" in page
    assert "Screening prompt injections" not in page


def test_dashboard_numbers_and_labels_come_from_the_code():
    from gleaner.scoring import RANKING_MODES

    page = _page()
    # Score Inspector weights are scoring's real ones, not a hard-coded copy
    assert "const ALL_WEIGHTS = " + json.dumps(RANKING_MODES) in page
    assert "40% in Balanced" not in page
    # Add Source sends the field the API reads; the API requires an HN query
    assert "cadence_s: cadence * 60" in page and "cadence_minutes" not in page
    assert "leave empty for frontpage" not in page
    # the answer model is configurable: the page names the one that ran
    assert "Muse Spark" not in page and "data.synthesized_by" in page


def test_impact_card_counts_what_gleaner_did_from_the_database(tmp_path):
    db = Database(tmp_path / "impact.db")
    for sid in ("blog", "arxiv"):
        db.upsert_source(SourceRow(id=sid, type="rss", params={}))
    statuses = ["scored", "exploring", "dropped_stage_a", "dropped_stage_a", "dropped_stage_b"]
    for n, status in enumerate(statuses):
        db.upsert_item(Item(id=f"i{n}", source_id="blog" if n % 2 else "arxiv", title="t", text="t", status=status,
                            fetched_at=datetime(2026, 10, 28, 15, tzinfo=UTC) + timedelta(hours=n)))
    db.upsert_question(LG_QUESTION)
    for n, model in enumerate(("llama3.2:3b", "heuristic")):
        db.upsert_claim(Claim(id=f"c{n}", question_id=LG_QUESTION.id, item_id="i0", text="t", quote=f"quote {n}",
                              stance="context", origin_key="o", model=model))

    impact = db.impact_counts()
    assert impact == {"read": 5, "sources": 2, "since": "2026-10-28", "kept": 2, "filtered": 3,
                      "quotes": 2, "checked": 1, "checked_by": ["llama3.2:3b"]}

    page = _page(impact=impact)
    assert "What gleaner did for you" in page and "Since Oct 28" in page
    assert '<span class="tile-num">5</span><span class="tile-cap">items read</span>' in page
    assert "kept for you (40%)" in page
    assert '<span class="tile-num">60%</span><span class="tile-cap">skipped as off-topic before any AI read it' in page
    assert "2 quotes on your questions · 1 checked by llama3.2:3b" in page
    assert "Last collect" in page and "Today\u2019s collect" not in page
    db.close()


def test_question_cards_say_each_thing_once_in_plain_words():
    one = QuestionView(question=LG_QUESTION, latest=None, previous=None, claims=[
        Claim(id="c1", question_id=LG_QUESTION.id, item_id="i1", text="t", quote="Checkpoints grew without bound.",
              stance="context", origin_key="o", model="heuristic")], items={}, hit_rate={})
    empty = QuestionView(question=Question(id="q-empty", text="Is anyone shipping CRDTs in production?"),
                         latest=None, previous=None, claims=[], items={}, hit_rate={})
    page = _page(questions=[one, empty])

    assert "1 quote from 1 website" in page and "quotes</span>" not in page  # no count badge beside the count
    assert page.count('class="q-foot"') == 1  # nothing to rate on an empty card
    assert "Write answer" in page and "⚡" not in page
    assert "Your questions" in page and "Add question" in page
    assert "Meta Llama hackathon" not in page


def test_confidence_shows_which_way_it_moved_since_the_last_answer():
    claim = Claim(id="c1", question_id="q-lg", item_id="i1", text="t", stance="supports", quote="q", origin_key="o")

    def card(before: str, now: str) -> str:
        old = Answer(question_id="q-lg", version=1, confidence=before, sentences=[Sentence(text="Old.")])
        new = Answer(question_id="q-lg", version=2, confidence=now, synthesized_by="muse-spark-1.3",
                     sentences=[Sentence(text="New.", claim_ids=["c1"])])
        return _page(questions=[QuestionView(question=LG_QUESTION, latest=new, previous=old, claims=[claim])])

    assert "High confidence" in card("insufficient", "high") and "↑ from unconfirmed" in card("insufficient", "high")
    assert "↓ from high" in card("high", "low")
    assert "↑" not in card("medium", "medium") and "↓" not in card("medium", "medium")
    assert "Unconfirmed" in card("low", "insufficient")  # quotes exist, so never "No evidence yet"


def test_adding_a_question_says_how_much_evidence_it_found():
    page = _page()

    assert "data.evidence" in page and "in what you've collected" in page
    assert "Question added to Brief." not in page


# A first-time reader's walkthrough (2026-11-03): each test below is one place they got lost.


def test_ratings_list_each_kind_instead_of_a_percent_that_disagrees_with_the_count():
    # "Useful 3 of 4 (100%)": 'Knew it' was in the 4 but not in the percent
    hit_rate = {"useful": 3, "wrong": 0, "already_knew": 1, "total": 4, "hit_rate_pct": 100.0}
    claim, item = _keyword_claim(1, datetime.now(UTC))

    page = _page(questions=[QuestionView(question=LG_QUESTION, claims=[claim], items={item.id: item}, hit_rate=hit_rate)])

    assert "Your ratings: 3 useful · 1 knew it" in page
    assert "(100%)" not in page


def test_deadlines_show_a_date_a_person_can_read():
    opp = {"id": "o1", "type": "event", "title": "Open-source agent showcase", "deadline": "2026-11-09T00:53:38.644743+00:00",
           "amount": None, "eligibility": None, "link": None}

    page = _page(opportunities=[opp])

    assert "Due 9 Nov 2026" in page
    assert "T00:53" not in page
    assert "Event ·" in page and "EVENT" not in page


def test_a_new_question_is_shown_where_the_reader_can_see_it_and_can_be_removed():
    page = _page(questions=[QuestionView(question=LG_QUESTION)])

    # after adding, the page opens at the new question and highlights it
    assert "location.assign('/?added=' + encodeURIComponent(data.id) + '#question-' + data.id)" in page
    assert ".q-card:target" in page
    # a question added by mistake can be taken off the Brief
    assert 'onclick="removeQuestion(this.dataset.qid, this)"' in page
    assert "method: 'DELETE'" in page
    # an empty question says what happens next
    assert "Choose Collect to look now." in page


def test_add_question_form_asks_only_for_the_question():
    page = _page()

    # the keyword box was never read by matching, so it promised something that didn't happen
    assert 'id="question-interests"' not in page
    assert "Linked Interests" not in page and "Standing Question" not in page
    assert "What do you want to keep up with?" in page


def test_feed_actions_can_be_found_without_hovering():
    page = _page(cards=[_card("a", "article")])

    assert ".card-actions { display: flex; gap: 0; opacity: .6;" in page
    # the score is out of order on purpose (one item per story), so the card offers an explanation, not a bare number
    assert 'onclick="inspectScore(this)">Why this?</button>' in page
    assert "'\" onclick=\"inspectScore(this)\">Why this?</button>" in page  # the page script renders the same


def test_rating_buttons_and_write_answer_say_what_they_do():
    claim, item = _keyword_claim(1, datetime.now(UTC))
    page = _page(questions=[QuestionView(question=LG_QUESTION, claims=[claim], items={item.id: item})])

    assert 'title="I already knew this"' in page
    assert 'title="This helped me"' in page and 'title="This is wrong"' in page
    assert "<strong>Knew it</strong>" in page  # explained in the guide too


def test_guide_names_the_model_that_writes_answers():
    page = _page()

    assert "With Llama running, it also writes a cited answer" not in page
    assert "the answer model writes a cited answer" in page  # named generically: the model is configurable


def test_side_cards_count_sources_one_way():
    sources = [SourceRow(id="a", type="rss", params={}, cadence_s=3600), SourceRow(id="b", type="rss", params={}, cadence_s=3600)]
    impact = {"read": 5, "sources": 1, "since": "2026-10-28", "kept": 2, "filtered": 3, "quotes": 0, "checked": 0, "checked_by": []}
    telemetry = {"fetched": 60, "scored": 38, "judged_llama": 0}

    page = _page(impact=impact, sources=sources, telemetry=telemetry)

    assert "items read from" not in page  # "17 sources" beside "19 of 19 sources OK" read as a contradiction
    assert "2 of 2 websites working" in page
    assert "<dt>Matched your interests</dt><dd>38</dd>" in page
    assert "38 of the 60 items in the last collect matched your interests." in page


def test_digest_links_each_point_to_its_item_instead_of_a_code():
    from gleaner.digest import DigestBullet

    item = Item(id="0ca69a27ffff", source_id="arxiv-cs-ai", title="MAServeBench", text="t", url="https://arxiv.org/abs/1")
    digest = Digest(date=TODAY.isoformat(), bullets=[DigestBullet(text="MAServeBench", item_ids=[item.id])])

    page = _page(digest=digest, digest_items={item.id: item})

    assert "#0ca69a27" not in page
    assert '<a href="https://arxiv.org/abs/1" target="_blank" rel="noopener noreferrer">arxiv-cs-ai</a>' in page


def test_a_first_time_reader_understands_the_page_without_the_guide():
    page = _page(sources=[SourceRow(id="a", type="rss", params={}, cadence_s=3600)])

    # tabs named in everyday words; the internal tab ids stay
    for label in ("My questions", "For you", "Top 5", "Trending", "Deadlines", "Ask", "Websites", "Blocked"):
        assert f"{label}</button>" in page or f">{label}<span" in page, label
    for jargon in (">Brief<", ">Digest<", ">Quarantine<", ">Sources<"):
        assert jargon not in page, jargon
    # the first screen says what gleaner is and what to do, until the reader dismisses it
    assert 'id="welcome"' in page
    assert "gleaner reads 1 website for you and finds the sentences that answer your questions." in page
    assert "gleaner-welcome-done" in page
    # every view has a sentence saying what it is
    assert "Everything gleaner kept for you, best match first." in page
    assert "The websites gleaner reads for you." in page
    assert "Text that tried to give the AI orders." in page
    # ranking choices in plain words
    for mode in ("Best match", "Newest ideas", "Popular now", "Things to do"):
        assert f">{mode}</button>" in page, mode
    assert 'title="Read all your websites now"' in page


# A second walkthrough's report (2026-11-04): places where the page said two different things.


def test_status_line_and_collect_card_agree_on_whether_llama_checked_the_last_collect():
    # "Checked by llama3.2:3b" beside "Checked by Llama: 0": the name came from any old judgment
    telemetry = {"fetched": 60, "scored": 38, "judged_llama": 0}

    page = _page(telemetry=telemetry, judge_model="llama3.2:3b")

    assert "Checked by llama3.2:3b" not in page
    assert "<span>Llama didn\u2019t check the last collect</span>" in page
    assert "Checked by llama3.2:3b" in _page(telemetry=telemetry | {"judged_llama": 12}, judge_model="llama3.2:3b")


def test_a_question_with_no_quotes_has_nothing_to_rate():
    hit_rate = {"useful": 4, "wrong": 0, "already_knew": 0, "total": 4, "hit_rate_pct": 100.0}

    page = _page(questions=[QuestionView(question=LG_QUESTION, hit_rate=hit_rate)])

    assert "Nothing yet." in page
    assert "Was this useful?" not in page and 'id="hitrate-q-lg"' not in page
    assert 'onclick="removeQuestion(this.dataset.qid, this)"' in page


def test_a_question_counts_the_websites_its_quotes_came_from():
    # "19 quotes from 18 sources" counted papers; they came from 4 websites
    now = datetime.now(UTC)
    claims, items = [], {}
    for n in range(3):
        item = Item(id=f"p{n}", source_id="arxiv-test-time-compute", title=f"Paper {n}", text="t", url=f"https://arxiv.org/abs/{n}")
        quote = f"LangGraph checkpoints grew without bound in production, case {n}."
        claims.append(Claim(id=f"c{n}", question_id="q-lg", item_id=item.id, text=quote, quote=quote, stance="context",
                            origin_key=f"arxiv:{n}", model="llama3.2:3b", created_at=now))
        items[item.id] = item

    page = _page(questions=[QuestionView(question=LG_QUESTION, claims=claims, items=items)])

    assert "3 quotes from 1 website" in page


def test_why_this_uses_the_chosen_ranking_and_plain_words():
    from gleaner.scoring import RANKING_MODES

    page = _page(cards=[_card("a", "article")])

    assert "const ALL_WEIGHTS = " + json.dumps(RANKING_MODES) in page
    assert "% in Balanced" not in page and "Vector similarity" not in page and "Rank score = freshness" not in page
    assert "How well it matches your interests" in page


def test_blocked_cards_top_five_and_icon_speak_plainly():
    from gleaner.digest import DigestBullet

    quarantine = [{"title": "Simulated attack", "why": "Blocked as a prompt injection (Matched pattern: instruction_override)", "text": "x"}]
    digest = Digest(date="2026-11-04", title="Daily Briefing — 2026-11-04", bullets=[DigestBullet(text="A", item_ids=["i"])])

    page = _page(quarantine=quarantine, digest=digest)

    assert ">Quarantined<" not in page and "Matched pattern" not in page.split('id="tab-quarantine"')[1].split("</section>")[0]
    assert "It tried to give the AI orders, so gleaner blocked it." in page
    assert "Daily Briefing — 2026-11-04" not in page and "Top 1 for Wednesday, 4 November" in page  # counts the points shown
    assert '<link rel="icon" href="data:image/svg+xml,' in page


def test_add_website_form_uses_everyday_words_and_logs_show_local_time():
    page = _page()

    assert '<h2 id="add-source-title">Add a website</h2>' in page
    assert ">Identifier<" not in page and "Source Trust" not in page and "ArXiv Preprints" not in page
    assert "Time (UTC)" not in page


def _visible_text(page: str) -> list[str]:
    """Text a person can read on the page: content, tooltips, labels and placeholders."""
    from html.parser import HTMLParser

    class Text(HTMLParser):
        def __init__(self):
            super().__init__()
            self.skip, self.out = 0, []

        def handle_starttag(self, tag, attrs):
            self.skip += tag in ("script", "style")
            self.out += [v for k, v in attrs if k in ("title", "aria-label", "placeholder") and v]

        def handle_endtag(self, tag):
            self.skip -= tag in ("script", "style")

        def handle_data(self, data):
            if not self.skip and data.strip():
                self.out.append(data.strip())

    parser = Text()
    parser.feed(page)
    return parser.out


def test_copy_has_no_semicolons_or_dashes():
    """The owner's style: plain sentences, no semicolons and no dashes as punctuation."""
    from gleaner.dashboard import JS
    from gleaner.digest import generate_digest

    now = datetime.now(UTC)
    pairs = [_keyword_claim(n, now - timedelta(minutes=n)) for n in range(2)]
    latest = Answer(question_id="q-lg", version=2, what_changed="Checkpoints keep growing.",
                    what_changed_by="llama3.2:1b", sentences=[Sentence(text="x", claim_ids=["c0"])], created_at=now)
    view = QuestionView(question=LG_QUESTION, latest=latest, claims=[c for c, _ in pairs], items={i.id: i for _, i in pairs})
    bad = [t for t in _visible_text(_page(questions=[view], judge_model="heuristic"))
           if ";" in t or "\u2014" in t or "\u2013" in t]
    assert bad == []
    assert "collected yet; the next" not in JS
    assert "\u2014" not in generate_digest(Database(":memory:")).title


def test_ui_is_simplified_to_primary_views_and_actions():
    """Secondary tabs and card actions are tucked under More, voice says Talk everywhere, and trends use plain words."""
    now = datetime.now(UTC)
    claim, item = _keyword_claim(1, now)
    view = QuestionView(question=LG_QUESTION, claims=[claim], items={item.id: item})
    sources = [SourceRow(id="blog-simon", type="rss", params={"url": "https://simonwillison.net/atom/everything/"})]
    trends = [
        from_trend := __import__("gleaner.models", fromlist=["Trend"]).Trend(
            id="t1", label="Agent checkpoints", story_ids=["s1", "s2"], z_score=2.4, window="24h"
        )
    ]
    assert from_trend.id == "t1"
    impact = {"read": 10, "sources": 1, "since": "2026-10-28", "kept": 4, "filtered": 6, "quotes": 1, "checked": 0, "checked_by": []}
    telemetry = {"fetched": 10, "scored": 4, "judged_llama": 0}

    page = _page(questions=[view], cards=[_card("c1", "article")], sources=sources, trends=trends, impact=impact, telemetry=telemetry)

    # 1. Top bar keeps 4 primary tabs out and tucks Top 5, Trending, Websites and Blocked into #nav-more
    nav_more = page.split('id="nav-more"')[1].split("</details>")[0]
    for sec in ('data-tab="digest"', 'data-tab="trends"', 'data-tab="sources"', 'data-tab="quarantine"'):
        assert sec in nav_more
    # 2. When impact exists, Last collect is folded into the impact card instead of a separate side-card
    assert '<details class="more side-more"><summary>Last collect</summary>' in page
    # 3. Question cards with quotes tuck Find websites and Remove into a More menu
    q_card = page.split('id="question-q-lg"')[1].split("</article>")[0]
    assert '<details class="q-more">' in q_card
    # 4. Feed cards show only More like this and Less like this in .card-actions (no Save/Copy/Ask clutter)
    f_card = page.split('id="card-c1"')[1].split("</article>")[0]
    assert 'data-action="like"' in f_card and 'data-action="dislike"' in f_card
    assert 'data-action="save"' not in f_card and "copyLink(" not in f_card and "askAboutItem(" not in f_card
    # 5. Both voice buttons say Talk (never Speak)
    assert ">Speak<" not in page and "press Speak" not in page
    assert 'id="btn-ask-voice"' in page and "press Talk and speak" in page
    # 6. Trending and Websites use plain English instead of z-scores and bare slugs
    assert "Surge z-score" not in page
    assert "2 items across your websites in the last 24h" in page
    assert "simonwillison.net · RSS" in page

