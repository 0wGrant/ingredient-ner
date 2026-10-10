import random

import pytest
import requests

from ingredient_ner.jsonl import append_jsonl
from ingredient_ner.scraping import (SiteSaysSlowDown, assemble, crawl, fetch_html, parse_recipe_page, read_log,
                                     sample_recipes, titles_match)

PAGE = """<html><head><title>Торт «Наполеон» – кулинарный рецепт</title></head><body>
<h1 itemprop="name">Торт  &laquo;Наполеон&raquo;</h1>
<div class="article-text" itemprop="description">
  <p>Классический торт из&nbsp;слоёного теста.</p>
</div>
<h2>Рецепт &laquo;Торт Наполеон&raquo;:</h2>
<ul itemprop="recipeInstructions">
  <li class="cooking-bl">
    <span class="cook-img"><a href="#"><img itemprop="image" alt="" src="1.jpg"></a></span>
    <div><p>Муку&nbsp;смешать с холодным <b>маслом</b>, добавить яйцо.</p></div>
  </li>
  <li class="cooking-bl"><div><p>   </p></div></li>
  <li class="cooking-bl"><div><p>Испечь коржи.</p><p>Смазать кремом</p></div></li>
  <li class="cooking-bl"><div><p>Дать настояться<br>ночь в холодильнике.</p></div></li>
</ul>
<div class="article-tags">Теги рецепта</div>
</body></html>"""

EXPECTED_STEPS = [
    "Муку смешать с холодным маслом, добавить яйцо.",
    "Испечь коржи. Смазать кремом",
    "Дать настояться ночь в холодильнике.",
]

TEXT_PAGE = """<html><body>
<h1 itemprop="name">Салат &laquo;Поляна&raquo;</h1>
<div class="article-text" itemprop="description"><p>Простой салат.</p></div>
<h2>Ингредиенты для &laquo;Салат Поляна&raquo;:</h2>
<h2>Рецепт &laquo;Салат Поляна&raquo;:</h2>
<div>Обжарить лук
на сковороде.<br>
<br>
Нарезать <b>курицу</b>, смешать с&nbsp;луком.<br><br>   <br>Подробности <a href="https://example.com/r/1">в другом рецепте</a>, там проще.<br>Подавать холодным.<img src="1.jpg"><img src="2.jpg"></div>
<div class="article-tags"><p>Теги: салат</p></div>
</body></html>"""

TEXT_STEPS = [
    "Обжарить лук на сковороде.",
    "Нарезать курицу, смешать с луком.",
    "Подробности в другом рецепте, там проще.",
    "Подавать холодным.",
]

VIDEO_PAGE = """<html><body>
<h1 itemprop="name">Рыбные котлеты</h1>
<div class="article-text" itemprop="description"><p>Котлеты из фарша.</p></div>
<h2>Рецепт &laquo;Рыбные котлеты&raquo;:</h2>
<div class="video-bl"><div><iframe src="https://example.com/video"></iframe></div></div>
<div>Разморозить фарш.<br>Добавить яйцо и&nbsp;соль.<br><br>Жарить до корочки.</div>
<div class="article-tags"><p>Теги: рыба котлеты</p></div>
</body></html>"""

VIDEO_STEPS = ["Разморозить фарш.", "Добавить яйцо и соль.", "Жарить до корочки."]


def video_page(after_video):
    return f"""<h1 itemprop="name">Кекс</h1>
<h2>Рецепт &laquo;Кекс&raquo;:</h2>
<div class="video-bl"><div><iframe src="https://example.com/video"></iframe></div></div>
{after_video}
<div class="article-tags"><p>Теги: выпечка кекс</p></div>"""


def scrape_record(url, status=200, title="Торт «Наполеон»", steps=("Шаг.",), description="Описание.", error=None,
                  layout="steps"):
    return {"url": url, "status": status, "error": error, "title": title, "description": description,
            "steps": list(steps), "layout": layout}


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


def make_get(*outcomes):
    calls = []

    def get(url):
        calls.append(url)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    get.calls = calls
    return get


class TestReadLog:
    def test_missing_file_is_empty(self, tmp_path):
        assert read_log(tmp_path / "nothing.jsonl") == []

    def test_tolerates_broken_last_line(self, tmp_path):
        path = tmp_path / "log.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n{"c": "обо', encoding="utf-8")
        assert read_log(path) == [{"a": 1}, {"b": 2}]

    def test_raises_on_broken_middle_line(self, tmp_path):
        path = tmp_path / "log.jsonl"
        path.write_text('{"a": 1}\n{"b": \n{"c": 3}\n', encoding="utf-8")
        with pytest.raises(ValueError):
            read_log(path)


class TestSampleRecipes:
    recipes = [{"url": f"u{i}"} for i in range(100)]

    def test_size(self):
        assert len(sample_recipes(self.recipes, 10, seed=1)) == 10

    def test_same_seed_same_sample(self):
        assert sample_recipes(self.recipes, 10, seed=1) == sample_recipes(self.recipes, 10, seed=1)

    def test_different_seeds_differ(self):
        assert sample_recipes(self.recipes, 10, seed=1) != sample_recipes(self.recipes, 10, seed=2)

    def test_n_larger_than_population_returns_all(self):
        small = self.recipes[:5]
        assert {id(r) for r in sample_recipes(small, 50, seed=0)} == {id(r) for r in small}


class TestParseRecipePage:
    def test_title(self):
        assert parse_recipe_page(PAGE)["title"] == "Торт «Наполеон»"

    def test_description(self):
        assert parse_recipe_page(PAGE)["description"] == "Классический торт из слоёного теста."

    def test_steps(self):
        result = parse_recipe_page(PAGE)
        assert result["steps"] == EXPECTED_STEPS
        assert result["layout"] == "steps"

    def test_missing_blocks(self):
        page = "<html><body><p>Страница не найдена</p></body></html>"
        assert parse_recipe_page(page) == {"title": None, "description": None, "steps": [], "layout": "unknown"}

    def test_empty_description_is_none(self):
        page = '<h1 itemprop="name">Борщ</h1><div itemprop="description">  </div>'
        assert parse_recipe_page(page)["description"] is None

    def test_text_layout(self):
        result = parse_recipe_page(TEXT_PAGE)
        assert result["layout"] == "text"
        assert result["steps"] == TEXT_STEPS
        assert result["title"] == "Салат «Поляна»"

    def test_video_layout_with_text(self):
        result = parse_recipe_page(VIDEO_PAGE)
        assert result["layout"] == "video"
        assert result["steps"] == VIDEO_STEPS

    def test_video_with_empty_text_block(self):
        assert parse_recipe_page(video_page("<div></div>")) == {
            "title": "Кекс", "description": None, "steps": [], "layout": "video"}

    def test_video_without_text_block_does_not_take_tags(self):
        result = parse_recipe_page(video_page(""))
        assert result["layout"] == "video"
        assert result["steps"] == []

    def test_other_h2_is_not_recipe_heading(self):
        page = '<h1 itemprop="name">Борщ</h1><h2>Рецепты из этой категории</h2><div>Щи<br>Солянка</div>'
        assert parse_recipe_page(page)["layout"] == "unknown"

    def test_recipe_heading_followed_by_unexpected_tag(self):
        page = '<h1 itemprop="name">Борщ</h1><h2>Рецепт &laquo;Борщ&raquo;:</h2><p>Сварить.</p>'
        result = parse_recipe_page(page)
        assert result["layout"] == "unknown"
        assert result["steps"] == []


class TestTitlesMatch:
    def test_case_and_spaces(self):
        assert titles_match("  Борщ   Украинский ", "борщ украинский")

    def test_yo(self):
        assert titles_match("Ёжики из фарша", "ежики из фарша")

    def test_quote_styles(self):
        assert titles_match("Торт «Наполеон»", 'Торт "Наполеон"')

    def test_different(self):
        assert not titles_match("Борщ", "Щи")

    def test_missing_page_title(self):
        assert not titles_match(None, "Борщ")

    def test_empty_titles_never_match(self):
        assert not titles_match("«»", "")


class TestFetchHtml:
    def test_ok(self):
        result = fetch_html("u", make_get(FakeResponse(200, "<html>ok</html>")), sleep=lambda s: None)
        assert result == {"status": 200, "html": "<html>ok</html>", "error": None}

    def test_not_found_no_retry(self):
        get = make_get(FakeResponse(404))
        assert fetch_html("u", get, sleep=lambda s: None) == {"status": 404, "html": None, "error": None}
        assert len(get.calls) == 1

    @pytest.mark.parametrize("code", [429, 503])
    def test_slow_down_statuses_raise(self, code):
        with pytest.raises(SiteSaysSlowDown):
            fetch_html("u", make_get(FakeResponse(code)), sleep=lambda s: None)

    def test_retry_then_success(self):
        sleeps = []
        get = make_get(requests.ConnectionError(), FakeResponse(200, "ok"))
        assert fetch_html("u", get, sleep=sleeps.append)["status"] == 200
        assert sleeps == [2.0]

    def test_all_retries_fail(self):
        sleeps = []
        get = make_get(requests.ConnectionError(), requests.Timeout(), requests.ConnectionError())
        result = fetch_html("u", get, sleep=sleeps.append, retries=3, backoff=2.0)
        assert result == {"status": None, "html": None, "error": "ConnectionError"}
        assert sleeps == [2.0, 4.0]


class TestCrawl:
    @staticmethod
    def fetcher(pages, calls):
        def fetch(url):
            calls.append(url)
            if url in pages:
                return {"status": 200, "html": pages[url], "error": None}
            return {"status": 404, "html": None, "error": None}
        return fetch

    def test_writes_parsed_record(self, tmp_path):
        log = tmp_path / "log.jsonl"
        crawl([{"url": "u1"}], log, self.fetcher({"u1": PAGE}, []), sleep=lambda s: None)
        assert read_log(log) == [{"url": "u1", "status": 200, "error": None, "title": "Торт «Наполеон»",
                                  "description": "Классический торт из слоёного теста.", "steps": EXPECTED_STEPS,
                                  "layout": "steps"}]

    def test_failed_page_record(self, tmp_path):
        log = tmp_path / "log.jsonl"
        crawl([{"url": "gone"}], log, self.fetcher({}, []), sleep=lambda s: None)
        assert read_log(log) == [{"url": "gone", "status": 404, "error": None, "title": None, "description": None,
                                  "steps": [], "layout": None}]

    def test_pauses_between_requests(self, tmp_path):
        sleeps = []
        crawl([{"url": u} for u in ("u1", "u2", "u3")], tmp_path / "log.jsonl", self.fetcher({}, []),
              sleep=sleeps.append, delay=2.0, jitter=1.0, rng=random.Random(0))
        assert len(sleeps) == 2
        assert all(2.0 <= s <= 3.0 for s in sleeps)

    def test_resume_skips_answered_urls(self, tmp_path):
        log = tmp_path / "log.jsonl"
        append_jsonl(log, scrape_record("u1"))
        append_jsonl(log, scrape_record("u2", status=404, title=None, steps=()))
        calls = []
        stats = crawl([{"url": u} for u in ("u1", "u2", "u3")], log, self.fetcher({}, calls), sleep=lambda s: None)
        assert calls == ["u3"]
        assert stats == {"fetched": 1, "skipped": 2}

    def test_network_errors_are_retried_on_resume(self, tmp_path):
        log = tmp_path / "log.jsonl"
        append_jsonl(log, scrape_record("u1", status=None, title=None, steps=(), description=None,
                                        error="ConnectionError"))
        calls = []
        crawl([{"url": "u1"}], log, self.fetcher({"u1": PAGE}, calls), sleep=lambda s: None)
        assert calls == ["u1"]

    def test_slow_down_stops_and_keeps_log(self, tmp_path):
        log = tmp_path / "log.jsonl"

        def fetch(url):
            if url == "u2":
                raise SiteSaysSlowDown()
            return {"status": 200, "html": PAGE, "error": None}

        with pytest.raises(SiteSaysSlowDown):
            crawl([{"url": "u1"}, {"url": "u2"}, {"url": "u3"}], log, fetch, sleep=lambda s: None)
        assert [r["url"] for r in read_log(log)] == ["u1"]


class TestAssemble:
    prepared = [
        {"url": "u1", "name": "Торт «Наполеон»", "ingredients": {"Мука": "300 г"}},
        {"url": "u2", "name": "Борщ", "ingredients": {"Свекла": "1 шт"}},
    ]

    def test_joins_fields(self):
        scraped = [scrape_record("u1", steps=("Испечь.",), description="Торт.")]
        assert assemble(self.prepared, scraped) == [{"url": "u1", "name": "Торт «Наполеон»",
                                                     "ingredients": {"Мука": "300 г"}, "description": "Торт.",
                                                     "steps": ["Испечь."]}]

    @pytest.mark.parametrize("record", [scrape_record("u1", status=404), scrape_record("u1", steps=()),
                                        scrape_record("u1", title="Слойки с начинкой")])
    def test_skips_bad_records(self, record):
        assert assemble(self.prepared, [record]) == []

    def test_error_then_success_uses_success(self):
        scraped = [scrape_record("u2", status=None, title=None, steps=(), error="Timeout"),
                   scrape_record("u2", title="Борщ", steps=("Сварить.",))]
        assert [r["steps"] for r in assemble(self.prepared, scraped)] == [["Сварить."]]

    def test_order_follows_prepared(self):
        scraped = [scrape_record("u2", title="Борщ"), scrape_record("u1")]
        assert [r["url"] for r in assemble(self.prepared, scraped)] == ["u1", "u2"]
