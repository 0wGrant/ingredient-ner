import json
import random
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

from .jsonl import append_jsonl

_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_QUOTES = re.compile(r"[«»\"“”„]")
_RECIPE_HEADING = "Рецепт «"


class SiteSaysSlowDown(Exception):
    """Сайт ответил 429 или 503"""


def read_log(path: Path) -> list[dict]:
    """Записи журнала обхода, оборванная последняя строка пропускается"""
    path = Path(path)
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            if i < len(lines) - 1:
                raise ValueError(f"битая строка {line!r:.80}") from e
    return records


def sample_recipes(recipes: list[dict], n: int, seed: int) -> list[dict]:
    """n случайных рецептов"""
    return random.Random(seed).sample(recipes, min(n, len(recipes)))


def clean_text(text: str) -> str:
    """Схлопнутые пробелы, без пробела перед знаками препинания"""
    return _SPACE_BEFORE_PUNCT.sub(r"\1", _WS.sub(" ", text).strip())


def split_text_block(block: Tag) -> list[str]:
    """Шаги текстового блока, разделённые <br>"""
    steps = []
    current = []
    for child in block.children:
        if isinstance(child, Tag) and child.name == "br":
            steps.append(" ".join(current))
            current = []
            continue
        current.append(child.get_text(" ") if isinstance(child, Tag) else str(child))
    steps.append(" ".join(current))
    return [step for step in map(clean_text, steps) if step]


def parse_recipe_page(html: str) -> dict:
    """title, description, steps, layout страницы рецепта"""
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.select_one('h1[itemprop="name"]')
    title = clean_text(h1.get_text(" ")) if h1 is not None else None

    description_tag = soup.select_one('[itemprop="description"]')
    description = clean_text(description_tag.get_text(" ")) if description_tag is not None else None
    description = description or None

    instructions = soup.select_one('ul[itemprop="recipeInstructions"]')
    if instructions is not None:
        steps = []
        for li in instructions.find_all("li", recursive=False):
            step = clean_text(" ".join(p.get_text(" ") for p in li.find_all("p")))
            if step:
                steps.append(step)
        return {"title": title, "description": description, "steps": steps, "layout": "steps"}

    heading = next((h2 for h2 in soup.find_all("h2") if clean_text(h2.get_text(" ")).startswith(_RECIPE_HEADING)),
                   None)
    block = heading.find_next_sibling() if heading is not None else None

    if block is not None and block.name == "div" and "video-bl" in (block.get("class") or []):
        after_video = block.find_next_sibling()
        if after_video is not None and after_video.name == "div" and not after_video.get("class"):
            steps = split_text_block(after_video)
        else:
            steps = []
        return {"title": title, "description": description, "steps": steps, "layout": "video"}

    if block is not None and block.name == "div" and not block.get("class"):
        return {"title": title, "description": description, "steps": split_text_block(block), "layout": "text"}

    return {"title": title, "description": description, "steps": [], "layout": "unknown"}


def normalize_title(title: str | None) -> str:
    """Нижний регистр, е вместо ё, без кавычек"""
    if not title:
        return ""
    return _WS.sub(" ", _QUOTES.sub("", title.lower().replace("ё", "е"))).strip()


def titles_match(page_title: str | None, dataset_name: str | None) -> bool:
    """Название на странице совпадает с названием в датасете"""
    page = normalize_title(page_title)
    return bool(page) and page == normalize_title(dataset_name)


def fetch_html(url: str, get, sleep, retries: int = 3, backoff: float = 2.0) -> dict:
    """status, html, error; повторы при сетевых ошибках"""
    error = None
    for attempt in range(retries):
        try:
            response = get(url)
        except requests.RequestException as e:
            error = type(e).__name__
            if attempt < retries - 1:
                sleep(backoff * 2 ** attempt)
            continue

        if response.status_code in (429, 503):
            raise SiteSaysSlowDown(f"{url}: {response.status_code}")
        if response.status_code != 200:
            return {"status": response.status_code, "html": None, "error": None}
        return {"status": 200, "html": response.text, "error": None}

    return {"status": None, "html": None, "error": error}


def crawl(recipes: list[dict], log_path: Path, fetch, sleep,
          delay: float = 2.0, jitter: float = 1.0, rng: random.Random | None = None) -> dict:
    """Обход рецептов с журналом; отвеченные url пропускаются"""
    rng = rng or random.Random()
    answered = {record["url"] for record in read_log(log_path) if record["status"] is not None}

    fetched = skipped = 0
    for recipe in recipes:
        url = recipe["url"]
        if url in answered:
            skipped += 1
            continue
        if fetched:
            sleep(delay + rng.uniform(0, jitter))

        result = fetch(url)
        if result["html"] is not None:
            parsed = parse_recipe_page(result["html"])
        else:
            parsed = {"title": None, "description": None, "steps": [], "layout": None}
        append_jsonl(log_path, {"url": url, "status": result["status"], "error": result["error"]} | parsed)
        fetched += 1

    return {"fetched": fetched, "skipped": skipped}


def assemble(prepared: list[dict], scraped: list[dict]) -> list[dict]:
    """Рецепты датасета с текстами из журнала"""
    by_url = {}
    for record in scraped:
        by_url.setdefault(record["url"], []).append(record)

    dataset = []
    for recipe in prepared:
        for record in by_url.get(recipe["url"], []):
            if record["status"] != 200 or not record["steps"] or not titles_match(record["title"], recipe["name"]):
                continue
            dataset.append({
                "url": recipe["url"],
                "name": recipe["name"],
                "ingredients": recipe["ingredients"],
                "description": record["description"],
                "steps": record["steps"],
            })
            break
    return dataset
