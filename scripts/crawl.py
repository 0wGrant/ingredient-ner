import argparse
import gzip
import re
import time
from pathlib import Path

import requests

from ingredient_ner.jsonl import read_jsonl
from ingredient_ner.paths import PAGES, PREPARED, SCRAPE_LOG
from ingredient_ner.scraping import SiteSaysSlowDown, crawl, fetch_html, sample_recipes

SAMPLE_SIZE = 30000
SEED = 42
USER_AGENT = "povarenok-texts-study/0.1 (educational, non-commercial)"
TIMEOUT = 30
DELAY = 2.0
JITTER = 1.0


def page_path(url: str) -> Path:
    """Файл сохранённой страницы"""
    name = re.sub(r"[^0-9A-Za-z_-]", "_", url.rstrip("/").rsplit("/", 1)[-1])
    return PAGES / f"{name}.html.gz"


def repair_log() -> None:
    """Обрезка оборванной последней строки журнала"""
    if not SCRAPE_LOG.exists():
        return
    raw = SCRAPE_LOG.read_bytes()
    if not raw.endswith(b"\n"):
        SCRAPE_LOG.write_bytes(raw[:raw.rfind(b"\n") + 1])


def main():
    parser = argparse.ArgumentParser(description="Тексты рецептов со страниц povarenok.ru")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    recipes = sample_recipes(read_jsonl(PREPARED), SAMPLE_SIZE, SEED)[:args.limit]
    PAGES.mkdir(parents=True, exist_ok=True)
    repair_log()

    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT

        def get(url):
            response = session.get(url, timeout=TIMEOUT)
            if response.status_code == 200:
                page_path(url).write_bytes(gzip.compress(response.content))
            return response

        try:
            stats = crawl(recipes, SCRAPE_LOG, lambda url: fetch_html(url, get, time.sleep), sleep=time.sleep,
                          delay=DELAY, jitter=JITTER)
        except SiteSaysSlowDown as e:
            print(f"остановка: {e}")
            return
    print(f"загружено {stats['fetched']}, пропущено {stats['skipped']}")


if __name__ == "__main__":
    main()
