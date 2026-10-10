from collections import Counter

from ingredient_ner.jsonl import read_jsonl, write_jsonl
from ingredient_ner.paths import PREPARED, RECIPES, SCRAPE_LOG
from ingredient_ner.scraping import assemble, read_log


def main():
    scraped = read_log(SCRAPE_LOG)
    dataset = assemble(read_jsonl(PREPARED), scraped)
    write_jsonl(RECIPES, dataset)

    statuses = Counter(record["status"] for record in scraped)
    layouts = Counter(record["layout"] for record in scraped if record["status"] == 200)
    print(f"в журнале {len(scraped)}, в датасете {len(dataset)}")
    print(f"статусы {dict(statuses)}; шаблоны {dict(layouts)}")


if __name__ == "__main__":
    main()
