import sys
from collections import Counter

from ingredient_ner.dataset import gold_record, near_duplicates, silver_record, validate_gold
from ingredient_ner.dictionary import IngredientMatcher, lemmatizer
from ingredient_ner.jsonl import read_jsonl, write_jsonl
from ingredient_ner.paths import DEV_GOLD, NER, TEST_GOLD
from ingredient_ner.recipes import load_recipes, recipe_text

DUPLICATE_THRESHOLD = 0.5


def read_gold(path) -> list[dict]:
    """Ручная разметка; при механических ошибках — выход"""
    annotated = read_jsonl(path)
    broken = [(record["url"], problems) for record in annotated if (problems := validate_gold(record))]
    if broken:
        sys.exit(f"{path.name}: ошибки разметки {broken[:5]}")
    return annotated


def describe(name: str, records: list[dict]) -> None:
    tags = Counter(tag for record in records for tag in record["tags"])
    words = sum(tags.values())
    spans = sum(len(record["spans"]) for record in records)
    print(f"{name:5} рецептов {len(records):6}, слов {words:8}, отрезков {spans:6}, "
          f"B {tags['B-ING'] / words:.1%}, I {tags['I-ING'] / words:.1%}")


def main():
    recipes = load_recipes()
    names = {name for recipe in recipes for name in recipe["ingredients"]}
    matcher = IngredientMatcher(names, lemmatizer(names))

    test = [gold_record(record) for record in read_gold(TEST_GOLD)]
    dev = [gold_record(record) for record in read_gold(DEV_GOLD)]
    held_out = {record["url"]: record["text"] for record in test + dev}
    pool = {recipe["url"]: recipe for recipe in recipes if recipe["url"] not in held_out}
    texts = {url: recipe_text(recipe) for url, recipe in pool.items()}

    duplicates = near_duplicates(held_out, texts, DUPLICATE_THRESHOLD)
    removed = {url for _held, url, _jaccard in duplicates}
    print(f"пул {len(pool)}, почти дублей dev/test {len(removed)}")

    train = [silver_record(url, texts[url], matcher.find(texts[url]), recipe["ingredients"])
             for url, recipe in pool.items() if url not in removed]

    for name, records in (("train", train), ("dev", dev), ("test", test)):
        write_jsonl(NER / f"{name}.jsonl", records)
        describe(name, records)


if __name__ == "__main__":
    main()
