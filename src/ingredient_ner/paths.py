import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("INGREDIENT_NER_DATA", ROOT / "data"))

PREPARED = DATA / "recipes_prepared.jsonl"
SCRAPE_LOG = DATA / "scrape_log.jsonl"
PAGES = DATA / "pages"
RECIPES = DATA / "recipes_with_texts.jsonl"
CANON = DATA / "ingredient_canon.json"
TEST_GOLD = DATA / "gold_mentions.jsonl"
DEV_GOLD = DATA / "dev_mentions.jsonl"
NER = DATA / "ner"
MODELS = DATA / "models"
