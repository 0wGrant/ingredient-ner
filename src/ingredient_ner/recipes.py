import json
import re
from collections import Counter
from pathlib import Path

from .jsonl import read_jsonl
from .paths import CANON, RECIPES

_WS = re.compile(r"\s+")
_BRACKETS = re.compile(r"\([^)]*\)")


def normalize_text(text: str | None) -> str:
    """Нижний регистр, схлопнутые пробелы"""
    if not text:
        return ""
    return _WS.sub(" ", text.strip().lower())


def clean_ingredient_name(name: str | None) -> str:
    """Название без скобок"""
    if not name:
        return ""
    return normalize_text(_BRACKETS.sub(" ", name))


def clean_recipes(recipes: list[dict]) -> list[dict]:
    """Рецепты с полем cleaned_ingredients"""
    cleaned_recipes = []
    for recipe in recipes:
        cleaned_ingredients = []
        seen = set()
        for raw_name in recipe["ingredients"]:
            name = clean_ingredient_name(raw_name)
            if name and name not in seen:
                cleaned_ingredients.append(name)
                seen.add(name)
        cleaned_recipes.append(recipe | {"cleaned_ingredients": cleaned_ingredients})
    return cleaned_recipes


def dedupe_recipes(cleaned_recipes: list[dict]) -> list[dict]:
    """Без дублей по названию и составу"""
    deduped_recipes = []
    seen = set()
    for recipe in cleaned_recipes:
        key = normalize_text(recipe["name"]), frozenset(recipe["cleaned_ingredients"])
        if key not in seen:
            seen.add(key)
            deduped_recipes.append(recipe)
    return deduped_recipes


def filter_recipes(cleaned_recipes: list[dict], min_ingredients: int = 2, max_ingredients: int = 30) -> list[dict]:
    """Рецепты с числом ингредиентов в [min, max]"""
    return [r for r in cleaned_recipes if min_ingredients <= len(r["cleaned_ingredients"]) <= max_ingredients]


def build_ingredient_vocab(cleaned_recipes: list[dict]) -> Counter[str]:
    """Название → число рецептов"""
    vocab = Counter()
    for recipe in cleaned_recipes:
        vocab.update(recipe["cleaned_ingredients"])
    return vocab


def canonicalize_recipes(recipes: list[dict], canon: dict[str, str]) -> list[dict]:
    """Рецепты с каноническими названиями ингредиентов"""
    result = []
    for recipe in recipes:
        ingredients = {}
        for name, amount in recipe["ingredients"].items():
            ingredients.setdefault(canon.get(name, name), amount)
        result.append(recipe | {"ingredients": ingredients})
    return result


def recipe_text(recipe: dict) -> str:
    """Описание и шаги одной строкой"""
    if recipe["description"]:
        return "\n".join([recipe["description"], *recipe["steps"]])
    return "\n".join(recipe["steps"])


def load_recipes(path: Path = RECIPES, canon_path: Path = CANON) -> list[dict]:
    """Рецепты с текстами, названия ингредиентов канонические"""
    canon = json.loads(Path(canon_path).read_text(encoding="utf-8"))
    return canonicalize_recipes(read_jsonl(path), canon)
