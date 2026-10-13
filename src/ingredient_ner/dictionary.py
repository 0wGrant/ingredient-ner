import json
import re
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import pymorphy3

from .recipes import recipe_text

_LETTERS = re.compile(r"[^\W\d_]+")


def _table(name: str) -> dict:
    return json.loads(Path(__file__).with_name(name).read_text(encoding="utf-8"))


GENERIC = _table("generic_names.json")
PARTS = _table("part_names.json")
FORMS = _table("word_forms.json")


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Буквенные токены с позициями: нижний регистр, ё → е"""
    return [(m.group().lower().replace("ё", "е"), m.start(), m.end()) for m in _LETTERS.finditer(text)]


def lemmatizer(names):
    """Лемматизатор pymorphy3 с кэшем: первый разбор с леммой из названий, иначе первый"""
    morph = pymorphy3.MorphAnalyzer()
    known = set()
    for name in names:
        for token, _start, _end in tokenize(name):
            nominative = next((parse for parse in morph.parse(token) if parse.tag.case == "nomn"), None)
            if nominative:
                known.add(nominative.normal_form)

    @cache
    def lemmatize(word: str) -> str:
        lemmas = [parse.normal_form for parse in morph.parse(word)]
        return next((lemma for lemma in lemmas if lemma in known), lemmas[0])

    return lemmatize


class IngredientMatcher:
    """Поиск названий справочника в тексте по леммам; форма слова находит то же, что её название"""

    def __init__(self, names, lemmatize, forms: dict = FORMS):
        self._lemmatize = lemmatize
        self._index = {}
        for name in names:
            lemmas = self._lemmas(name)
            if not lemmas:
                continue
            self._index.setdefault(tuple(sorted(lemmas)), set()).add(name)
            self._index.setdefault((lemmas[0],), set()).add(name)
        for form, name in forms.items():
            found = self._index.get(tuple(sorted(self._lemmas(name))))
            if found:
                self._index.setdefault(tuple(sorted(self._lemmas(form))), set()).update(found)
        self._max_key_len = max((len(key) for key in self._index), default=0)

    def _lemmas(self, text: str) -> list[str]:
        return [self._lemmatize(token) for token, _start, _end in tokenize(text)]

    def find(self, text: str) -> list[dict]:
        """Упоминания: start, end, text, candidates"""
        tokens = [(self._lemmatize(token), start, end) for token, start, end in tokenize(text)]
        result = []
        i = 0
        while i < len(tokens):
            for size in range(min(self._max_key_len, len(tokens) - i), 0, -1):
                window = tokens[i:i + size]
                key = tuple(sorted(lemma for lemma, _start, _end in window))
                if key in self._index:
                    start, end = window[0][1], window[-1][2]
                    result.append({"start": start, "end": end, "text": text[start:end],
                                   "candidates": sorted(self._index[key])})
                    i += size
                    break
            else:
                i += 1
        return result


def canonical_names(freq: Counter, lemmatize) -> dict[str, str]:
    """Название → самое частое название с теми же леммами"""
    groups = defaultdict(list)
    for name in freq:
        key = tuple(sorted(lemmatize(token) for token, _start, _end in tokenize(name)))
        if key:
            groups[key].append(name)
    canon = {}
    for names in groups.values():
        main = min(names, key=lambda name: (-freq[name], name))
        canon |= {name: main for name in names if name != main}
    return canon


def split_by_recipe(mentions: list[dict], ingredients, generic: dict = GENERIC,
                    parts: dict = PARTS) -> tuple[list[dict], list[dict]]:
    """Находки со ссылкой на состав и остальные: общее → частное, часть ↔ целое, все кандидаты частные → общее"""
    ingredients = set(ingredients)
    broader, contained = {}, {}
    for general, specifics in generic.items():
        for name in specifics:
            broader.setdefault(name, set()).add(general)
    for part, wholes in parts.items():
        for whole in wholes:
            contained.setdefault(whole, set()).add(part)
    kept, dropped = [], []
    for mention in mentions:
        candidates = mention["candidates"]
        names = set(candidates).union(*(generic.get(name, ()) for name in candidates),
                                      *(parts.get(name, ()) for name in candidates),
                                      *(contained.get(name, ()) for name in candidates))
        specific = bool(candidates) and all(ingredients & broader.get(name, set()) for name in candidates)
        (kept if ingredients & names or specific else dropped).append(mention)
    return kept, dropped


def recipe_coverage(recipe: dict, matcher: IngredientMatcher) -> dict:
    """Ингредиенты рецепта, найденные и не найденные в тексте"""
    mentioned = set()
    for mention in matcher.find(recipe_text(recipe)):
        mentioned.update(mention["candidates"])
    return {
        "found": [name for name in recipe["ingredients"] if name in mentioned],
        "missed": [name for name in recipe["ingredients"] if name not in mentioned],
    }


def predict_dictionary(records: list[dict], recipes: list[dict], lemmatize) -> tuple[dict, dict]:
    """url → находки словаря: как есть и после фильтра по составу"""
    matcher = IngredientMatcher({name for recipe in recipes for name in recipe["ingredients"]}, lemmatize)
    ingredients = {recipe["url"]: recipe["ingredients"] for recipe in recipes}
    found = {record["url"]: matcher.find(record["text"]) for record in records}
    filtered = {url: split_by_recipe(spans, ingredients[url])[0] for url, spans in found.items()}
    return found, filtered
