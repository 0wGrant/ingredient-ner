import json
from pathlib import Path

from .context import mention_statuses
from .dictionary import IngredientMatcher
from .dish_metrics import allergen_levels

ALLERGENS = ("арахис", "аспартам", "горчица", "сульфиты", "глютен", "кунжут", "люпин", "моллюски", "молоко", "орехи",
             "ракообразные", "рыба", "сельдерей", "соя", "яйца")
TABLE = Path(__file__).with_name("allergen_table.json")
_RANK = {"may": 1, "contains": 2}


def load_table(path: Path = TABLE) -> dict[str, dict[str, str]]:
    """Название справочника → аллерген → уровень"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Linker:
    """Упоминание → кандидаты из справочника"""

    def __init__(self, names, lemmatize):
        self._matcher = IngredientMatcher(names, lemmatize)

    def link(self, text: str) -> list[list[str]]:
        """Кандидаты каждого названия внутри упоминания"""
        return [mention["candidates"] for mention in self._matcher.find(text)]


def candidate_allergens(candidates: list[str], table: dict) -> dict[str, str]:
    """Аллергены названия: у всех кандидатов — общий уровень, у части — may"""
    levels = [table.get(name, {}) for name in candidates]
    result = {}
    for allergen in sorted({a for level in levels for a in level}):
        found = [level[allergen] for level in levels if allergen in level]
        result[allergen] = min(found, key=_RANK.__getitem__) if len(found) == len(levels) else "may"
    return result


def mention_allergens(groups: list[list[str]], table: dict) -> dict[str, str]:
    """Аллергены упоминания: по всем названиям внутри, сильнейший уровень"""
    result = {}
    for candidates in groups:
        for allergen, level in candidate_allergens(candidates, table).items():
            if allergen not in result or _RANK[level] > _RANK[result[allergen]]:
                result[allergen] = level
    return result


def link_mentions(text: str, spans: list[dict], statuses: list[str], linker: Linker, table: dict) -> list[dict]:
    """Упоминания со статусом, кандидатами и аллергенами"""
    result = []
    for span, status in zip(spans, statuses, strict=True):
        groups = linker.link(text[span["start"]:span["end"]])
        result.append({"start": span["start"], "end": span["end"], "text": text[span["start"]:span["end"]],
                       "status": status, "candidates": groups, "allergens": mention_allergens(groups, table)})
    return result


def gold_mentions(annotated: dict, linker: Linker, table: dict) -> list[dict]:
    """Упоминания эталона с аллергенами по ссылке, без ссылки — по тексту"""
    result = []
    for mention in annotated["mentions"]:
        if mention["ingredient"] is not None:
            allergens = dict(table.get(mention["ingredient"], {}))
        else:
            allergens = mention_allergens(linker.link(mention["text"]), table)
        result.append({"status": mention["status"], "allergens": allergens})
    return result


def dish_levels(records: list[dict], spans: dict, linker: Linker, table: dict, lexicon: list[dict]) -> dict:
    """url → аллергены блюда по отрезкам системы"""
    result = {}
    for record in records:
        text, found = record["text"], spans.get(record["url"], [])
        mentions = link_mentions(text, found, mention_statuses(text, found, lexicon), linker, table)
        result[record["url"]] = allergen_levels(mentions)
    return result


def gold_levels(annotated: list[dict], linker: Linker, table: dict) -> dict:
    """url → аллергены блюда по ручной разметке"""
    return {record["url"]: allergen_levels(gold_mentions(record, linker, table)) for record in annotated}


def analyze(text: str, spans: list[dict], linker: Linker, table: dict, lexicon: list[dict]) -> dict:
    """Упоминания и аллергены блюда по тексту и найденным отрезкам"""
    mentions = link_mentions(text, spans, mention_statuses(text, spans, lexicon), linker, table)
    return {"mentions": mentions, "allergens": allergen_levels(mentions)}
