import re

from .bio import split_words

LEXICON = [
    {"phrase": "без", "category": "absent", "direction": "forward"},
    {"phrase": "не добавля*", "category": "absent", "direction": "both"},
    {"phrase": "не клад*", "category": "absent", "direction": "both"},
    {"phrase": "не использ*", "category": "absent", "direction": "both"},
    {"phrase": "не надо", "category": "absent", "direction": "backward"},
    {"phrase": "не нужн*", "category": "absent", "direction": "backward"},
    {"phrase": "не нужен", "category": "absent", "direction": "backward"},
    {"phrase": "не понадоб*", "category": "absent", "direction": "backward"},
    {"phrase": "не ем", "category": "absent", "direction": "backward"},
    {"phrase": "консистенци*", "category": "absent", "direction": "forward"},
    {"phrase": "размером", "category": "absent", "direction": "forward"},
    {"phrase": "величиной", "category": "absent", "direction": "forward"},
    {"phrase": "как густая", "category": "absent", "direction": "forward"},
    {"phrase": "оригинальн*", "category": "absent", "direction": "forward"},
    {"phrase": "по желанию", "category": "optional", "direction": "both"},
    {"phrase": "при желании", "category": "optional", "direction": "forward"},
    {"phrase": "можно", "category": "optional", "direction": "forward"},
    {"phrase": "или", "category": "optional", "direction": "forward"},
    {"phrase": "подава*", "category": "optional", "direction": "forward"},
    {"phrase": "подать", "category": "optional", "direction": "forward"},
    {"phrase": "подаем", "category": "optional", "direction": "forward"},
    {"phrase": "подаю", "category": "optional", "direction": "forward"},
    {"phrase": "подойд*", "category": "optional", "direction": "forward"},
    {"phrase": "желательно", "category": "optional", "direction": "forward"},
    {"phrase": "рекомендую", "category": "optional", "direction": "forward"},
    {"phrase": "не надо жалеть", "category": "pseudo"},
    {"phrase": "не жале*", "category": "pseudo"},
    {"phrase": "но", "category": "terminate"},
]
WINDOW = 5

_SENTENCE_END = re.compile(r"[.!?…\n]")


def _normalize(word: str) -> str:
    return word.lower().replace("ё", "е")


def _word_matches(pattern: str, word: str) -> bool:
    return word.startswith(pattern[:-1]) if pattern.endswith("*") else word == pattern


def _overlap(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def find_triggers(text: str, lexicon: list[dict]) -> list[dict]:
    """Триггеры в тексте: start, end, category, direction"""
    words = [(_normalize(word), start, end) for word, start, end in split_words(text)]
    found = []
    for entry in lexicon:
        pattern = [_normalize(word) for word in entry["phrase"].split()]
        for i in range(len(words) - len(pattern) + 1):
            if all(_word_matches(p, words[i + j][0]) for j, p in enumerate(pattern)):
                found.append({"start": words[i][1], "end": words[i + len(pattern) - 1][2],
                              "category": entry["category"], "direction": entry.get("direction")})

    pseudo = [t for t in found if t["category"] == "pseudo"]
    candidates = [t for t in found if t["category"] != "pseudo" and not any(_overlap(t, p) for p in pseudo)]
    candidates.sort(key=lambda t: (t["start"] - t["end"], t["start"]))
    kept = []
    for trigger in candidates:
        if not any(_overlap(trigger, other) for other in kept):
            kept.append(trigger)
    return sorted(kept, key=lambda t: t["start"])


def mention_statuses(text: str, spans: list[dict], lexicon: list[dict], window: int | None = WINDOW) -> list[str]:
    """Статус каждого отрезка: present, optional, absent"""
    triggers = find_triggers(text, lexicon)
    stops = [t for t in triggers if t["category"] == "terminate"]

    def reaches(left: int, right: int) -> bool:
        if _SENTENCE_END.search(text, left, right):
            return False
        if any(left <= stop["start"] and stop["end"] <= right for stop in stops):
            return False
        return window is None or len(split_words(text[left:right])) <= window

    statuses = []
    for span in spans:
        hits = set()
        for t in triggers:
            if t["category"] not in ("absent", "optional"):
                continue
            forward = t["direction"] in ("forward", "both") and t["end"] <= span["start"]
            backward = t["direction"] in ("backward", "both") and span["end"] <= t["start"]
            if (forward and reaches(t["end"], span["start"])) or (backward and reaches(span["end"], t["start"])):
                hits.add(t["category"])
        statuses.append("absent" if "absent" in hits else "optional" if "optional" in hits else "present")
    return statuses
