from .bio import covered_words, spans_to_bio, split_words
from .dictionary import split_by_recipe


def bio_record(url: str, text: str, spans: list[dict], uncertain: list[dict] = ()) -> dict:
    """Запись набора NER: слова, позиции, BIO-метки, слова без доверия"""
    words = split_words(text)
    offsets = [(start, end) for _word, start, end in words]
    return {
        "url": url,
        "text": text,
        "words": [word for word, _start, _end in words],
        "offsets": offsets,
        "tags": spans_to_bio(spans, offsets),
        "uncertain": covered_words(uncertain, offsets),
        "spans": [{"start": s["start"], "end": s["end"]} for s in spans],
    }


def silver_record(url: str, text: str, mentions: list[dict], ingredients) -> dict:
    """Запись по находкам словаря: оставленное фильтром — метки, отброшенное — dropped"""
    kept, dropped = split_by_recipe(mentions, ingredients)
    record = bio_record(url, text, kept, dropped)
    record["dropped"] = [{"start": s["start"], "end": s["end"]} for s in dropped]
    return record


def gold_record(annotated: dict) -> dict:
    """Запись по ручной разметке, все статусы"""
    return bio_record(annotated["url"], annotated["text"], annotated["mentions"])


def validate_gold(record: dict) -> list[tuple[int, str]]:
    """Механические ошибки разметки: bounds, text, whitespace, order, overlap"""
    text = record["text"]
    problems = []
    prev = None
    for i, mention in enumerate(record["mentions"]):
        start, end = mention["start"], mention["end"]
        if not 0 <= start < end <= len(text):
            problems.append((i, "bounds"))
        else:
            piece = text[start:end]
            if piece != mention["text"]:
                problems.append((i, "text"))
            if piece != piece.strip():
                problems.append((i, "whitespace"))
        if prev is not None:
            if start < prev["start"]:
                problems.append((i, "order"))
            elif start < prev["end"]:
                problems.append((i, "overlap"))
        prev = mention
    return problems


def shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    """n-граммы слов в нижнем регистре"""
    words = [word.lower() for word, _start, _end in split_words(text) if word.isalnum()]
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def near_duplicates(queries: dict[str, str], pool: dict[str, str], threshold: float = 0.5,
                    n: int = 5) -> list[tuple[str, str, float]]:
    """Пары (запрос, текст пула, Жаккар по n-граммам) не ниже порога"""
    index = {}
    pool_shingles = {}
    for key, text in pool.items():
        pool_shingles[key] = shingles(text, n)
        for gram in pool_shingles[key]:
            index.setdefault(gram, []).append(key)
    result = []
    for q_key, text in queries.items():
        q = shingles(text, n)
        shared = {}
        for gram in q:
            for key in index.get(gram, ()):
                shared[key] = shared.get(key, 0) + 1
        for key, common in shared.items():
            jaccard = common / (len(q) + len(pool_shingles[key]) - common)
            if jaccard >= threshold:
                result.append((q_key, key, jaccard))
    return sorted(result, key=lambda item: -item[2])
