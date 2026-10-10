import re
from bisect import bisect_right

LABELS = ["O", "B-ING", "I-ING"]
IGNORE = -100

_WORDS = re.compile(r"[^\W\d_]+|\d+|\S")


def split_words(text: str) -> list[tuple[str, int, int]]:
    """Слова с позициями: буквенные куски, числа, одиночные знаки"""
    return [(m.group(), m.start(), m.end()) for m in _WORDS.finditer(text)]


def spans_to_bio(spans: list[dict], offsets: list[tuple[int, int]], label: str = "ING") -> list[str]:
    """BIO-метки слов по отрезкам"""
    tags = ["O"] * len(offsets)
    ends = [end for _start, end in offsets]
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        first = bisect_right(ends, span["start"])
        i = first
        while i < len(offsets) and offsets[i][0] < span["end"]:
            if tags[i] != "O":
                raise ValueError(f"слово {i} попало в два отрезка")
            tags[i] = f"B-{label}" if i == first else f"I-{label}"
            i += 1
        if i == first:
            raise ValueError(f"отрезок {span['start']}:{span['end']} не покрывает ни одного слова")
    return tags


def bio_to_spans(tags: list[str], offsets: list[tuple[int, int]]) -> list[dict]:
    """Отрезки по BIO-меткам слов"""
    if len(tags) != len(offsets):
        raise ValueError(f"меток {len(tags)}, позиций {len(offsets)}")

    spans = []
    current = None
    for tag, (start, end) in zip(tags, offsets):
        if tag == "O":
            current = None
            continue
        prefix, sep, label = tag.partition("-")
        if prefix not in ("B", "I") or not sep or not label:
            raise ValueError(f"неизвестная метка {tag!r}")
        if prefix == "B" or current is None or current["label"] != label:
            current = {"start": start, "end": end, "label": label}
            spans.append(current)
        else:
            current["end"] = end
    return spans


def covered_words(spans: list[dict], offsets: list[tuple[int, int]]) -> list[int]:
    """Номера слов, пересекающихся с отрезками"""
    ends = [end for _start, end in offsets]
    result = set()
    for span in spans:
        i = bisect_right(ends, span["start"])
        while i < len(offsets) and offsets[i][0] < span["end"]:
            result.add(i)
            i += 1
    return sorted(result)


def align_labels(tags: list[str], word_ids: list, uncertain=()) -> list[int]:
    """Номера меток подслов: метка слова — первому подслову, остальным −100"""
    label2id = {label: i for i, label in enumerate(LABELS)}
    uncertain = set(uncertain)
    result = []
    previous = None
    for word_id in word_ids:
        if word_id is None or word_id in uncertain or word_id == previous:
            result.append(IGNORE)
        else:
            result.append(label2id[tags[word_id]])
        previous = word_id
    return result
