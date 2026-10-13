import json
from collections import Counter

from ingredient_ner.dictionary import canonical_names, lemmatizer
from ingredient_ner.jsonl import read_jsonl
from ingredient_ner.paths import CANON, PREPARED


def main():
    freq = Counter(name for recipe in read_jsonl(PREPARED) for name in recipe["ingredients"])
    canon = canonical_names(freq, lemmatizer(freq))
    CANON.write_text(json.dumps(canon, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"названий {len(freq)}, заменяется {len(canon)}")


if __name__ == "__main__":
    main()
