import argparse
import random
from collections import Counter
from datetime import datetime

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ingredient_ner.allergens import Linker, dish_levels, gold_levels, load_table
from ingredient_ner.context import LEXICON, mention_statuses
from ingredient_ner.dictionary import lemmatizer, predict_dictionary
from ingredient_ner.dish_metrics import bootstrap_fbeta, fbeta, level_counts, missed_share
from ingredient_ner.jsonl import append_jsonl, read_jsonl
from ingredient_ner.metrics import N_SAMPLES
from ingredient_ner.ner import predict_spans
from ingredient_ner.paths import DEV_GOLD, MODELS, NER, TEST_GOLD
from ingredient_ner.recipes import load_recipes

STATUSES = ("present", "optional", "absent")


def model_spans(name: str, records: list[dict]) -> dict:
    """Отрезки сохранённой модели"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODELS / name)
    model = AutoModelForTokenClassification.from_pretrained(MODELS / name).to(device)
    return predict_spans(model, tokenizer, records, device)


def paired_bootstrap_fbeta(gold: dict, pred_a: dict, pred_b: dict, level: str = "may", beta: float = 2.0,
                           n_samples: int = N_SAMPLES, seed: int = 0) -> tuple[float, float]:
    """Интервал F-beta(b) − F-beta(a), бутстреп по блюдам"""
    urls = sorted(gold)
    counts_a = [level_counts({u: gold[u]}, {u: pred_a.get(u, {})}, level) for u in urls]
    counts_b = [level_counts({u: gold[u]}, {u: pred_b.get(u, {})}, level) for u in urls]
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_samples):
        idx = rng.choices(range(len(urls)), k=len(urls))
        f_a = fbeta(*map(sum, zip(*(counts_a[i] for i in idx))), beta)["fbeta"]
        f_b = fbeta(*map(sum, zip(*(counts_b[i] for i in idx))), beta)["fbeta"]
        diffs.append(f_b - f_a)
    diffs.sort()
    return diffs[int(n_samples * 0.025)], diffs[int(n_samples * 0.975) - 1]


def report_statuses(annotated: list[dict]) -> None:
    confusion = Counter()
    for record in annotated:
        predicted = mention_statuses(record["text"], record["mentions"], LEXICON)
        confusion.update(zip((m["status"] for m in record["mentions"]), predicted))
    print("статусы на эталонных отрезках (строка — эталон, столбец — ответ):")
    print(f"{'':10}" + "".join(f"{s:>10}" for s in STATUSES) + f"{'P':>8}{'R':>8}")
    for gold_status in STATUSES:
        tp = confusion[gold_status, gold_status]
        predicted = sum(confusion[g, gold_status] for g in STATUSES)
        actual = sum(confusion[gold_status, p] for p in STATUSES)
        row = "".join(f"{confusion[gold_status, p]:>10}" for p in STATUSES)
        print(f"{gold_status:10}{row}{tp / predicted if predicted else 0:8.3f}{tp / actual if actual else 0:8.3f}")


def report_linking(annotated: list[dict], linker: Linker) -> None:
    total = lenient = strict = 0
    for record in annotated:
        for mention in record["mentions"]:
            if mention["ingredient"] is None:
                continue
            groups = linker.link(mention["text"])
            candidates = {name for group in groups for name in group}
            total += 1
            lenient += mention["ingredient"] in candidates
            strict += groups == [[mention["ingredient"]]]
    print(f"\nсвязывание на {total} эталонных упоминаниях со ссылкой: среди кандидатов {lenient / total:.1%}, "
          f"единственный кандидат {strict / total:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Аллергены блюд: статусы, связывание, системы")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--models", nargs="*", default=[], help="папки в data/models")
    args = parser.parse_args()

    test_log = MODELS / "test_log.jsonl"
    if args.split == "test" and test_log.exists():
        print(f"test уже оценивался: {[run['model'] for run in read_jsonl(test_log)]}")

    records = read_jsonl(NER / f"{args.split}.jsonl")
    annotated = read_jsonl(DEV_GOLD if args.split == "dev" else TEST_GOLD)
    lemmatize = lemmatizer()
    recipes = load_recipes()
    linker = Linker({name for recipe in recipes for name in recipe["ingredients"]}, lemmatize)
    table = load_table()

    report_statuses(annotated)
    report_linking(annotated, linker)

    gold = gold_levels(annotated, linker, table)
    found, filtered = predict_dictionary(records, recipes, lemmatize)
    gold_spans = {record["url"]: record["mentions"] for record in annotated}
    systems = {
        "эталонные отрезки": dish_levels(records, gold_spans, linker, table, LEXICON),
        "словарь": dish_levels(records, found, linker, table, LEXICON),
        "словарь + фильтр": dish_levels(records, filtered, linker, table, LEXICON),
    }
    for name in args.models:
        systems[name] = dish_levels(records, model_spans(name, records), linker, table, LEXICON)

    with_allergens = sum(1 for levels in gold.values() if levels)
    pairs = sum(len(levels) for levels in gold.values())
    print(f"\n{args.split}: блюд {len(gold)}, с аллергенами {with_allergens}, пар «блюдо, аллерген» {pairs}")
    print(f"{'система':22} {'P':>6} {'R':>6} {'F2':>6}  {'95% интервал':15}  {'contains: P':>11} {'R':>6} {'F2':>6}  "
          f"{'пропуск блюд':>12}")
    results = {}
    for name, pred in systems.items():
        warn = fbeta(*level_counts(gold, pred, "may"))
        strong = fbeta(*level_counts(gold, pred, "contains"))
        low, high = bootstrap_fbeta(gold, pred, "may")
        missed = missed_share(gold, pred)
        results[name] = {"may": warn, "contains": strong, "interval": [low, high], "missed_share": missed}
        print(f"{name:22} {warn['precision']:6.3f} {warn['recall']:6.3f} {warn['fbeta']:6.3f}  [{low:.3f}, {high:.3f}]  "
              f"{strong['precision']:11.3f} {strong['recall']:6.3f} {strong['fbeta']:6.3f}  {missed:12.1%}")

    if args.models:
        print("\nразность F2 (предупреждения), 95% парный интервал:")
        baselines = ["словарь", "словарь + фильтр"] + args.models
        for model in args.models:
            for base in baselines[:baselines.index(model)]:
                low, high = paired_bootstrap_fbeta(gold, systems[base], systems[model])
                diff = results[model]["may"]["fbeta"] - results[base]["may"]["fbeta"]
                print(f"  {model} − {base}: {diff:+.3f} [{low:+.3f}, {high:+.3f}]")

    if args.split == "test" and args.models:
        append_jsonl(test_log, {"time": datetime.now().isoformat(timespec="seconds"), "model": args.models,
                                "kind": "allergens"} | results)


if __name__ == "__main__":
    main()
