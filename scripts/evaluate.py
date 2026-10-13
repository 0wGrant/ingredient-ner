import argparse
import random
from collections import Counter
from datetime import datetime

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ingredient_ner.bio import LABELS
from ingredient_ner.dictionary import lemmatizer, predict_dictionary
from ingredient_ner.jsonl import append_jsonl, read_jsonl
from ingredient_ner.metrics import (bootstrap_f1, document_counts, evaluate, match_spans, paired_bootstrap,
                                   permutation_test, prf)
from ingredient_ner.ner import predict_spans
from ingredient_ner.paths import MODELS, NER
from ingredient_ner.recipes import load_recipes

CONTEXT = 30


def model_predictions(name: str, records: list[dict], batch_size: int) -> dict:
    """Отрезки сохранённой модели"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODELS / name)
    model = AutoModelForTokenClassification.from_pretrained(MODELS / name).to(device)
    if [model.config.id2label[i] for i in range(model.config.num_labels)] != LABELS:
        raise ValueError(f"метки модели {model.config.id2label}, ожидались {LABELS}")
    return predict_spans(model, tokenizer, records, device, batch_size)


def model_errors(gold: dict, pred: dict) -> list[tuple[str, str, dict, dict | None]]:
    """Ошибки: граница (лишнее задевает пропущенное), лишнее, пропуск"""
    errors = []
    for url, spans in gold.items():
        matched = match_spans(spans, pred.get(url, []))
        paired = set()
        for p in matched["fp"]:
            near = next((g for g in matched["fn"] if g["start"] < p["end"] and p["start"] < g["end"]), None)
            errors.append(("граница" if near else "лишнее", url, p, near))
            if near:
                paired.add((near["start"], near["end"]))
        errors += [("пропуск", url, g, None) for g in matched["fn"] if (g["start"], g["end"]) not in paired]
    return errors


def with_context(text: str, span: dict) -> str:
    left = text[max(0, span["start"] - CONTEXT):span["start"]]
    right = text[span["end"]:span["end"] + CONTEXT]
    return f"…{left}[{text[span['start']:span['end']]}]{right}…".replace("\n", " ")


def main():
    parser = argparse.ArgumentParser(description="Словарь, словарь + фильтр и модель на ручной разметке")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--model", default=None, help="папка в data/models")
    parser.add_argument("--errors", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    test_log = MODELS / "test_log.jsonl"
    if args.split == "test" and test_log.exists():
        print(f"test уже оценивался: {[run['model'] for run in read_jsonl(test_log)]}")

    records = read_jsonl(NER / f"{args.split}.jsonl")
    texts = {record["url"]: record["text"] for record in records}
    gold = {record["url"]: record["spans"] for record in records}
    recipes = load_recipes()
    names = {name for recipe in recipes for name in recipe["ingredients"]}
    found, filtered = predict_dictionary(records, recipes, lemmatizer(names))
    systems = {"словарь": found, "словарь + фильтр": filtered}
    if args.model:
        systems["модель"] = model_predictions(args.model, records, args.batch_size)

    print(f"{args.split}: рецептов {len(gold)}, упоминаний {sum(map(len, gold.values()))}\n")
    print(f"{'система':18} {'P':>6} {'R':>6} {'F1':>6}  95% интервал")
    results = {}
    for name, pred in systems.items():
        metrics = evaluate(gold, pred)
        low, high = bootstrap_f1(gold, pred)
        results[name] = metrics | {"interval": [low, high]}
        print(f"{name:18} {metrics['precision']:6.3f} {metrics['recall']:6.3f} {metrics['f1']:6.3f}  "
              f"[{low:.3f}, {high:.3f}]")

    if not args.model:
        return

    print("\nразность F1: 95% интервал — парный бутстреп, p — перестановочный тест")
    urls = sorted(gold)
    model_counts = [document_counts(gold[u], systems["модель"].get(u, [])) for u in urls]
    for name in ("словарь", "словарь + фильтр"):
        low, high = paired_bootstrap(gold, systems[name], systems["модель"])
        counts = [document_counts(gold[u], systems[name].get(u, [])) for u in urls]
        p_value = permutation_test(counts, model_counts, lambda tp, fp, fn: prf(tp, fp, fn)["f1"])
        diff = results["модель"]["f1"] - results[name]["f1"]
        results[name] |= {"diff_interval": [low, high], "p": p_value}
        print(f"  модель − {name:17} {diff:+.3f} [{low:+.3f}, {high:+.3f}]  p = {p_value:.4f}")

    errors = model_errors(gold, systems["модель"])
    print(f"\nошибки модели: {dict(Counter(kind for kind, *_rest in errors))}")
    sample = random.Random(0).sample(errors, min(args.errors, len(errors)))
    for kind, url, span, near in sorted(sample, key=lambda error: error[0]):
        tail = f"  эталон [{texts[url][near['start']:near['end']]}]" if near else ""
        print(f"  {kind:8} {with_context(texts[url], span)}{tail}")

    if args.split == "test":
        append_jsonl(test_log, {"time": datetime.now().isoformat(timespec="seconds"), "model": args.model} | results)


if __name__ == "__main__":
    main()
