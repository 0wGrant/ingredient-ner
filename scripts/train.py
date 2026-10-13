import argparse
import json
import time
from datetime import datetime
from functools import partial

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ingredient_ner.allergens import Linker, dish_levels, gold_levels, load_table
from ingredient_ner.bio import LABELS
from ingredient_ner.context import LEXICON
from ingredient_ner.dictionary import lemmatizer
from ingredient_ner.dish_metrics import fbeta, level_counts
from ingredient_ner.jsonl import append_jsonl, read_jsonl
from ingredient_ner.metrics import evaluate
from ingredient_ner.ner import MODEL_NAME, VARIANTS, collate, encode_record, predict_spans, train_epoch
from ingredient_ner.paths import DEV_GOLD, MODELS, NER
from ingredient_ner.recipes import load_recipes

WARMUP_SHARE = 0.1
WEIGHT_DECAY = 0.01


def run_name(args) -> str:
    """Папка модели: вариант, зерно, обрезка train"""
    name = f"variant_{args.variant}_seed{args.seed}"
    return name if args.limit is None else f"{name}_limit{args.limit}"


def main():
    parser = argparse.ArgumentParser(description="Дообучение rubert-tiny2 на серебряной разметке")
    parser.add_argument("--variant", choices=VARIANTS, default="B")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="рецептов train, для отладки")
    args = parser.parse_args()

    transformers.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train = read_jsonl(NER / "train.jsonl")[:args.limit]
    dev = read_jsonl(NER / "dev.jsonl")
    features = [encode_record(record, tokenizer, args.variant, args.max_length) for record in train]
    loader = DataLoader(features, batch_size=args.batch_size, shuffle=True,
                        collate_fn=partial(collate, pad_id=tokenizer.pad_token_id))

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_SHARE * total_steps), total_steps)

    gold_spans = {record["url"]: record["spans"] for record in dev}
    names = {name for recipe in load_recipes() for name in recipe["ingredients"]}
    linker = Linker(names, lemmatizer(names))
    table = load_table()
    gold_allergens = gold_levels(read_jsonl(DEV_GOLD), linker, table)
    out_dir = MODELS / run_name(args)
    started = datetime.now().isoformat(timespec="seconds")
    print(f"train {len(train)}, dev {len(dev)}, {device}, шагов {total_steps} → {out_dir.name}")

    best = (-1.0, -1.0)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        loss = train_epoch(model, loader, optimizer, scheduler, device)
        pred = predict_spans(model, tokenizer, dev, device, args.batch_size)
        ner = evaluate(gold_spans, pred)
        allergens = fbeta(*level_counts(gold_allergens, dish_levels(dev, pred, linker, table, LEXICON), "may"))
        seconds = time.perf_counter() - epoch_start

        saved = (allergens["fbeta"], ner["f1"]) > best
        if saved:
            best = (allergens["fbeta"], ner["f1"])
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            summary = {"epoch": epoch, "dev_ner": ner, "dev_allergens": allergens} | vars(args)
            (out_dir / "training.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"эпоха {epoch}  потеря {loss:.4f}  NER P {ner['precision']:.3f} R {ner['recall']:.3f} F1 {ner['f1']:.3f}  "
              f"аллергены P {allergens['precision']:.3f} R {allergens['recall']:.3f} F2 {allergens['fbeta']:.3f}  "
              f"{seconds:.0f} с{'  *' if saved else ''}")
        append_jsonl(MODELS / "log.jsonl", {
            "started": started, "epoch": epoch, "loss": loss, "precision": ner["precision"], "recall": ner["recall"],
            "f1": ner["f1"], "allergen_precision": allergens["precision"], "allergen_recall": allergens["recall"],
            "f2": allergens["fbeta"], "seconds": round(seconds, 1), "saved": saved,
        } | vars(args))


if __name__ == "__main__":
    main()
