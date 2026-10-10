from functools import partial

import torch
from torch.utils.data import DataLoader

from .bio import IGNORE, LABELS, align_labels, bio_to_spans, spans_to_bio, split_words

MODEL_NAME = "cointegrated/rubert-tiny2"
EVAL_MAX_LENGTH = 2048
VARIANTS = ("A", "B", "C")


def word_labels(record: dict, variant: str) -> tuple[list[str], list[int]]:
    """BIO-метки слов и номера слов вне потери для варианта A, B или C"""
    if variant not in VARIANTS:
        raise ValueError(f"Нет такого варианта: {variant}")
    if variant == "A":
        return spans_to_bio(record["spans"] + record.get("dropped", []), record["offsets"]), []
    if variant == "B":
        return list(record["tags"]), []
    return list(record["tags"]), list(record["uncertain"])


def encode_record(record: dict, tokenizer, variant: str = "B", max_length: int = 512) -> dict:
    """input_ids, attention_mask, labels, word_ids"""
    tags, uncertain = word_labels(record, variant)
    enc = tokenizer(record["words"], is_split_into_words=True, truncation=True, max_length=max_length)
    word_ids = enc.word_ids()
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": align_labels(tags, word_ids, uncertain),
        "word_ids": word_ids,
    }


def collate(features: list[dict], pad_id: int) -> dict:
    """Пачка тензоров, добивка справа"""
    max_len = max(len(feature["input_ids"]) for feature in features)
    input_ids, attention_mask, labels = [], [], []
    for feature in features:
        pad_len = max_len - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(feature["attention_mask"] + [0] * pad_len)
        labels.append(feature["labels"] + [IGNORE] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def train_epoch(model: torch.nn.Module, loader, optimizer, scheduler, device, max_grad_norm: float = 1.0) -> float:
    """Эпоха обучения, средняя потеря по пачкам"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def predict(model: torch.nn.Module, loader, device) -> list[list[int]]:
    """Номера меток подслов без добивки"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items() if key in ("input_ids", "attention_mask")}
            predicted = model(**batch).logits.argmax(dim=-1).tolist()
            lengths = batch["attention_mask"].sum(dim=-1).tolist()
            predictions.extend(row[:length] for row, length in zip(predicted, lengths))
    return predictions


def decode_words(pred_ids: list[int], word_ids: list, n_words: int) -> list[str]:
    """Метки слов по первым подсловам; обрезанные слова — O"""
    result = ["O"] * n_words
    previous = None
    for pred_id, word_id in zip(pred_ids, word_ids):
        if word_id is not None and word_id != previous:
            result[word_id] = LABELS[pred_id]
        previous = word_id
    return result


def predict_spans(model, tokenizer, records: list[dict], device, batch_size: int = 16) -> dict[str, list[dict]]:
    """url → отрезки модели"""
    features = [encode_record(record, tokenizer, max_length=EVAL_MAX_LENGTH) for record in records]
    loader = DataLoader(features, batch_size=batch_size, collate_fn=partial(collate, pad_id=tokenizer.pad_token_id))
    result = {}
    for record, feature, row in zip(records, features, predict(model, loader, device), strict=True):
        tags = decode_words(row, feature["word_ids"], len(record["words"]))
        result[record["url"]] = bio_to_spans(tags, record["offsets"])
    return result


def predict_text(model, tokenizer, text: str, device) -> list[dict]:
    """Отрезки модели в произвольном тексте"""
    words = split_words(text)
    record = {"url": "", "words": [word for word, _start, _end in words],
              "offsets": [(start, end) for _word, start, end in words], "tags": ["O"] * len(words), "uncertain": []}
    return predict_spans(model, tokenizer, [record], device)[""]
