import random

N_SAMPLES = 2000


def prf(tp: int, fp: int, fn: int) -> dict:
    """Точность, полнота, F1"""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def match_spans(gold: list[dict], pred: list[dict]) -> dict:
    """Сопоставление один к одному по точным границам: tp — пары, fp, fn"""
    sorted_gold = sorted(gold, key=lambda s: (s["start"], s["end"]))
    sorted_pred = sorted(pred, key=lambda s: (s["start"], s["end"]))

    result = {"tp": [], "fp": [], "fn": []}
    locked = set()
    for g_span in sorted_gold:
        for i, p_span in enumerate(sorted_pred):
            if i not in locked and (g_span["start"], g_span["end"]) == (p_span["start"], p_span["end"]):
                result["tp"].append((g_span, p_span))
                locked.add(i)
                break
        else:
            result["fn"].append(g_span)

    result["fp"] = [p_span for i, p_span in enumerate(sorted_pred) if i not in locked]
    return result


def document_counts(gold_spans: list[dict], pred_spans: list[dict]) -> tuple[int, int, int]:
    """TP, FP, FN одного документа"""
    matched = match_spans(gold_spans, pred_spans)
    return len(matched["tp"]), len(matched["fp"]), len(matched["fn"])


def evaluate(gold: dict, pred: dict) -> dict:
    """Микро-метрики: url → отрезки"""
    tp = fp = fn = 0
    for url, spans in gold.items():
        doc_tp, doc_fp, doc_fn = document_counts(spans, pred.get(url, []))
        tp += doc_tp
        fp += doc_fp
        fn += doc_fn
    return {"tp": tp, "fp": fp, "fn": fn} | prf(tp, fp, fn)


def _interval(values: list[float], level: float) -> tuple[float, float]:
    values = sorted(values)
    n = len(values)
    return values[int(n * (1 - level) / 2)], values[int(n * (1 + level) / 2) - 1]


def _f1(counts: list[tuple[int, int, int]], idx) -> float:
    return prf(*map(sum, zip(*(counts[i] for i in idx))))["f1"]


def bootstrap_f1(gold: dict, pred: dict, n_samples: int = N_SAMPLES, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """Доверительный интервал F1, бутстреп по документам"""
    if not gold:
        raise ValueError("Пустой эталон")
    counts = [document_counts(spans, pred.get(url, [])) for url, spans in gold.items()]
    rng = random.Random(seed)
    scores = [_f1(counts, rng.choices(range(len(counts)), k=len(counts))) for _ in range(n_samples)]
    return _interval(scores, level)


def permutation_test(counts_a: list[tuple[int, int, int]], counts_b: list[tuple[int, int, int]], score,
                     n_samples: int = N_SAMPLES, seed: int = 0) -> float:
    """p-значение разности метрики двух систем: перестановки систем внутри документов"""
    if len(counts_a) != len(counts_b) or not counts_a:
        raise ValueError("Нужны счётчики двух систем по одним и тем же документам")

    def gap(a, b):
        return abs(score(*map(sum, zip(*a))) - score(*map(sum, zip(*b))))

    observed = gap(counts_a, counts_b)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_samples):
        swapped = [(b, a) if rng.random() < 0.5 else (a, b) for a, b in zip(counts_a, counts_b)]
        hits += gap([a for a, _b in swapped], [b for _a, b in swapped]) >= observed - 1e-12
    return (hits + 1) / (n_samples + 1)


def paired_bootstrap(gold: dict, pred_a: dict, pred_b: dict, n_samples: int = N_SAMPLES, seed: int = 0,
                     level: float = 0.95) -> tuple[float, float]:
    """Интервал F1(b) − F1(a) на общих выборках документов"""
    if not gold:
        raise ValueError("Пустой эталон")
    urls = sorted(gold)
    counts_a = [document_counts(gold[u], pred_a.get(u, [])) for u in urls]
    counts_b = [document_counts(gold[u], pred_b.get(u, [])) for u in urls]
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_samples):
        idx = rng.choices(range(len(urls)), k=len(urls))
        diffs.append(_f1(counts_b, idx) - _f1(counts_a, idx))
    return _interval(diffs, level)
