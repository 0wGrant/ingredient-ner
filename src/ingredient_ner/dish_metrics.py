import random

from .metrics import N_SAMPLES

LEVELS = ("may", "contains")


def allergen_levels(mentions: list[dict]) -> dict[str, str]:
    """Аллерген → уровень по упоминаниям блюда"""
    result = {}
    for mention in mentions:
        if mention["status"] not in ("present", "optional", "absent"):
            raise ValueError(f"Неизвестный статус: {mention["status"]}")
        if mention["status"] == "absent":
            continue
        for allergen, level in mention["allergens"].items():
            if mention["status"] == "optional":
                level = "may"
            if allergen not in result or LEVELS.index(level) > LEVELS.index(result[allergen]):
                result[allergen] = level
    return result


def _warned(levels: dict[str, str], level: str) -> set[str]:
    return {allergen for allergen, found in levels.items() if LEVELS.index(found) >= LEVELS.index(level)}


def _dish_counts(gold_levels: dict, pred_levels: dict, level: str) -> tuple[int, int, int]:
    gold, pred = _warned(gold_levels, level), _warned(pred_levels, level)
    return len(gold & pred), len(pred - gold), len(gold - pred)


def level_counts(gold: dict, pred: dict, level: str) -> tuple[int, int, int]:
    """TP, FP, FN по парам «блюдо, аллерген» на уровне may или contains"""
    if level not in LEVELS:
        raise ValueError(f"Неизвестный уровень: {level}")
    tp = fp = fn = 0
    for url, levels in gold.items():
        dish_tp, dish_fp, dish_fn = _dish_counts(levels, pred.get(url, {}), level)
        tp += dish_tp
        fp += dish_fp
        fn += dish_fn
    return tp, fp, fn


def fbeta(tp: int, fp: int, fn: int, beta: float = 2.0) -> dict:
    """Точность, полнота, F-beta"""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "fbeta": f}


def missed_share(gold: dict, pred: dict) -> float:
    """Доля блюд с аллергенами, где хоть один остался без предупреждения"""
    dishes = [url for url, levels in gold.items() if levels]
    if not dishes:
        return 0.0
    return sum(1 for url in dishes if set(gold[url]) - set(pred.get(url, {}))) / len(dishes)


def bootstrap_fbeta(gold: dict, pred: dict, level: str = "may", beta: float = 2.0, n_samples: int = N_SAMPLES,
                    seed: int = 0, ci: float = 0.95) -> tuple[float, float]:
    """Доверительный интервал F-beta, бутстреп по блюдам"""
    if not gold:
        raise ValueError("Пустой эталон")
    if level not in LEVELS:
        raise ValueError(f"Неизвестный уровень: {level}")
    counts = [_dish_counts(levels, pred.get(url, {}), level) for url, levels in gold.items()]
    rng = random.Random(seed)
    scores = []
    for _ in range(n_samples):
        sample = rng.choices(counts, k=len(counts))
        scores.append(fbeta(*map(sum, zip(*sample)), beta)["fbeta"])
    scores.sort()
    return scores[int(n_samples * (1 - ci) / 2)], scores[int(n_samples * (1 + ci) / 2) - 1]
