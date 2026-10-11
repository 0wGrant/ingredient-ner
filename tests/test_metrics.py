import random

import pytest

from ingredient_ner.metrics import bootstrap_f1, evaluate, match_spans, paired_bootstrap, permutation_test, prf


def span(start, end, **extra):
    return {"start": start, "end": end, **extra}


def pairs(matched):
    return [((g["start"], g["end"]), (p["start"], p["end"])) for g, p in matched]


def positions(spans):
    return [(s["start"], s["end"]) for s in spans]


class TestPrf:
    def test_basic(self):
        result = prf(8, 2, 4)
        assert result["precision"] == pytest.approx(0.8)
        assert result["recall"] == pytest.approx(8 / 12)
        assert result["f1"] == pytest.approx(16 / 22)

    def test_f1_is_harmonic_mean(self):
        assert prf(1, 0, 9)["f1"] == pytest.approx(2 * 1.0 * 0.1 / 1.1)

    def test_perfect(self):
        assert prf(5, 0, 0) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    @pytest.mark.parametrize("counts", [(0, 0, 5), (0, 3, 0), (0, 0, 0)])
    def test_zero_division(self, counts):
        assert prf(*counts) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


class TestMatchSpans:
    def test_exact(self):
        result = match_spans([span(0, 5), span(10, 15)], [span(0, 5), span(11, 15)])
        assert pairs(result["tp"]) == [((0, 5), (0, 5))]
        assert positions(result["fp"]) == [(11, 15)]
        assert positions(result["fn"]) == [(10, 15)]

    def test_both_boundaries_must_match(self):
        result = match_spans([span(0, 5), span(10, 15)], [span(0, 4), span(10, 16)])
        assert result["tp"] == []
        assert positions(result["fp"]) == [(0, 4), (10, 16)]

    def test_duplicate_prediction_matches_once(self):
        result = match_spans([span(0, 5)], [span(0, 5), span(0, 5)])
        assert pairs(result["tp"]) == [((0, 5), (0, 5))]
        assert positions(result["fp"]) == [(0, 5)]

    def test_unsorted_input(self):
        gold = [span(10, 15), span(0, 5)]
        pred = [span(20, 25), span(0, 5), span(10, 15), span(16, 18)]
        result = match_spans(gold, pred)
        assert pairs(result["tp"]) == [((0, 5), (0, 5)), ((10, 15), (10, 15))]
        assert positions(result["fp"]) == [(16, 18), (20, 25)]

    def test_returns_original_objects(self):
        gold = [span(0, 5, ingredient="Соль")]
        pred = [span(0, 5, candidates=["Соль"])]
        [(g, p)] = match_spans(gold, pred)["tp"]
        assert g is gold[0] and p is pred[0]

    def test_empty(self):
        assert match_spans([], []) == {"tp": [], "fp": [], "fn": []}


class TestEvaluate:
    def test_micro_not_macro(self):
        gold = {"a": [span(0, 5)], "b": [span(0, 5), span(10, 15), span(20, 25)]}
        pred = {"a": [span(0, 5)], "b": []}
        result = evaluate(gold, pred)
        assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 3)
        assert result["f1"] == pytest.approx(0.4)

    def test_predictions_for_unknown_document_ignored(self):
        result = evaluate({"a": [span(0, 5)]}, {"a": [span(0, 5)], "zzz": [span(0, 3)]})
        assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)


def mixed_documents():
    gold, pred = {}, {}
    for i in range(10):
        gold[f"good{i}"], pred[f"good{i}"] = [span(0, 5)], [span(0, 5)]
        gold[f"bad{i}"], pred[f"bad{i}"] = [span(0, 5)], [span(10, 15)]
    return gold, pred


class TestBootstrap:
    def test_interval_contains_estimate(self):
        gold, pred = mixed_documents()
        low, high = bootstrap_f1(gold, pred, seed=0)
        assert low < 0.5 < high

    def test_same_seed_same_interval(self):
        gold, pred = mixed_documents()
        assert bootstrap_f1(gold, pred, seed=7) == bootstrap_f1(gold, pred, seed=7)

    def test_global_random_untouched(self):
        gold, pred = mixed_documents()
        state = random.getstate()
        bootstrap_f1(gold, pred, seed=424242)
        assert random.getstate() == state

    def test_identical_documents_give_point_interval(self):
        gold = {f"d{i}": [span(0, 5), span(10, 15)] for i in range(6)}
        pred = {f"d{i}": [span(0, 5), span(20, 25)] for i in range(6)}
        assert bootstrap_f1(gold, pred, seed=2) == pytest.approx((0.5, 0.5))

    def test_lower_level_gives_narrower_interval(self):
        gold, pred = mixed_documents()
        low95, high95 = bootstrap_f1(gold, pred, seed=0, level=0.95)
        low50, high50 = bootstrap_f1(gold, pred, seed=0, level=0.5)
        assert low95 < low50 <= high50 < high95

    def test_empty_gold(self):
        with pytest.raises(ValueError):
            bootstrap_f1({}, {})


def f1(tp, fp, fn):
    return prf(tp, fp, fn)["f1"]


class TestPermutationTest:
    good = [(1, 0, 0)] * 20
    bad = [(0, 1, 1)] * 20

    def test_same_system_gives_one(self):
        assert permutation_test(self.good, self.good, f1, seed=0) == 1.0

    def test_clear_difference_is_significant(self):
        assert permutation_test(self.bad, self.good, f1, seed=0) < 0.01

    def test_small_difference_is_not(self):
        a = [(1, 0, 0)] * 10 + [(0, 1, 1)] * 10
        b = [(1, 0, 0)] * 11 + [(0, 1, 1)] * 9
        assert permutation_test(a, b, f1, seed=0) > 0.5

    def test_symmetric_and_seeded(self):
        a = [(1, 0, 0)] * 12 + [(0, 1, 1)] * 8
        b = [(1, 0, 0)] * 8 + [(0, 1, 1)] * 12
        assert permutation_test(a, b, f1, seed=3) == permutation_test(b, a, f1, seed=3)
        assert 0.0 < permutation_test(a, b, f1, seed=3) <= 1.0

    def test_score_is_used(self):
        a, b = [(1, 0, 1)] * 10, [(1, 1, 0)] * 10
        assert permutation_test(a, b, lambda tp, fp, fn: tp / (tp + fn), seed=0) < 0.01
        assert permutation_test(a, b, f1, seed=0) == 1.0

    def test_global_random_untouched(self):
        state = random.getstate()
        permutation_test(self.bad, self.good, f1, seed=424242)
        assert random.getstate() == state

    @pytest.mark.parametrize("a, b", [([], []), ([(1, 0, 0)], [])])
    def test_mismatched_or_empty(self, a, b):
        with pytest.raises(ValueError):
            permutation_test(a, b, f1)


class TestPairedBootstrap:
    def test_same_system_gives_zero(self):
        gold, pred = mixed_documents()
        assert paired_bootstrap(gold, pred, pred, seed=0) == (0.0, 0.0)

    def test_better_system_positive(self):
        gold, worse = mixed_documents()
        better = dict(worse) | {f"bad{i}": [span(0, 5)] for i in range(5)}
        low, high = paired_bootstrap(gold, worse, better, seed=0)
        assert 0.0 < low <= high

    def test_order_flips_sign(self):
        gold, worse = mixed_documents()
        better = dict(worse) | {f"bad{i}": [span(0, 5)] for i in range(5)}
        low, high = paired_bootstrap(gold, worse, better, seed=0)
        assert paired_bootstrap(gold, better, worse, seed=0) == pytest.approx((-high, -low))

    def test_empty_gold(self):
        with pytest.raises(ValueError):
            paired_bootstrap({}, {}, {})
