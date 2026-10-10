import random

import pytest

from ingredient_ner.dish_metrics import allergen_levels, bootstrap_fbeta, fbeta, level_counts, missed_share
from ingredient_ner.metrics import prf

C, M = "contains", "may"


def mention(allergens, status="present"):
    return {"status": status, "allergens": allergens}


class TestAllergenLevels:
    def test_present_keeps_levels(self):
        assert allergen_levels([mention({"молоко": C, "соя": M})]) == {"молоко": C, "соя": M}

    def test_optional_caps_to_may(self):
        assert allergen_levels([mention({"молоко": C, "соя": M}, "optional")]) == {"молоко": M, "соя": M}

    def test_absent_skipped(self):
        assert allergen_levels([mention({"орехи": C}, "absent")]) == {}

    @pytest.mark.parametrize("order", [1, -1])
    def test_strongest_level_wins(self, order):
        assert allergen_levels([mention({"молоко": M}), mention({"молоко": C})][::order]) == {"молоко": C}

    def test_optional_does_not_lower_present(self):
        assert allergen_levels([mention({"молоко": C}), mention({"молоко": C}, "optional")]) == {"молоко": C}

    def test_absent_does_not_cancel_other_mentions(self):
        assert allergen_levels([mention({"яйца": C}), mention({"яйца": C}, "absent")]) == {"яйца": C}

    def test_union_over_mentions(self):
        assert allergen_levels([mention({"молоко": C}), mention({"яйца": M}), mention({})]) == {"молоко": C, "яйца": M}

    def test_empty(self):
        assert allergen_levels([]) == {}

    def test_unknown_status(self):
        with pytest.raises(ValueError):
            allergen_levels([mention({"молоко": C}, "maybe")])

    def test_input_not_mutated(self):
        mentions = [mention({"молоко": C}, "optional")]
        allergen_levels(mentions)
        assert mentions == [mention({"молоко": C}, "optional")]


GOLD = {"u": {"молоко": C, "яйца": M}}
PRED = {"u": {"молоко": M, "орехи": C}}


class TestLevelCounts:
    def test_may_level(self):
        assert level_counts(GOLD, PRED, "may") == (1, 1, 1)

    def test_contains_level(self):
        assert level_counts(GOLD, PRED, "contains") == (0, 1, 1)

    def test_contains_counts_as_warning(self):
        gold, pred = {"u": {"яйца": M}}, {"u": {"яйца": C}}
        assert level_counts(gold, pred, "may") == (1, 0, 0)
        assert level_counts(gold, pred, "contains") == (0, 1, 0)

    def test_missing_dish_in_pred(self):
        assert level_counts({"u": {"молоко": C}}, {}, "may") == (0, 0, 1)

    def test_dish_only_in_pred_ignored(self):
        assert level_counts({"u": {"молоко": C}}, {"u": {"молоко": C}, "v": {"орехи": C}}, "may") == (1, 0, 0)

    def test_sums_over_dishes(self):
        gold = {"u": {"молоко": C}, "v": {"яйца": C, "соя": M}}
        pred = {"u": {"молоко": C, "орехи": M}, "v": {"яйца": M}}
        assert level_counts(gold, pred, "may") == (2, 1, 1)

    def test_unknown_level(self):
        with pytest.raises(ValueError):
            level_counts(GOLD, PRED, "maybe")


class TestFbeta:
    def test_beta_one_is_f1(self):
        assert fbeta(8, 2, 4, beta=1)["fbeta"] == pytest.approx(prf(8, 2, 4)["f1"])

    def test_f2_value(self):
        p, r = 0.8, 8 / 12
        assert fbeta(8, 2, 4, beta=2)["fbeta"] == pytest.approx(5 * p * r / (4 * p + r))

    def test_default_beta_is_two(self):
        assert fbeta(8, 2, 4) == fbeta(8, 2, 4, beta=2)

    def test_precision_and_recall(self):
        result = fbeta(8, 2, 4)
        assert set(result) == {"precision", "recall", "fbeta"}
        assert result["precision"] == pytest.approx(0.8)
        assert result["recall"] == pytest.approx(8 / 12)

    def test_f2_favors_recall(self):
        assert fbeta(6, 6, 0)["fbeta"] > fbeta(6, 0, 6)["fbeta"]

    def test_large_beta_approaches_recall(self):
        assert fbeta(6, 6, 2, beta=100)["fbeta"] == pytest.approx(0.75, abs=1e-3)

    @pytest.mark.parametrize("counts", [(0, 0, 5), (0, 3, 0), (0, 0, 0)])
    def test_zero_division(self, counts):
        assert fbeta(*counts) == {"precision": 0.0, "recall": 0.0, "fbeta": 0.0}


class TestMissedShare:
    def test_example(self):
        gold = {"u1": {"молоко": C}, "u2": {"яйца": M}, "u3": {}}
        pred = {"u1": {}, "u2": {"яйца": C}}
        assert missed_share(gold, pred) == pytest.approx(0.5)

    def test_may_warning_is_enough(self):
        assert missed_share({"u": {"молоко": C}}, {"u": {"молоко": M}}) == 0.0

    def test_one_missed_allergen_misses_dish(self):
        assert missed_share({"u": {"молоко": C, "яйца": C}}, {"u": {"молоко": C}}) == 1.0

    def test_missing_dish_in_pred(self):
        assert missed_share({"u": {"молоко": C}}, {}) == 1.0

    def test_extra_allergens_do_not_matter(self):
        assert missed_share({"u": {"молоко": C}}, {"u": {"молоко": C, "орехи": C}}) == 0.0

    def test_no_dishes_with_allergens(self):
        assert missed_share({"u": {}}, {"u": {"орехи": C}}) == 0.0
        assert missed_share({}, {}) == 0.0


def mixed_dishes():
    gold, pred = {}, {}
    for i in range(10):
        gold[f"good{i}"], pred[f"good{i}"] = {"молоко": C}, {"молоко": C}
        gold[f"bad{i}"], pred[f"bad{i}"] = {"молоко": C}, {}
    return gold, pred


class TestBootstrapFbeta:
    def test_interval_contains_estimate(self):
        gold, pred = mixed_dishes()
        estimate = fbeta(*level_counts(gold, pred, "may"))["fbeta"]
        low, high = bootstrap_fbeta(gold, pred, seed=0)
        assert low < estimate < high

    def test_identical_dishes_give_point_interval(self):
        gold = {f"d{i}": {"молоко": C, "яйца": C} for i in range(6)}
        pred = {f"d{i}": {"молоко": C, "орехи": C} for i in range(6)}
        assert bootstrap_fbeta(gold, pred, seed=1) == pytest.approx((0.5, 0.5))

    def test_level_is_used(self):
        gold = {f"d{i}": {"молоко": C} for i in range(5)}
        pred = {f"d{i}": {"молоко": M} for i in range(5)}
        assert bootstrap_fbeta(gold, pred, level="may", seed=1) == (1.0, 1.0)
        assert bootstrap_fbeta(gold, pred, level="contains", seed=1) == (0.0, 0.0)

    def test_beta_is_used(self):
        gold = {f"d{i}": {"молоко": C, "яйца": C} for i in range(5)}
        pred = {f"d{i}": {"молоко": C} for i in range(5)}
        assert bootstrap_fbeta(gold, pred, beta=1, seed=1) == pytest.approx((2 / 3, 2 / 3))
        assert bootstrap_fbeta(gold, pred, beta=2, seed=1) == pytest.approx((5 / 9, 5 / 9))

    def test_same_seed_same_interval(self):
        gold, pred = mixed_dishes()
        assert bootstrap_fbeta(gold, pred, seed=7) == bootstrap_fbeta(gold, pred, seed=7)

    def test_global_random_untouched(self):
        gold, pred = mixed_dishes()
        state = random.getstate()
        bootstrap_fbeta(gold, pred, seed=424242)
        assert random.getstate() == state

    def test_lower_ci_gives_narrower_interval(self):
        gold, pred = mixed_dishes()
        low95, high95 = bootstrap_fbeta(gold, pred, seed=0, ci=0.95)
        low50, high50 = bootstrap_fbeta(gold, pred, seed=0, ci=0.5)
        assert low95 < low50 <= high50 < high95

    def test_empty_gold(self):
        with pytest.raises(ValueError):
            bootstrap_fbeta({}, {})
