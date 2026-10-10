import pytest

import ingredient_ner.allergens as allergens
from ingredient_ner.allergens import (ALLERGENS, Linker, candidate_allergens, gold_mentions, link_mentions, load_table,
                                      mention_allergens)

C, M = "contains", "may"

TABLE = {
    "Масло сливочное": {"молоко": C},
    "Масло топленое": {"молоко": C},
    "Майонез": {"яйца": C, "горчица": M},
    "Горчица": {"горчица": C},
}
NAMES = ["Масло растительное", "Масло сливочное", "Масло топленое", "Соль", "Майонез"]
LEMMAS = {"масла": "масло", "сливочного": "сливочный", "сливочное": "сливочный", "растительное": "растительный",
          "топленое": "топленый", "соли": "соль"}


def lemmatize(word):
    return LEMMAS.get(word, word)


@pytest.fixture
def linker():
    return Linker(NAMES, lemmatize)


class TestTable:
    def test_allergens_and_levels_are_known(self):
        table = load_table()
        assert table
        assert {a for levels in table.values() for a in levels} <= set(ALLERGENS)
        assert {level for levels in table.values() for level in levels.values()} <= {C, M}

    def test_examples(self):
        table = load_table()
        assert table["Майонез"]["яйца"] == C
        assert table["Мука пшеничная"] == {"глютен": C}
        assert "Масло растительное" not in table


class TestCandidateAllergens:
    def test_partial_becomes_may(self):
        assert candidate_allergens(["Масло сливочное", "Масло растительное"], TABLE) == {"молоко": M}

    def test_all_candidates_agree(self):
        assert candidate_allergens(["Масло сливочное", "Масло топленое"], TABLE) == {"молоко": C}

    def test_common_allergen_keeps_weakest_level(self):
        assert candidate_allergens(["Майонез", "Горчица"], TABLE) == {"горчица": M, "яйца": M}

    def test_single_candidate(self):
        assert candidate_allergens(["Майонез"], TABLE) == {"яйца": C, "горчица": M}

    def test_empty_and_unknown(self):
        assert candidate_allergens([], TABLE) == {}
        assert candidate_allergens(["Соль"], TABLE) == {}


class TestMentionAllergens:
    def test_strongest_over_groups(self):
        groups = [["Масло сливочное"], ["Масло сливочное", "Масло растительное"]]
        assert mention_allergens(groups, TABLE) == {"молоко": C}

    def test_no_groups(self):
        assert mention_allergens([], TABLE) == {}


class TestLinker:
    def test_full_name(self, linker):
        assert linker.link("сливочное масло") == [["Масло сливочное"]]

    def test_head_word_is_ambiguous(self, linker):
        assert linker.link("масло") == [["Масло растительное", "Масло сливочное", "Масло топленое"]]

    def test_several_names(self, linker):
        assert linker.link("соли и масла") == [["Соль"], ["Масло растительное", "Масло сливочное", "Масло топленое"]]

    def test_unknown_word(self, linker):
        assert linker.link("арбуз") == []


class TestLinkMentions:
    text = "Растопить масло, добавить соль"

    def spans(self):
        return [{"start": 10, "end": 15}, {"start": 26, "end": 30}]

    def test_fields_and_statuses(self, linker):
        result = link_mentions(self.text, self.spans(), ["present", "absent"], linker, TABLE)
        assert [m["text"] for m in result] == ["масло", "соль"]
        assert [m["status"] for m in result] == ["present", "absent"]
        assert result[0]["allergens"] == {"молоко": M}
        assert result[1]["candidates"] == [["Соль"]]

    def test_lengths_must_match(self, linker):
        with pytest.raises(ValueError):
            link_mentions(self.text, self.spans(), ["present"], linker, TABLE)


class TestGoldMentions:
    def test_link_or_text(self, linker):
        annotated = {"mentions": [
            {"text": "майонезом", "ingredient": "Майонез", "status": "optional"},
            {"text": "масло", "ingredient": None, "status": "present"},
        ]}
        assert gold_mentions(annotated, linker, TABLE) == [
            {"status": "optional", "allergens": {"яйца": C, "горчица": M}},
            {"status": "present", "allergens": {"молоко": M}},
        ]


def test_analyze_wires_statuses_and_levels(monkeypatch, linker):
    monkeypatch.setattr(allergens, "mention_statuses", lambda text, spans, lexicon: ["present"] * len(spans))
    monkeypatch.setattr(allergens, "allergen_levels", lambda mentions: {"всего": len(mentions)})
    result = allergens.analyze("сливочное масло", [{"start": 0, "end": 15}], linker, TABLE, [])
    assert result["allergens"] == {"всего": 1}
    assert result["mentions"][0]["allergens"] == {"молоко": C}
