from collections import Counter

import pytest

from ingredient_ner.dictionary import (GENERIC, IngredientMatcher, canonical_names, recipe_coverage, split_by_recipe,
                                       tokenize)

LEMMAS = {
    "муку": "мука", "муки": "мука",
    "пшеничную": "пшеничный", "пшеничная": "пшеничный",
    "яйца": "яйцо",
    "куриное": "куриный",
    "маслом": "масло",
    "сливочное": "сливочный",
    "растительное": "растительный",
    "луком": "лук", "лука": "лук",
    "зеленым": "зеленый",
    "порея": "порей",
    "перца": "перец", "черного": "черный", "молотого": "молотый",
    "соевым": "соевый", "соусом": "соус",
}


def lemmatize(word):
    return LEMMAS.get(word, word)


NAMES = [
    "Мука пшеничная", "Яйцо куриное", "Масло сливочное", "Масло растительное",
    "Лук репчатый", "Лук зеленый", "Лук-порей", "Перец черный молотый", "Соль",
    "Соевый соус",
]


@pytest.fixture
def matcher():
    return IngredientMatcher(NAMES, lemmatize)


def texts(mentions):
    return [m["text"] for m in mentions]


class TestTokenize:
    def test_tokens_with_offsets(self):
        assert tokenize("Добавить Муку и яйца") == [("добавить", 0, 8), ("муку", 9, 13), ("и", 14, 15),
                                                    ("яйца", 16, 20)]

    def test_lowercase_and_yo(self):
        assert tokenize("Чёрный Перец") == [("черный", 0, 6), ("перец", 7, 12)]
        assert tokenize("Ёжики") == [("ежики", 0, 5)]

    def test_digits_and_punctuation_are_not_tokens(self):
        assert tokenize("2 яйца, 150г муки!") == [("яйца", 2, 6), ("г", 11, 12), ("муки", 13, 17)]

    def test_hyphen_splits_words(self):
        assert tokenize("лук-порей") == [("лук", 0, 3), ("порей", 4, 9)]

    def test_no_letters(self):
        assert tokenize("123 ...") == []


class TestMatcherFind:
    def test_mention_structure(self, matcher):
        assert matcher.find("Посыпать: соль") == [{"start": 10, "end": 14, "text": "соль", "candidates": ["Соль"]}]

    def test_inflected_form_found_by_lemma(self, matcher):
        assert texts(matcher.find("Всыпать муку")) == ["муку"]

    def test_full_name_in_natural_word_order(self, matcher):
        [mention] = matcher.find("Добавить пшеничную муку")
        assert mention["text"] == "пшеничную муку"
        assert mention["candidates"] == ["Мука пшеничная"]

    def test_head_word_is_ambiguous(self, matcher):
        [mention] = matcher.find("Растопить масло")
        assert mention["candidates"] == ["Масло растительное", "Масло сливочное"]

    def test_candidates_sorted_many(self):
        names = ["Сыр твердый", "Сыр мягкий", "Сыр плавленый", "Сыр копченый", "Сыр", "Сыр творожный"]
        [mention] = IngredientMatcher(names, lemmatize).find("Натереть сыр")
        assert mention["candidates"] == sorted(names)

    def test_longest_match_wins(self, matcher):
        [mention] = matcher.find("Сверху лук зеленый")
        assert mention["candidates"] == ["Лук зеленый"]

    def test_three_word_name(self, matcher):
        assert texts(matcher.find("Щепотка черного молотого перца")) == ["черного молотого перца"]

    def test_matches_do_not_overlap(self, matcher):
        assert texts(matcher.find("Растопить сливочное масло")) == ["сливочное масло"]

    def test_hyphenated_name(self, matcher):
        assert texts(matcher.find("Добавить лука-порея")) == ["лука-порея"]

    def test_mention_text_is_original(self, matcher):
        [mention] = matcher.find("Добавить Чёрный Молотый Перец")
        assert (mention["start"], mention["end"], mention["text"]) == (9, 29, "Чёрный Молотый Перец")

    def test_several_mentions_in_text_order(self, matcher):
        mentions = matcher.find("Смешать муку, яйца и соль")
        assert [m["start"] for m in mentions] == [8, 14, 21]

    def test_word_inside_other_word_is_not_found(self, matcher):
        assert matcher.find("Отварить фасоль") == []

    def test_empty_vocabulary(self):
        assert IngredientMatcher([], lemmatize).find("мука и соль") == []

    def test_names_without_letters_are_skipped(self):
        assert texts(IngredientMatcher(["", "123", "Соль"], lemmatize).find("соль")) == ["соль"]

    def test_exact_mode_with_identity(self):
        exact = IngredientMatcher(NAMES, lambda word: word)
        assert exact.find("Всыпать муку") == []
        assert texts(exact.find("Мука просеяна")) == ["Мука"]

    def test_names_are_not_lemmatized_again_on_find(self):
        calls = []

        def counting(word):
            calls.append(word)
            return lemmatize(word)

        matcher = IngredientMatcher(NAMES, counting)
        calls.clear()
        matcher.find("муку и яйца")
        assert len(calls) <= 3


class TestRecipeCoverage:
    @staticmethod
    def recipe(ingredients, steps, description=None):
        return {"ingredients": dict.fromkeys(ingredients, ""), "description": description, "steps": steps}

    def test_found_and_missed(self, matcher):
        recipe = self.recipe(["Мука пшеничная", "Яйцо куриное", "Соль", "Лук зеленый"], ["Смешать муку и яйца."])
        assert recipe_coverage(recipe, matcher) == {"found": ["Мука пшеничная", "Яйцо куриное"],
                                                    "missed": ["Соль", "Лук зеленый"]}

    def test_description_counts(self, matcher):
        recipe = self.recipe(["Лук репчатый"], ["Жарить."], description="Котлеты с луком.")
        assert recipe_coverage(recipe, matcher)["found"] == ["Лук репчатый"]

    def test_head_word_credits_every_candidate(self, matcher):
        assert recipe_coverage(self.recipe(["Масло сливочное"], ["Растопить масло."]), matcher)["found"] == \
               ["Масло сливочное"]


class TestSplitByRecipe:
    def test_split(self):
        mentions = [{"start": 0, "end": 3, "candidates": ["Сахар", "Сахар коричневый"]},
                    {"start": 4, "end": 9, "candidates": ["Тесто дрожжевое"]}]
        kept, dropped = split_by_recipe(mentions, {"Сахар": "1 ст.", "Мука пшеничная": "2 ст."})
        assert kept == [mentions[0]]
        assert dropped == [mentions[1]]

    def test_ingredients_as_list(self):
        mentions = [{"start": 0, "end": 3, "candidates": ["Соль"]}]
        assert split_by_recipe(mentions, ["Соль"]) == (mentions, [])

    def test_generic_name_refers_to_specific(self):
        mentions = [{"start": 0, "end": 4, "candidates": ["Мясо", "Мясо соевое"]}]
        generic = {"Мясо": ["Говядина", "Свинина"]}
        assert split_by_recipe(mentions, ["Свинина", "Лук"], generic) == (mentions, [])
        assert split_by_recipe(mentions, ["Свинина", "Лук"], {}) == ([], mentions)

    def test_specific_does_not_refer_to_generic(self):
        mentions = [{"start": 0, "end": 8, "candidates": ["Свинина"]}]
        assert split_by_recipe(mentions, ["Мясо"], {"Мясо": ["Свинина"]}) == ([], mentions)

    def test_default_table_has_meat(self):
        assert {"Свинина", "Говядина", "Курица"} <= set(GENERIC["Мясо"])


class TestCanonicalNames:
    def test_most_frequent_wins(self):
        freq = Counter({"Масло сливочное": 50, "Сливочное масло": 10, "Соль": 90})
        assert canonical_names(freq, lemmatize) == {"Сливочное масло": "Масло сливочное"}

    def test_tie_broken_by_name(self):
        freq = Counter({"Лук зеленый": 5, "Зеленый лук": 5})
        assert canonical_names(freq, lemmatize) == {"Лук зеленый": "Зеленый лук"}

    def test_different_lemmas_not_merged(self):
        freq = Counter({"Масло сливочное": 5, "Масло растительное": 5})
        assert canonical_names(freq, lemmatize) == {}
