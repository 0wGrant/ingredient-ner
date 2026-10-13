import json
from collections import Counter

from ingredient_ner.recipes import (build_ingredient_vocab, canonicalize_recipes, clean_ingredient_name, clean_recipes,
                                    dedupe_recipes, filter_recipes, load_recipes, normalize_text, recipe_text)


def recipe(name, ingredients, text=""):
    return {"name": name, "ingredients": ingredients, "text": text}


def cleaned_recipe(name, cleaned_ingredients, text="", raw_ingredients=None):
    return {
        "name": name,
        "ingredients": {} if raw_ingredients is None else raw_ingredients,
        "text": text,
        "cleaned_ingredients": list(cleaned_ingredients),
    }


class TestNormalizeText:
    def test_all_together(self):
        assert normalize_text("  Яйцо   Куриное  ") == "яйцо куриное"

    def test_tabs_and_newlines(self):
        assert normalize_text("яйцо\tкуриное\nсвежее") == "яйцо куриное свежее"

    def test_empty(self):
        assert normalize_text("") == ""

    def test_none(self):
        assert normalize_text(None) == ""

    def test_only_spaces(self):
        assert normalize_text("   ") == ""


class TestCleanIngredientName:
    def test_basic_normalize(self):
        assert clean_ingredient_name("Яйцо куриное") == "яйцо куриное"

    def test_removes_parentheses(self):
        assert clean_ingredient_name("Сыр твердый (Российский)") == "сыр твердый"

    def test_parentheses_in_middle(self):
        assert clean_ingredient_name("Мука (высший сорт) пшеничная") == "мука пшеничная"

    def test_parentheses_without_spaces_do_not_glue_words(self):
        assert clean_ingredient_name("Мука(высший сорт)пшеничная") == "мука пшеничная"

    def test_none(self):
        assert clean_ingredient_name(None) == ""


class TestCleanRecipes:
    def test_adds_cleaned_ingredients(self):
        r = recipe("а", {"Яйцо куриное": "2 шт", "Соль": "1 щепот."})
        assert clean_recipes([r])[0]["cleaned_ingredients"] == ["яйцо куриное", "соль"]

    def test_drops_names_empty_after_cleaning(self):
        r = recipe("а", {"": "1", "(по вкусу)": "", "Соль": "1"})
        assert clean_recipes([r])[0]["cleaned_ingredients"] == ["соль"]

    def test_collapses_duplicates_keeping_first_order(self):
        r = recipe("а", {"Мука": "200 г", "Соль": "1 щепот.", "Яйцо": "2 шт", "Соль (морская)": "по вкусу",
                         "мука ": "50 г", "Масло": "10 г", "Сахар": "1 ст. л."})
        assert clean_recipes([r])[0]["cleaned_ingredients"] == ["мука", "соль", "яйцо", "масло", "сахар"]

    def test_keeps_other_fields_and_raw_ingredients(self):
        r = recipe("Омлет", {"Соль (морская)": "1 щепот.", "": "1"}, text="Взбить яйца")
        result = clean_recipes([r])[0]
        assert result["name"] == "Омлет"
        assert result["text"] == "Взбить яйца"
        assert result["ingredients"] == {"Соль (морская)": "1 щепот.", "": "1"}

    def test_does_not_mutate_input(self):
        r = recipe("а", {"Соль (морская)": "1"})
        result = clean_recipes([r])
        assert r == recipe("а", {"Соль (морская)": "1"})
        assert result[0] is not r

    def test_empty(self):
        assert clean_recipes([]) == []


class TestDedupeRecipes:
    def test_same_name_same_ingredients_is_duplicate(self):
        a = cleaned_recipe("Омлет", ["яйцо", "соль"])
        b = cleaned_recipe("Омлет", ["яйцо", "соль"])
        assert dedupe_recipes([a, b]) == [a]

    def test_normalizes_name_before_comparing(self):
        a = cleaned_recipe("Омлет", ["яйцо"])
        b = cleaned_recipe("  ОМЛЕТ  ", ["яйцо"])
        assert dedupe_recipes([a, b]) == [a]

    def test_ingredient_order_does_not_matter(self):
        a = cleaned_recipe("Борщ", ["свекла", "капуста", "говядина"])
        b = cleaned_recipe("Борщ", ["говядина", "свекла", "капуста"])
        assert dedupe_recipes([a, b]) == [a]

    def test_same_name_different_ingredients_kept(self):
        a = cleaned_recipe("Борщ", ["свекла", "капуста"])
        b = cleaned_recipe("Борщ", ["свекла", "говядина"])
        assert dedupe_recipes([a, b]) == [a, b]

    def test_keeps_first_occurrence_and_order(self):
        a = cleaned_recipe("а", ["x"])
        b = cleaned_recipe("б", ["x"])
        c = cleaned_recipe("а", ["x"], text="дубль")
        result = dedupe_recipes([a, b, c])
        assert result == [a, b]
        assert result[0] is a


class TestFilterRecipes:
    def test_filters_too_few(self):
        assert filter_recipes([cleaned_recipe("мало", ["соль"])], min_ingredients=2) == []

    def test_filters_too_many(self):
        many = [f"ing{i}" for i in range(40)]
        assert filter_recipes([cleaned_recipe("много", many)], max_ingredients=30) == []

    def test_boundaries_inclusive(self):
        two = cleaned_recipe("два", ["а", "б"])
        assert filter_recipes([two], min_ingredients=2, max_ingredients=2) == [two]

    def test_counts_cleaned_ingredients_not_raw(self):
        r = cleaned_recipe("очищенный", ["соль"], raw_ingredients={"": "1", "Соль (морская)": "2", "Соль": "3"})
        assert filter_recipes([r], min_ingredients=2) == []


class TestBuildIngredientVocab:
    def test_counts_across_recipes(self):
        vocab = build_ingredient_vocab([cleaned_recipe("а", ["соль", "яйцо куриное"]),
                                        cleaned_recipe("б", ["соль", "мука"])])
        assert vocab == Counter({"соль": 2, "яйцо куриное": 1, "мука": 1})

    def test_empty_input(self):
        assert build_ingredient_vocab([]) == Counter()


class TestPipeline:
    def test_full_chain(self):
        recipes = [
            recipe("Омлет", {"Яйцо куриное": "2 шт", "Соль": "1 щепот."}),
            recipe("ОМЛЕТ ", {"Соль": "по вкусу", "Яйцо куриное (С0)": "2 шт"}),
            recipe("Омлет", {"Яйцо куриное": "3 шт", "Молоко": "50 мл"}),
            recipe("Соль в банке", {"Соль": "1 кг"}),
            recipe("Блины", {"Мука": "200 г", "Яйцо куриное": "1 шт", "Молоко": "300 мл"}),
        ]
        prepared = filter_recipes(dedupe_recipes(clean_recipes(recipes)), min_ingredients=2)
        assert [r["name"] for r in prepared] == ["Омлет", "Омлет", "Блины"]
        assert build_ingredient_vocab(prepared)["яйцо куриное"] == 3


class TestCanonicalizeRecipes:
    CANON = {"Сливочное масло": "Масло сливочное", "бульон": "Бульон"}

    def test_names_replaced(self):
        [result] = canonicalize_recipes([{"url": "u", "ingredients": {"Сливочное масло": "50 г", "Соль": ""}}],
                                        self.CANON)
        assert result["ingredients"] == {"Масло сливочное": "50 г", "Соль": ""}

    def test_duplicates_merged_first_amount_wins(self):
        [result] = canonicalize_recipes([{"ingredients": {"Масло сливочное": "50 г", "Сливочное масло": "20 г"}}],
                                        self.CANON)
        assert result["ingredients"] == {"Масло сливочное": "50 г"}

    def test_order_kept(self):
        [result] = canonicalize_recipes([{"ingredients": {"Соль": "", "Сливочное масло": "", "Мука": ""}}],
                                        self.CANON)
        assert list(result["ingredients"]) == ["Соль", "Масло сливочное", "Мука"]

    def test_input_not_modified(self):
        recipes = [{"ingredients": {"Сливочное масло": "50 г"}}]
        canonicalize_recipes(recipes, self.CANON)
        assert recipes == [{"ingredients": {"Сливочное масло": "50 г"}}]


class TestLoadRecipes:
    def test_water_is_not_an_ingredient(self, tmp_path):
        path, canon = tmp_path / "recipes.jsonl", tmp_path / "canon.json"
        ingredients = {"Вода": "1 ст.", "Сливочное масло": "50 г", "Вода минеральная": "", "Вода розовая": ""}
        path.write_text(json.dumps({"url": "u", "ingredients": ingredients}, ensure_ascii=False) + "\n", encoding="utf-8")
        canon.write_text(json.dumps({"Сливочное масло": "Масло сливочное"}, ensure_ascii=False), encoding="utf-8")
        [result] = load_recipes(path, canon)
        assert result["ingredients"] == {"Масло сливочное": "50 г", "Вода розовая": ""}


class TestRecipeText:
    def test_description_then_steps(self):
        assert recipe_text({"description": "Вкусно.", "steps": ["Шаг один.", "Шаг два."]}) == \
               "Вкусно.\nШаг один.\nШаг два."

    def test_no_description(self):
        assert recipe_text({"description": None, "steps": ["Шаг один.", "Шаг два."]}) == "Шаг один.\nШаг два."

    def test_empty_description(self):
        assert recipe_text({"description": "", "steps": ["Шаг один."]}) == "Шаг один."
