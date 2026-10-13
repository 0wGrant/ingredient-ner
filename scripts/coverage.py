from ingredient_ner.dictionary import IngredientMatcher, lemmatizer, recipe_coverage
from ingredient_ner.recipes import load_recipes


def share_found(recipes: list[dict], matcher: IngredientMatcher) -> float:
    """Доля ингредиентов состава, найденных в тексте рецепта"""
    total = sum(len(recipe["ingredients"]) for recipe in recipes)
    found = sum(len(recipe_coverage(recipe, matcher)["found"]) for recipe in recipes)
    return found / total


def main():
    recipes = load_recipes()
    names = {name for recipe in recipes for name in recipe["ingredients"]}
    exact = share_found(recipes, IngredientMatcher(names, lambda word: word))
    lemmas = share_found(recipes, IngredientMatcher(names, lemmatizer(names)))
    print(f"рецептов {len(recipes)}, названий {len(names)}; найдено как есть {exact:.1%}, по леммам {lemmas:.1%}")


if __name__ == "__main__":
    main()
