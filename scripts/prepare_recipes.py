import ast

from datasets import load_dataset

from ingredient_ner.jsonl import write_jsonl
from ingredient_ner.paths import PREPARED
from ingredient_ner.recipes import build_ingredient_vocab, clean_recipes, dedupe_recipes, filter_recipes

DATASET = "rogozinushka/povarenok-recipes"
FIELDS = ("url", "name", "ingredients")


def main():
    raw = []
    for recipe in load_dataset(DATASET)["train"]:
        recipe["ingredients"] = ast.literal_eval(recipe["ingredients"]) if recipe["ingredients"] else {}
        raw.append(recipe)

    prepared = filter_recipes(dedupe_recipes(clean_recipes(raw)))
    write_jsonl(PREPARED, ({key: recipe[key] for key in FIELDS} for recipe in prepared))

    vocab = build_ingredient_vocab(prepared)
    once = sum(1 for count in vocab.values() if count == 1)
    print(f"рецептов {len(raw)} → {len(prepared)}; названий {len(vocab)}, одиночек {once / len(vocab):.0%}")


if __name__ == "__main__":
    main()
