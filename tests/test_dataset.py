from ingredient_ner.dataset import bio_record, gold_record, near_duplicates, silver_record, validate_gold

TEXT = "Замесить тесто, добавить сахар."


class TestBioRecord:
    def test_fields(self):
        record = bio_record("u", TEXT, [{"start": 25, "end": 30, "ingredient": "Сахар"}], [{"start": 9, "end": 14}])
        assert record["words"] == ["Замесить", "тесто", ",", "добавить", "сахар", "."]
        assert record["tags"] == ["O", "O", "O", "O", "B-ING", "O"]
        assert record["uncertain"] == [1]
        assert record["spans"] == [{"start": 25, "end": 30}]
        assert record["offsets"][4] == (25, 30)


class TestSilverRecord:
    mentions = [{"start": 9, "end": 14, "candidates": ["Тесто слоеное"]},
                {"start": 25, "end": 30, "candidates": ["Сахар"]}]

    def test_filter_decides_labels(self):
        record = silver_record("u", TEXT, self.mentions, {"Сахар": "1 ст."})
        assert record["tags"] == ["O", "O", "O", "O", "B-ING", "O"]
        assert record["uncertain"] == [1]
        assert record["dropped"] == [{"start": 9, "end": 14}]

    def test_nothing_dropped(self):
        record = silver_record("u", TEXT, self.mentions, ["Сахар", "Тесто слоеное"])
        assert record["tags"] == ["O", "B-ING", "O", "O", "B-ING", "O"]
        assert record["dropped"] == []


class TestGoldRecord:
    def test_all_mentions_become_spans(self):
        annotated = {"url": "u", "text": TEXT, "mentions": [
            {"start": 9, "end": 14, "text": "тесто", "status": "absent"},
            {"start": 25, "end": 30, "text": "сахар", "status": "present"}]}
        assert gold_record(annotated)["spans"] == [{"start": 9, "end": 14}, {"start": 25, "end": 30}]


def annotated(text, *mentions):
    return {"url": "u", "text": text, "mentions": [{"start": s, "end": e, "text": t} for s, e, t in mentions]}


class TestValidateGold:
    TEXT = "Растопить сливочное масло, добавить муку."

    def test_clean(self):
        assert validate_gold(annotated(self.TEXT, (10, 25, "сливочное масло"), (36, 40, "муку"))) == []

    def test_bounds(self):
        assert validate_gold(annotated("соль", (0, 9, "соль"))) == [(0, "bounds")]
        assert validate_gold(annotated(self.TEXT, (5, 5, ""))) == [(0, "bounds")]

    def test_text_mismatch(self):
        assert validate_gold(annotated(self.TEXT, (10, 25, "сливочное мясо"))) == [(0, "text")]

    def test_whitespace_on_border(self):
        assert validate_gold(annotated(self.TEXT, (9, 25, " сливочное масло"))) == [(0, "whitespace")]

    def test_punctuation_is_not_whitespace(self):
        assert validate_gold(annotated(self.TEXT, (10, 26, "сливочное масло,"))) == []

    def test_overlap(self):
        assert validate_gold(annotated(self.TEXT, (10, 25, "сливочное масло"), (20, 25, "масло"))) == [(1, "overlap")]

    def test_touching_mentions_do_not_overlap(self):
        assert validate_gold(annotated("лук-порей", (0, 3, "лук"), (3, 9, "-порей"))) == []

    def test_order(self):
        assert validate_gold(annotated(self.TEXT, (36, 40, "муку"), (10, 25, "сливочное масло"))) == [(1, "order")]

    def test_several_problems(self):
        record = annotated(self.TEXT, (10, 25, "сливочное масло"), (20, 26, "масло "), (36, 99, "муку"))
        assert validate_gold(record) == [(1, "text"), (1, "overlap"), (2, "bounds")]


class TestNearDuplicates:
    def test_identical(self):
        text = "раз два три четыре пять шесть семь восемь"
        assert near_duplicates({"q": text}, {"p": text}) == [("q", "p", 1.0)]

    def test_unrelated(self):
        a = "раз два три четыре пять шесть семь восемь"
        b = "один другой третий пятый десятый сотый тысячный"
        assert near_duplicates({"q": a}, {"p": b}) == []

    def test_threshold(self):
        a = "раз два три четыре пять шесть семь восемь девять десять"
        b = "раз два три четыре пять шесть совсем другое окончание текста здесь"
        assert near_duplicates({"q": a}, {"p": b}, threshold=0.9) == []
        assert near_duplicates({"q": a}, {"p": b}, threshold=0.1)[0][:2] == ("q", "p")
