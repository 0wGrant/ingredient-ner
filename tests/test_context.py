import pytest

from ingredient_ner.context import find_triggers, mention_statuses

LEX = [
    {"phrase": "без", "category": "absent", "direction": "forward"},
    {"phrase": "можно без", "category": "absent", "direction": "forward"},
    {"phrase": "не добавля*", "category": "absent", "direction": "backward"},
    {"phrase": "не надо", "category": "absent", "direction": "backward"},
    {"phrase": "не надо жалеть", "category": "pseudo"},
    {"phrase": "по желанию", "category": "optional", "direction": "both"},
    {"phrase": "подава*", "category": "optional", "direction": "forward"},
    {"phrase": "но", "category": "terminate"},
]


def span(text, word):
    start = text.index(word)
    return {"start": start, "end": start + len(word)}


def statuses(text, *words, lexicon=LEX, window=None):
    return mention_statuses(text, [span(text, word) for word in words], lexicon, window)


def positions(triggers):
    return [(t["start"], t["end"], t["category"]) for t in triggers]


class TestFindTriggers:
    def test_fields(self):
        assert find_triggers("Без сахара", LEX) == [{"start": 0, "end": 3, "category": "absent", "direction": "forward"}]

    def test_whole_word_only(self):
        assert find_triggers("Взбить безе", LEX) == []

    def test_star_matches_word_start(self):
        assert positions(find_triggers("Сахар не добавляю", LEX)) == [(6, 17, "absent")]
        assert positions(find_triggers("Сахар не добавлять", LEX)) == [(6, 18, "absent")]

    def test_without_star_word_must_be_equal(self):
        lexicon = [{"phrase": "добавля", "category": "absent", "direction": "backward"}]
        assert find_triggers("Сахар добавляю", lexicon) == []

    def test_case_and_yo(self):
        lexicon = [{"phrase": "уберем", "category": "absent", "direction": "backward"}]
        assert positions(find_triggers("Соль УБЕРЁМ", lexicon)) == [(5, 11, "absent")]

    def test_spaces_between_words_do_not_matter(self):
        assert positions(find_triggers("Масло   не   надо", LEX)) == [(8, 17, "absent")]

    def test_pseudo_removes_overlapping_trigger(self):
        assert find_triggers("Масла не надо жалеть", LEX) == []

    def test_pseudo_keeps_other_triggers(self):
        text = "Масла не надо жалеть, а сахар не надо"
        start = text.rindex("не надо")
        assert positions(find_triggers(text, LEX)) == [(start, start + 7, "absent")]

    def test_longest_match_wins(self):
        assert positions(find_triggers("Можно без сахара", LEX)) == [(0, 9, "absent")]

    @pytest.mark.parametrize("reverse", [False, True])
    def test_equal_length_overlap_keeps_earlier(self, reverse):
        lexicon = [{"phrase": "можно без", "category": "absent", "direction": "forward"},
                   {"phrase": "без масла", "category": "optional", "direction": "forward"}]
        lexicon = lexicon[::-1] if reverse else lexicon
        assert positions(find_triggers("можно без масла", lexicon)) == [(0, 9, "absent")]

    def test_terminate_is_returned(self):
        assert [t["category"] for t in find_triggers("Без сахара, но с мёдом", LEX)] == ["absent", "terminate"]

    def test_punctuation_phrase(self):
        lexicon = [{"phrase": "(", "category": "terminate"}]
        assert positions(find_triggers("соль (крупная)", lexicon)) == [(5, 6, "terminate")]

    def test_sorted_by_start(self):
        triggers = find_triggers("Сахар не добавляю, но по желанию мёд, без соли", LEX)
        assert [t["start"] for t in triggers] == sorted(t["start"] for t in triggers)
        assert [t["category"] for t in triggers] == ["absent", "terminate", "optional", "absent"]

    def test_repeated_phrase(self):
        assert positions(find_triggers("без сахара и без соли", LEX)) == [(0, 3, "absent"), (13, 16, "absent")]

    def test_nothing_found(self):
        assert find_triggers("Смешать муку и сахар", LEX) == []


class TestMentionStatuses:
    def test_present_without_triggers(self):
        assert statuses("Смешать муку и сахар", "муку", "сахар") == ["present", "present"]

    def test_forward_absent(self):
        assert statuses("Готовим без масла", "масла") == ["absent"]

    def test_forward_does_not_look_back(self):
        assert statuses("Сахар всыпать, тесто печь без формы", "Сахар") == ["present"]

    def test_backward_absent(self):
        assert statuses("Соль не добавляю", "Соль") == ["absent"]

    def test_backward_does_not_look_forward(self):
        assert statuses("Не добавляю соль", "соль") == ["present"]

    def test_both_directions(self):
        assert statuses("Изюм по желанию", "Изюм") == ["optional"]
        assert statuses("По желанию изюм", "изюм") == ["optional"]

    @pytest.mark.parametrize("sep", [". ", "! ", "? ", "… ", "\n"])
    def test_sentence_end_stops_forward(self, sep):
        assert statuses(f"Готовим без масла{sep}Сахар добавить", "Сахар") == ["present"]

    @pytest.mark.parametrize("sep", [". ", "\n"])
    def test_sentence_end_stops_backward(self, sep):
        assert statuses(f"Соль положить{sep}Перец не добавляю", "Соль", "Перец") == ["present", "absent"]

    def test_terminate_stops_scope(self):
        assert statuses("Без сахара, но с мёдом", "сахара", "мёдом") == ["absent", "present"]
        assert statuses("Сахар положить, но соль не добавлять", "Сахар", "соль") == ["present", "absent"]

    def test_scope_continues_without_terminate(self):
        assert statuses("Без сахара и мёда", "мёда") == ["absent"]

    def test_comma_as_terminate(self):
        assert statuses("Без сахара, мёд", "мёд") == ["absent"]
        assert statuses("Без сахара, мёд", "мёд", lexicon=LEX + [{"phrase": ",", "category": "terminate"}]) == ["present"]

    def test_absent_beats_optional(self):
        assert statuses("По желанию можно без масла", "масла") == ["absent"]

    def test_optional_forward(self):
        assert statuses("Подавать со сметаной или сливками", "сметаной", "сливками") == ["optional", "optional"]

    def test_pseudo_trigger(self):
        assert statuses("Масла не надо жалеть", "Масла") == ["present"]
        assert statuses("Масло не надо", "Масло") == ["absent"]

    @pytest.mark.parametrize("window, expected", [(None, "absent"), (2, "absent"), (1, "present"), (0, "present")])
    def test_window_forward(self, window, expected):
        assert statuses("Без большого количества масла", "масла", window=window) == [expected]

    @pytest.mark.parametrize("window, expected", [(2, "absent"), (1, "present")])
    def test_window_backward(self, window, expected):
        assert statuses("Соль сегодня точно не добавляю", "Соль", window=window) == [expected]

    def test_trigger_overlapping_span_does_not_count(self):
        assert mention_statuses("без масла", [{"start": 0, "end": 9}], LEX) == ["present"]

    def test_several_spans(self):
        text = "Сахар не добавляю, мёд по желанию, соль положить"
        assert statuses(text, "Сахар", "мёд", "соль") == ["absent", "optional", "optional"]

    def test_extra_keys_in_spans(self):
        text = "Готовим без масла"
        assert mention_statuses(text, [span(text, "масла") | {"label": "ING"}], LEX) == ["absent"]

    def test_no_spans(self):
        assert mention_statuses("Готовим без масла", [], LEX) == []
