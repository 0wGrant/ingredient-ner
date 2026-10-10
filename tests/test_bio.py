import random

import pytest

from ingredient_ner.bio import IGNORE, align_labels, bio_to_spans, covered_words, spans_to_bio, split_words


def offsets_for(n, width=3, gap=1):
    return [(i * (width + gap), i * (width + gap) + width) for i in range(n)]


def span(start, end, label="ING"):
    return {"start": start, "end": end, "label": label}


def words_of(text):
    return [word for word, _start, _end in split_words(text)]


class TestSplitWords:
    def test_words_and_punctuation(self):
        assert words_of("Смешать муку, соль.") == ["Смешать", "муку", ",", "соль", "."]

    def test_digits_split_from_letters(self):
        assert words_of("3яйца и 200гр") == ["3", "яйца", "и", "200", "гр"]

    def test_hyphen_is_separate(self):
        assert words_of("какао-порошок") == ["какао", "-", "порошок"]

    def test_offsets(self):
        assert split_words("  Сыр\nи мёд") == [("Сыр", 2, 5), ("и", 6, 7), ("мёд", 8, 11)]


class TestSpansToBio:
    def test_simple(self):
        assert spans_to_bio([{"start": 4, "end": 11}], [(0, 3), (4, 7), (8, 11)]) == ["O", "B-ING", "I-ING"]

    def test_unsorted_spans(self):
        spans = [{"start": 8, "end": 11}, {"start": 0, "end": 3}]
        assert spans_to_bio(spans, [(0, 3), (4, 7), (8, 11)]) == ["B-ING", "O", "B-ING"]

    def test_touching_spans(self):
        spans = [{"start": 0, "end": 3}, {"start": 3, "end": 7}]
        assert spans_to_bio(spans, [(0, 3), (3, 4), (4, 7)]) == ["B-ING", "B-ING", "I-ING"]

    def test_label(self):
        assert spans_to_bio([{"start": 0, "end": 3}], [(0, 3)], label="FOOD") == ["B-FOOD"]

    def test_overlapping_spans_raise(self):
        with pytest.raises(ValueError):
            spans_to_bio([{"start": 0, "end": 7}, {"start": 4, "end": 7}], [(0, 3), (4, 7)])

    def test_span_without_words_raises(self):
        with pytest.raises(ValueError):
            spans_to_bio([{"start": 3, "end": 4}], [(0, 3), (4, 7)])


class TestBioToSpans:
    def test_empty(self):
        assert bio_to_spans([], []) == []

    def test_single_token_span(self):
        assert bio_to_spans(["O", "B-ING", "O"], offsets_for(3)) == [span(4, 7)]

    def test_span_at_the_very_end_is_closed(self):
        assert bio_to_spans(["O", "B-ING", "I-ING"], offsets_for(3)) == [span(4, 11)]

    def test_adjacent_b_tags_are_two_spans(self):
        assert bio_to_spans(["B-ING", "B-ING"], offsets_for(2)) == [span(0, 3), span(4, 7)]

    def test_positions_come_from_offsets(self):
        offsets = [(0, 6), (7, 20), (21, 22), (30, 41)]
        assert bio_to_spans(["O", "B-ING", "O", "B-ING"], offsets) == [span(7, 20), span(30, 41)]

    def test_offsets_may_be_lists(self):
        assert bio_to_spans(["B-ING", "I-ING"], [[0, 3], [4, 9]]) == [span(0, 9)]

    def test_i_at_start_opens_span(self):
        assert bio_to_spans(["I-ING", "I-ING", "O"], offsets_for(3)) == [span(0, 7)]

    def test_i_after_outside_opens_span(self):
        assert bio_to_spans(["B-ING", "O", "I-ING", "I-ING"], offsets_for(4)) == [span(0, 3), span(8, 15)]

    def test_i_of_other_type_opens_new_span(self):
        assert bio_to_spans(["B-ING", "I-FOO", "O"], offsets_for(3)) == [span(0, 3), span(4, 7, "FOO")]

    def test_label_with_hyphen(self):
        assert bio_to_spans(["B-SUB-TYPE", "I-SUB-TYPE"], offsets_for(2)) == [span(0, 7, "SUB-TYPE")]

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            bio_to_spans(["O", "B-ING"], offsets_for(3))

    @pytest.mark.parametrize("bad", ["B", "I", "B-", "I-", "X-ING", "b-ING", "O-ING", "", "ING", "BING"])
    def test_malformed_tag(self, bad):
        with pytest.raises(ValueError):
            bio_to_spans(["O", bad, "O"], offsets_for(3))

    def test_round_trip_with_spans_to_bio(self):
        rng = random.Random(7)
        for _ in range(300):
            n = rng.randint(1, 30)
            offsets = offsets_for(n, width=rng.randint(1, 4), gap=rng.randint(0, 2))
            spans, i = [], 0
            while i < n:
                if rng.random() < 0.3:
                    j = min(n, i + rng.randint(1, 4))
                    spans.append(span(offsets[i][0], offsets[j - 1][1]))
                    i = j
                else:
                    i += 1
            assert bio_to_spans(spans_to_bio(spans, offsets), offsets) == spans


class TestCoveredWords:
    def test_indices(self):
        offsets = [(0, 3), (4, 7), (8, 11), (12, 15)]
        assert covered_words([{"start": 4, "end": 11}, {"start": 12, "end": 15}], offsets) == [1, 2, 3]

    def test_partial_overlap_counts(self):
        assert covered_words([{"start": 5, "end": 6}], [(0, 3), (4, 7)]) == [1]


class TestAlignLabels:
    tags = ["O", "B-ING", "I-ING", "O"]
    word_ids = [None, 0, 0, 1, 1, 1, 2, 3, None]

    def test_first_subword_only(self):
        assert align_labels(self.tags, self.word_ids) == [IGNORE, 0, IGNORE, 1, IGNORE, IGNORE, 2, 0, IGNORE]

    def test_uncertain_words(self):
        assert align_labels(self.tags, self.word_ids, uncertain=[0, 1]) == [IGNORE] * 6 + [2, 0, IGNORE]
