import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ingredient_ner.bio import IGNORE  # noqa: E402
from ingredient_ner.ner import (MODEL_NAME, collate, decode_words, encode_record, predict, predict_spans,  # noqa: E402
                                predict_text, train_epoch, word_labels)

CUDA = torch.cuda.is_available()


def tiny_model(seed=0, vocab_size=40):
    torch.manual_seed(seed)
    config = transformers.BertConfig(vocab_size=vocab_size, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                     intermediate_size=32, max_position_embeddings=64, num_labels=3,
                                     hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    return transformers.BertForTokenClassification(config)


def toy_batches(n_batches=4, batch_size=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        lengths = torch.randint(4, 10, (batch_size,), generator=g)
        width = int(lengths.max())
        input_ids = torch.randint(5, 40, (batch_size, width), generator=g)
        mask = (torch.arange(width)[None, :] < lengths[:, None]).long()
        labels = torch.where(mask.bool(), input_ids % 3, torch.full_like(input_ids, IGNORE))
        labels[:, 0] = IGNORE
        batches.append({"input_ids": input_ids * mask, "attention_mask": mask, "labels": labels})
    return batches


class Recorder:
    def __init__(self, model):
        self.events = []
        self.forward_calls = []
        self.clean_before_backward = []
        self.norms = []
        self.model = model
        model.classifier.weight.register_hook(self._on_backward)
        model.register_forward_pre_hook(self._on_forward, with_kwargs=True)

    def _on_backward(self, grad):
        param = self.model.classifier.weight
        self.clean_before_backward.append(param.grad is None or not bool(param.grad.abs().sum()))
        self.events.append("backward")
        return grad

    def _on_forward(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        self.forward_calls.append({"training": module.training, "grad": torch.is_grad_enabled(),
                                   "kwargs": set(kwargs), "device": ids.device.type})

    def total_grad_norm(self):
        grads = [p.grad.detach().norm() for p in self.model.parameters() if p.grad is not None]
        return float(torch.stack(grads).norm()) if grads else 0.0


class SpyOptimizer:
    def __init__(self, optimizer, recorder):
        self.optimizer = optimizer
        self.recorder = recorder
        self.param_groups = optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        self.recorder.events.append("zero_grad")
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.recorder.norms.append(self.recorder.total_grad_norm())
        self.recorder.events.append("step")
        return self.optimizer.step()


class SpyScheduler:
    def __init__(self, recorder):
        self.recorder = recorder

    def step(self):
        self.recorder.events.append("scheduler")


def spied(model, lr=0.0):
    recorder = Recorder(model)
    return recorder, SpyOptimizer(torch.optim.SGD(model.parameters(), lr=lr), recorder), SpyScheduler(recorder)


RECORD = {
    "url": "u",
    "words": ["Взбить", "белки", "с", "сахаром", "."],
    "offsets": [[0, 6], [7, 12], [13, 14], [15, 22], [22, 23]],
    "tags": ["O", "B-ING", "O", "B-ING", "O"],
    "uncertain": [1],
    "spans": [{"start": 7, "end": 12}, {"start": 15, "end": 22}],
    "dropped": [{"start": 0, "end": 6}],
}


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    except OSError:
        pytest.skip("нет токенизатора")


class TestWordLabels:
    def test_variants(self):
        assert word_labels(RECORD, "B") == (["O", "B-ING", "O", "B-ING", "O"], [])
        assert word_labels(RECORD, "C") == (["O", "B-ING", "O", "B-ING", "O"], [1])
        assert word_labels(RECORD, "A") == (["B-ING", "B-ING", "O", "B-ING", "O"], [])

    def test_variant_a_without_dropped(self):
        record = {key: value for key, value in RECORD.items() if key != "dropped"}
        assert word_labels(record, "A") == (["O", "B-ING", "O", "B-ING", "O"], [])

    def test_variant_a_does_not_mutate(self):
        record = {key: list(value) if isinstance(value, list) else value for key, value in RECORD.items()}
        word_labels(record, "A")
        assert record["spans"] == RECORD["spans"]

    @pytest.mark.parametrize("bad", ["D", "b", "", "AB"])
    def test_unknown_variant(self, bad):
        with pytest.raises(ValueError):
            word_labels(RECORD, bad)


class TestEncodeRecord:
    def test_keys_and_lengths(self, tokenizer):
        encoded = encode_record(RECORD, tokenizer)
        assert set(encoded) == {"input_ids", "attention_mask", "labels", "word_ids"}
        assert len({len(value) for value in encoded.values()}) == 1

    def test_ids_match_tokenizer(self, tokenizer):
        encoded = encode_record(RECORD, tokenizer)
        assert encoded["input_ids"] == tokenizer(RECORD["words"], is_split_into_words=True)["input_ids"]
        assert encoded["word_ids"] == [None, 0, 0, 1, 2, 3, 4, None]

    def test_labels_by_variant(self, tokenizer):
        assert encode_record(RECORD, tokenizer, "B")["labels"] == [IGNORE, 0, IGNORE, 1, 0, 1, 0, IGNORE]
        assert encode_record(RECORD, tokenizer, "C")["labels"] == [IGNORE, 0, IGNORE, IGNORE, 0, 1, 0, IGNORE]
        assert encode_record(RECORD, tokenizer, "A")["labels"] == [IGNORE, 1, IGNORE, 1, 0, 1, 0, IGNORE]

    def test_truncation(self, tokenizer):
        encoded = encode_record(RECORD, tokenizer, max_length=5)
        assert encoded["input_ids"][-1] == tokenizer.sep_token_id
        assert encoded["word_ids"] == [None, 0, 0, 1, None]


class TestCollate:
    features = [
        {"input_ids": [2, 10, 3], "attention_mask": [1, 1, 1], "labels": [IGNORE, 1, IGNORE],
         "word_ids": [None, 0, None]},
        {"input_ids": [2, 11, 12, 13, 3], "attention_mask": [1, 1, 1, 1, 1], "labels": [IGNORE, 0, 2, 0, IGNORE],
         "word_ids": [None, 0, 1, 2, None]},
    ]

    def test_padding(self):
        batch = collate(self.features, pad_id=7)
        assert batch["input_ids"].tolist() == [[2, 10, 3, 7, 7], [2, 11, 12, 13, 3]]
        assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        assert batch["labels"].tolist() == [[IGNORE, 1, IGNORE, IGNORE, IGNORE], [IGNORE, 0, 2, 0, IGNORE]]

    def test_only_long_tensors_for_model(self):
        batch = collate(self.features, pad_id=0)
        assert set(batch) == {"input_ids", "attention_mask", "labels"}
        assert all(value.dtype == torch.long for value in batch.values())

    def test_features_not_mutated(self):
        features = [{key: list(value) for key, value in f.items()} for f in self.features]
        collate(features, pad_id=0)
        assert features == self.features


class TestTrainEpoch:
    def test_returns_mean_batch_loss(self):
        model = tiny_model()
        batches = toy_batches()
        with torch.no_grad():
            expected = sum(model(**b).loss.item() for b in batches) / len(batches)
        _recorder, optimizer, scheduler = spied(model, lr=0.0)
        result = train_epoch(model, batches, optimizer, scheduler, "cpu")
        assert isinstance(result, float)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_one_backward_step_and_scheduler_per_batch(self):
        recorder, optimizer, scheduler = spied(model := tiny_model())
        train_epoch(model, toy_batches(n_batches=5), optimizer, scheduler, "cpu")
        assert [e for e in recorder.events if e != "zero_grad"] == ["backward", "step", "scheduler"] * 5

    def test_gradients_zeroed_before_each_backward(self):
        model = tiny_model()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        recorder, optimizer, scheduler = spied(model)
        train_epoch(model, toy_batches(), optimizer, scheduler, "cpu")
        assert recorder.clean_before_backward == [True] * 4

    def test_gradient_clipping(self):
        recorder, optimizer, scheduler = spied(model := tiny_model())
        train_epoch(model, toy_batches(), optimizer, scheduler, "cpu", max_grad_norm=1e-3)
        assert all(norm <= 1e-3 * 1.001 for norm in recorder.norms)

    def test_default_clipping_is_one(self):
        model = tiny_model()
        for p in model.classifier.parameters():
            p.data *= 1000
        recorder, optimizer, scheduler = spied(model)
        train_epoch(model, toy_batches(), optimizer, scheduler, "cpu")
        assert all(norm <= 1.001 for norm in recorder.norms)

    def test_train_mode_and_labels(self):
        model = tiny_model()
        model.eval()
        recorder, optimizer, scheduler = spied(model)
        train_epoch(model, toy_batches(), optimizer, scheduler, "cpu")
        assert all(call["training"] and call["grad"] and "labels" in call["kwargs"] for call in recorder.forward_calls)

    def test_loss_goes_down(self):
        model = tiny_model()
        batches = toy_batches()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        losses = [train_epoch(model, batches, optimizer, scheduler, "cpu") for _ in range(40)]
        assert losses[-1] < 0.5 * losses[0]

    @pytest.mark.skipif(not CUDA, reason="нет CUDA")
    def test_moves_batch_to_device(self):
        model = tiny_model().to("cuda")
        recorder, optimizer, scheduler = spied(model)
        train_epoch(model, toy_batches(), optimizer, scheduler, torch.device("cuda"))
        assert all(call["device"] == "cuda" for call in recorder.forward_calls)


class TestPredict:
    @staticmethod
    def expected(model, batches):
        model.eval()
        result = []
        with torch.no_grad():
            for b in batches:
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
                result += [row[:int(mask.sum())].tolist() for row, mask in zip(logits.argmax(-1), b["attention_mask"])]
        return result

    def test_argmax_without_padding(self):
        model = tiny_model(seed=3)
        batches = toy_batches(n_batches=3, seed=5)
        expected = self.expected(model, batches)
        model.train()
        assert predict(model, batches, "cpu") == expected

    def test_python_ints(self):
        result = predict(tiny_model(), toy_batches(n_batches=1), "cpu")
        assert all(type(value) is int for row in result for value in row)

    def test_eval_mode_no_grad_no_labels(self):
        model = tiny_model()
        recorder = Recorder(model)
        predict(model, toy_batches(n_batches=2), "cpu")
        assert all(not call["training"] and not call["grad"] and "labels" not in call["kwargs"]
                   for call in recorder.forward_calls)

    def test_batches_without_labels(self):
        batches = [{k: v for k, v in b.items() if k != "labels"} for b in toy_batches(n_batches=2)]
        assert len(predict(tiny_model(), batches, "cpu")) == 6

    @pytest.mark.skipif(not CUDA, reason="нет CUDA")
    def test_on_cuda(self):
        model = tiny_model(seed=3)
        batches = toy_batches(n_batches=2, seed=5)
        expected = self.expected(model, batches)
        assert predict(model.to("cuda"), batches, "cuda") == expected


class TestDecodeWords:
    word_ids = [None, 0, 0, 1, 2, 3, 4, None]

    def test_first_subword(self):
        assert decode_words([2, 1, 2, 2, 0, 1, 0, 2], self.word_ids, 5) == ["B-ING", "I-ING", "O", "B-ING", "O"]

    def test_continuation_ignored(self):
        assert decode_words([0, 0, 1, 0, 0, 0, 0, 0], self.word_ids, 5) == ["O"] * 5

    def test_truncated_words_are_outside(self):
        assert decode_words([0, 1, 0, 2, 0, 1, 0, 0], self.word_ids, 7) == ["B-ING", "I-ING", "O", "B-ING", "O",
                                                                             "O", "O"]

    def test_no_words(self):
        assert decode_words([0, 0], [None, None], 0) == []


class TestPredictText:
    def test_every_word_when_model_always_says_b(self, tokenizer):
        model = tiny_model(vocab_size=tokenizer.vocab_size)
        with torch.no_grad():
            model.classifier.bias.copy_(torch.tensor([0.0, 100.0, 0.0]))
        assert predict_text(model, tokenizer, "Сыр, мёд", "cpu") == [
            {"start": 0, "end": 3, "label": "ING"}, {"start": 3, "end": 4, "label": "ING"},
            {"start": 5, "end": 8, "label": "ING"}]

    def test_nothing_when_model_always_says_o(self, tokenizer):
        model = tiny_model(vocab_size=tokenizer.vocab_size)
        with torch.no_grad():
            model.classifier.bias.copy_(torch.tensor([100.0, 0.0, 0.0]))
        assert predict_text(model, tokenizer, "Сыр и мёд", "cpu") == []

    def test_empty_text(self, tokenizer):
        assert predict_text(tiny_model(vocab_size=tokenizer.vocab_size), tokenizer, "", "cpu") == []


class TestPredictSpans:
    def test_spans_follow_word_offsets(self, tokenizer):
        model = tiny_model(vocab_size=tokenizer.vocab_size)
        with torch.no_grad():
            model.classifier.bias.copy_(torch.tensor([0.0, 100.0, 0.0]))
        records = [RECORD, RECORD | {"url": "v"}]
        result = predict_spans(model, tokenizer, records, "cpu", batch_size=1)
        expected = [{"start": s, "end": e, "label": "ING"} for s, e in RECORD["offsets"]]
        assert result == {"u": expected, "v": expected}
