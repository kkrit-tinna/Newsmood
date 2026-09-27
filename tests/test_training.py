import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from torch import nn

from newsmood.training.dataset import SentimentDataset
from newsmood.training.finetune import (
    DEVICE_CHOICES,
    ID2LABEL,
    LABEL2ID,
    EarlyStopping,
    class_weights,
    get_device,
    resolve_device,
    seed_everything,
)


def test_label_maps_match_guide():
    assert ID2LABEL == {0: "negative", 1: "neutral", 2: "positive"}
    assert LABEL2ID == {"negative": 0, "neutral": 1, "positive": 2}


def test_class_weights_balanced_formula():
    # n=6, k=3: 6/(3*1), 6/(3*3), 6/(3*2)
    weights = class_weights([0, 1, 1, 1, 2, 2], num_classes=3)
    assert weights.tolist() == pytest.approx([2.0, 2 / 3, 1.0])


def test_class_weights_rarest_class_weighs_most():
    labels = [0] * 12 + [1] * 62 + [2] * 26
    weights = class_weights(labels, num_classes=3)
    assert weights.argmax().item() == 0
    assert weights.argmin().item() == 1


def test_class_weights_missing_class_raises():
    with pytest.raises(ValueError, match=r"\[2\]"):
        class_weights([0, 1, 1], num_classes=3)


def test_early_stopping_restores_best_weights_not_live_ones():
    model = nn.Linear(2, 2)
    best = {k: v.clone() for k, v in model.state_dict().items()}
    stopper = EarlyStopping(patience=1)

    assert stopper(1.0, model) is False
    with torch.no_grad():
        model.weight.fill_(5.0)  # in-place, as optimizer.step() does
    assert stopper(2.0, model) is True

    for k, v in model.state_dict().items():
        assert torch.equal(v, best[k]), f"{k} was not restored"


def test_early_stopping_counter_resets_on_improvement():
    model = nn.Linear(2, 2)
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert stopper(1.0, model) is False
    assert stopper(1.0, model) is False  # no improvement, counter 1
    assert stopper(0.5, model) is False  # improvement, counter 0
    assert stopper(0.5, model) is False  # counter 1
    assert stopper(0.5, model) is True  # counter 2


class RecordingTokenizer:
    """Word-level stand-in that records what it was handed."""

    def __init__(self):
        self.seen = []

    def __call__(self, text, truncation, max_length):
        self.seen.append(text)
        ids = list(range(len(text.split())))
        if truncation:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_sentiment_dataset_passes_raw_text_and_truncates_without_padding():
    tokenizer = RecordingTokenizer()
    texts = ["Profit fell, NOT rose!", "one two three four five six"]
    ds = SentimentDataset(texts, [0, 2], tokenizer, max_length=4)

    short, long = ds[0], ds[1]

    assert tokenizer.seen == texts
    assert len(short["input_ids"]) == 4 and len(long["input_ids"]) == 4
    assert SentimentDataset(["a b"], [1], tokenizer, max_length=4)[0]["input_ids"] == [0, 1]
    assert short["label"] == 0 and long["label"] == 2


def test_resolve_device_explicit_cpu_overrides_available_accelerator(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_defers_to_get_device(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("mps")


@pytest.mark.parametrize("choice, backend", [("mps", "mps"), ("cuda", "cuda")])
def test_resolve_device_unavailable_backend_raises(monkeypatch, choice, backend):
    module = torch.backends.mps if backend == "mps" else torch.cuda
    monkeypatch.setattr(module, "is_available", lambda: False)
    with pytest.raises(ValueError, match=choice):
        resolve_device(choice)


def test_resolve_device_unknown_raises():
    with pytest.raises(ValueError, match="tpu"):
        resolve_device("tpu")


def test_cli_device_choices_match_finetune():
    # cli.py lists the choices literally so it can import finetune lazily.
    from newsmood.cli import Device

    assert tuple(d.value for d in Device) == DEVICE_CHOICES


@pytest.mark.parametrize(
    "cuda, mps, expected",
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_get_device_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert get_device() == torch.device(expected)


def test_seed_everything_makes_all_three_rngs_repeat():
    import random

    import numpy as np

    def draw():
        return random.random(), np.random.rand(), torch.rand(3).tolist(), nn.Linear(4, 3).weight.tolist()

    seed_everything(42)
    first = draw()
    seed_everything(42)
    assert draw() == first
    seed_everything(43)
    assert draw() != first


# --- T2.2 training loop, driven with a tiny random DistilBERT (no download) ---

from newsmood.config import TrainingSettings
from newsmood.training import finetune

WORDS = ["profit", "rose", "fell", "loss", "sales", "flat", "record", "cut"]


def _tiny_model_and_tokenizer(tmp_path):
    from transformers import BertTokenizer, DistilBertConfig, DistilBertForSequenceClassification

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    # transformers 5 ignores vocab_file here (every word maps to [UNK]); pass the dict.
    tokenizer = BertTokenizer(vocab={word: i for i, word in enumerate(vocab)})
    config = DistilBertConfig(
        vocab_size=len(vocab), dim=8, n_layers=1, n_heads=2, hidden_dim=16, max_position_embeddings=32,
        num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    return DistilBertForSequenceClassification(config), tokenizer


def _hp(tmp_path, **overrides):
    values = dict(
        max_length=16, epochs=2, batch_size=4, learning_rate=1e-3, weight_decay=0.01, warmup_ratio=0.1,
        early_stopping_patience=3, early_stopping_min_delta=0.001, output_dir=str(tmp_path / "local"),
    )
    return TrainingSettings(**(values | overrides))


def _data():
    texts = [" ".join(WORDS[i % 8 : i % 8 + 3]) for i in range(18)]
    labels = [i % 3 for i in range(18)]
    return texts[:12], labels[:12], texts[12:], labels[12:]


def _run(tmp_path, hp):
    seed_everything(0)
    model, tokenizer = _tiny_model_and_tokenizer(tmp_path)
    return finetune.train_loop(model, tokenizer, *_data(), hp, torch.device("cpu"), 0, {"base_model": "tiny"})


def test_train_loop_writes_history_and_saves_loadable_model_and_tokenizer(tmp_path, capsys):
    import json

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hp = _hp(tmp_path)
    returned = _run(tmp_path, hp)

    out = tmp_path / "local"
    history = json.loads((out / "history.json").read_text())
    assert history == returned
    assert history["status"] == "complete"
    assert history["device"] == "cpu" and history["seed"] == 0 and history["base_model"] == "tiny"
    assert history["hyperparameters"]["learning_rate"] == 1e-3
    assert history["hyperparameters"]["total_steps"] == 6  # ceil(12 / 4) * 2
    assert history["hyperparameters"]["warmup_steps"] == 0  # int(0.1 * 6)
    assert [e["epoch"] for e in history["epochs"]] == [1, 2]
    assert set(history["epochs"][0]) == {"epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "seconds"}

    model = AutoModelForSequenceClassification.from_pretrained(out)
    tokenizer = AutoTokenizer.from_pretrained(out)
    assert model.config.id2label == ID2LABEL
    assert tokenizer("profit rose")["input_ids"][1:3] == [5, 6]

    printed = capsys.readouterr().out
    assert "epoch 1/2  train_loss" in printed and "val_acc" in printed and "val_macro_f1" in printed


def test_train_loop_is_reproducible_under_the_same_seed(tmp_path):
    first = _run(tmp_path / "a", _hp(tmp_path / "a"))
    second = _run(tmp_path / "b", _hp(tmp_path / "b"))
    strip = lambda h: [{k: v for k, v in e.items() if k != "seconds"} for e in h["epochs"]]
    assert strip(first) == strip(second)


def test_history_keeps_finished_epochs_when_a_later_epoch_crashes(tmp_path, monkeypatch):
    import json

    real = finetune.evaluate_split
    calls = []

    def crash_on_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated crash in epoch 2")
        return real(*args, **kwargs)

    monkeypatch.setattr(finetune, "evaluate_split", crash_on_second)
    with pytest.raises(RuntimeError, match="epoch 2"):
        _run(tmp_path, _hp(tmp_path))

    history = json.loads((tmp_path / "local" / "history.json").read_text())
    assert history["status"] == "running"
    assert [e["epoch"] for e in history["epochs"]] == [1]
    assert not (tmp_path / "local" / "config.json").exists()


def test_format_epoch_shows_all_five_numbers():
    line = finetune.format_epoch(
        {"epoch": 1, "train_loss": 0.81234, "val_loss": 0.6, "val_accuracy": 0.7821, "val_macro_f1": 0.721, "seconds": 38.24},
        3,
    )
    assert line == "epoch 1/3  train_loss 0.8123  val_loss 0.6000  val_acc 78.21%  val_macro_f1 72.10%  38.2s"
