import pytest

from newsmood.models.classifier import IncompleteCheckpoint, load_classifier, output_labels, predict

# ProsusAI/finbert config.json at the pinned revision.
FINBERT_ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}


def test_output_labels_follow_the_checkpoint_not_our_order():
    assert output_labels(FINBERT_ID2LABEL) == ["positive", "negative", "neutral"]


def test_output_labels_normalizes_case():
    assert output_labels({0: "Positive", 1: "NEGATIVE", 2: "neutral"}) == ["positive", "negative", "neutral"]


@pytest.mark.parametrize(
    "id2label",
    [
        {0: "positive", 1: "negative"},
        {0: "positive", 1: "negative", 2: "mixed"},
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
    ],
)
def test_output_labels_rejects_a_checkpoint_that_does_not_name_our_labels(id2label):
    with pytest.raises(ValueError, match="does not name exactly"):
        output_labels(id2label)


def test_predict_maps_each_output_index_by_name():
    torch = pytest.importorskip("torch")

    class Config:
        id2label = FINBERT_ID2LABEL

    class FakeModel:
        config = Config()

        def eval(self):
            pass

        def to(self, device):
            return self

        def __call__(self, input_ids, **_):
            # Row i's argmax is its first token id: 0, 1, 2 -> positive, negative, neutral.
            logits = torch.nn.functional.one_hot(input_ids[:, 0], num_classes=3).float()
            return type("Out", (), {"logits": logits})()

    class Batch(dict):
        def to(self, device):
            return self

    def fake_tokenizer(texts, **_):
        return Batch(input_ids=torch.tensor([[int(t)] for t in texts]))

    sentences = ["0", "1", "2", "1"]
    preds = predict(sentences, FakeModel(), fake_tokenizer, batch_size=3, device="cpu")
    assert preds == ["positive", "negative", "neutral", "negative"]


def _tiny(tmp_path, with_head: bool):
    pytest.importorskip("torch")
    from transformers import BertTokenizer, DistilBertConfig, DistilBertForSequenceClassification, DistilBertModel

    config = DistilBertConfig(
        vocab_size=8, dim=8, n_layers=1, n_heads=2, hidden_dim=16, max_position_embeddings=16,
        num_labels=3, id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
    )
    model = DistilBertForSequenceClassification(config) if with_head else DistilBertModel(config)
    model.save_pretrained(tmp_path)
    BertTokenizer(vocab={w: i for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "up"])}).save_pretrained(tmp_path)


def test_load_classifier_loads_a_complete_checkpoint(tmp_path):
    _tiny(tmp_path, with_head=True)
    model, tokenizer = load_classifier(str(tmp_path))
    assert output_labels(model.config.id2label) == ["negative", "neutral", "positive"]
    assert predict(["up"], model, tokenizer, batch_size=1, device="cpu")[0] in {"negative", "neutral", "positive"}


def test_load_classifier_refuses_a_checkpoint_without_a_classifier_head(tmp_path):
    # A base-model checkpoint loads "fine" into a classification class, with a
    # randomly initialised head — exactly what must not be scored.
    _tiny(tmp_path, with_head=False)
    with pytest.raises(IncompleteCheckpoint, match="classifier"):
        load_classifier(str(tmp_path))
