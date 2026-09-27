"""SentimentDataset, ported from the archived repo (IMPLEMENTATION_GUIDE.md §3).

The raw `sentence` goes to the tokenizer unmodified — preprocessing.py is for
TF-IDF only, and cleaning fights WordPiece (CLAUDE.md).

Rows are truncated to `max_length` but not padded. DataCollatorWithPadding pads
each batch to its own longest row, so the cap only matters for the few
sentences that exceed it. §9 Deviations, 2026-09-27.
"""

from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "label": int(self.labels[idx]),
        }
