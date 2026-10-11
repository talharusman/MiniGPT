"""Deterministic character-level tokenizer.

Vocabulary: the sorted set of unique characters in the corpus, so the same corpus
always yields the same token IDs (ID 0 = smallest character code point).

Special-token policy: none. A single contiguous corpus needs no PAD/BOS/EOS tokens
because batches are fixed-length windows, and there is no UNK token because the
vocabulary is built from the full corpus. Characters outside the vocabulary are
silently dropped by encode(); the generate CLI reports this only indirectly, so
keep prompts within the corpus character set.
"""
import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"chars": self.chars}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["chars"])
