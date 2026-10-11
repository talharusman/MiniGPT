"""Corpus download, train/val split, and next-token (shifted-target) windows."""
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_corpus(path, url=TINY_SHAKESPEARE_URL) -> str:
    """Return corpus text, downloading it once if it is not already on disk."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path.read_text(encoding="utf-8")


def split_text(text: str, val_fraction: float = 0.1):
    """Contiguous split: first part trains, last part validates (no shuffling of text)."""
    cut = int(len(text) * (1 - val_fraction))
    return text[:cut], text[cut:]


class NextTokenDataset(Dataset):
    """Fixed-length windows with shifted targets.

    For window start s:  x = ids[s : s+T]  and  y = ids[s+1 : s+T+1].
    So the target at position t is the token immediately after input token t.
    """

    def __init__(self, ids, context_length: int, stride: int = None):
        self.ids = torch.as_tensor(list(ids), dtype=torch.long)
        self.T = context_length
        self.stride = stride or context_length
        n = (len(self.ids) - self.T - 1) // self.stride + 1
        self.n = max(0, n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, k):
        s = k * self.stride
        x = self.ids[s : s + self.T]
        y = self.ids[s + 1 : s + self.T + 1]
        return x, y
