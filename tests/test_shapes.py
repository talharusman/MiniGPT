"""Shape checks through every component, plus the next-token target shift."""
import torch

from src.attention import MultiHeadSelfAttention
from src.dataset import NextTokenDataset, split_text
from src.gpt import GPTConfig, MiniGPT
from src.positional_encoding import LearnedPositionalEmbedding, SinusoidalPositionalEncoding, sinusoidal_encoding
from src.tokenizer import CharTokenizer
from src.transformer_block import DecoderBlock


def test_split_and_merge_heads_is_identity():
    attn = MultiHeadSelfAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 5, 32)
    h = attn.split_heads(x)
    assert h.shape == (2, 4, 5, 8)
    assert torch.equal(attn.merge_heads(h), x)


def test_block_preserves_shape():
    block = DecoderBlock(d_model=32, n_heads=4, ffn_dim=128, dropout=0.0)
    x = torch.randn(3, 9, 32)
    assert block(x).shape == (3, 9, 32)


def test_model_logits_shape():
    cfg = GPTConfig(vocab_size=65, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = MiniGPT(cfg)
    idx = torch.randint(0, 65, (4, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, 16, 65)
    assert loss.dim() == 0


def test_sinusoidal_encoding_shape_and_distinct_positions():
    pe = sinusoidal_encoding(64, 16)
    assert pe.shape == (64, 16)
    assert not torch.allclose(pe[0], pe[1])  # each position gets a unique signature
    assert torch.allclose(pe[0, 0::2], torch.zeros(8))  # sin(0) = 0 at position 0
    assert torch.allclose(pe[0, 1::2], torch.ones(8))   # cos(0) = 1 at position 0


def test_position_modules_add_without_changing_shape():
    x = torch.zeros(2, 10, 16)
    assert SinusoidalPositionalEncoding(32, 16)(x).shape == (2, 10, 16)
    assert LearnedPositionalEmbedding(32, 16)(x).shape == (2, 10, 16)


def test_next_token_targets_are_shifted_by_one():
    ids = list(range(100))
    ds = NextTokenDataset(ids, context_length=8)
    x, y = ds[0]
    assert torch.equal(y, x + 1)
    x, y = ds[3]
    assert torch.equal(x, torch.arange(24, 32))
    assert torch.equal(y, torch.arange(25, 33))


def test_tokenizer_is_deterministic_and_round_trips():
    tok = CharTokenizer.from_text("hello world")
    assert tok.vocab_size == len(set("hello world"))
    assert tok.decode(tok.encode("hello")) == "hello"
    assert CharTokenizer.from_text("hello world").chars == tok.chars


def test_split_text_is_contiguous():
    train, val = split_text("abcdefghij", val_fraction=0.2)
    assert train + val == "abcdefghij"
    assert len(val) == 2
