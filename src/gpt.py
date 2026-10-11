"""MiniGPT: token embedding + position -> causal Transformer blocks -> LayerNorm -> LM head."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.positional_encoding import LearnedPositionalEmbedding, SinusoidalPositionalEncoding
from src.transformer_block import DecoderBlock


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    pos_encoding: str = "sinusoidal"  # "sinusoidal" or "learned"
    tie_weights: bool = False


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_encoding == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(cfg.context_length, cfg.d_model)
        elif cfg.pos_encoding == "learned":
            self.pos = LearnedPositionalEmbedding(cfg.context_length, cfg.d_model)
        else:
            raise ValueError(f"unknown pos_encoding {cfg.pos_encoding!r}")
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(cfg.d_model, cfg.n_heads, cfg.ffn_mult * cfg.d_model, cfg.dropout, causal=True)
             for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight  # optional experiment
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, verbose: bool = False):
        """idx: (B, T) token IDs. Returns logits (B, T, V) and optional mean cross-entropy loss."""
        B, T = idx.shape
        if T > self.cfg.context_length:
            raise ValueError(f"sequence length {T} exceeds context_length {self.cfg.context_length}")
        x = self.tok_emb(idx)  # (B, T, D)
        if verbose:
            print(f"[model] token ids {tuple(idx.shape)} -> embeddings {tuple(x.shape)}")
        x = self.drop(self.pos(x))
        for block in self.blocks:
            x = block(x, verbose=verbose)
        logits = self.lm_head(self.ln_f(x))  # (B, T, V)
        if verbose:
            print(f"[model] logits {tuple(logits.shape)}  (B, T, vocab)")
        loss = None
        if targets is not None:
            # Position t is scored against targets[t] = token t+1 (shift is done by the dataset).
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = None, greedy: bool = False):
        """Autoregressive sampling. Each step re-runs the causal model on the last context_length tokens."""
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.cfg.context_length:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]  # only the last position predicts the next token
                if greedy:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / max(temperature, 1e-5)
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
        finally:
            self.train(was_training)
        return idx
