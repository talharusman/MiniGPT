# MiniGPT: a Transformer language model built from scratch

A small GPT-style causal language model. Attention, multi-head splitting, positional encoding,
the Transformer block, the causal mask, the shifted next-token loss, and sampling are all
implemented in `src/` with plain PyTorch tensor operations. No pretrained weights and no
`transformers` model classes are used.

Dataset: Tiny Shakespeare (character level, see `data/README.md`).

## Repository layout

```
configs/tiny_gpt.yaml        all hyper-parameters (every value is overridable)
src/attention.py             scaled dot-product attention, causal mask, multi-head attention   <- core
src/positional_encoding.py   sinusoidal (hand-written) and learned positions
src/transformer_block.py     FFN (GELU) + pre-LayerNorm decoder block with residuals
src/gpt.py                   MiniGPT: embeddings -> blocks -> LayerNorm -> LM head; generation
src/tokenizer.py             deterministic character tokenizer
src/dataset.py               download, contiguous split, shifted-target windows
src/train.py                 training loop, overfit gate, logging, checkpoints
src/generate.py              inference CLI
src/api.py                   FastAPI REST service
tests/                       unit tests: attention, mask, shapes, overfit gate
experiments/                 figures, ablations, evaluation
notebooks/MiniGPT_Colab.ipynb  end-to-end Colab run
report/REPORT.md             technical report (hand calculation included)
GUIDE.md                     step-by-step instructions for you
```

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the tests

```bash
pytest -v
```

The tests run offline on CPU in under a minute. They check the hand-worked attention example,
the causal mask (future weights exactly zero; changing future tokens leaves past logits
unchanged), head split/merge round trips, shapes, target shifting, and that a tiny batch can be
overfit.

## Train

```bash
# 1. Debugging gate: loss must fall substantially on one tiny batch (about 1 minute on CPU)
python -m src.train --overfit

# 2. Full training (CPU: hours; T4 GPU: roughly 10-25 minutes)
python -m src.train --config configs/tiny_gpt.yaml --run-name mini_gpt
```

Outputs: `checkpoints/mini_gpt_best.pt`, `outputs/logs/mini_gpt/{metrics.csv,summary.json,config.json}`,
`outputs/samples/mini_gpt_samples.md`.

## Generate text

```bash
python -m src.generate --prompt "ROMEO:" --max-new-tokens 80 --temperature 0.8
python -m src.generate --prompt "To be, or not to be" --greedy
```

Example output (your exact text will differ):

```
Prompt: ROMEO:
Generated: ROMEO:
...
model parameters: 0,xxx,xxx
context length: 128
temperature: 0.8
```

## REST API

```bash
uvicorn src.api:app --port 8000
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" \
     -d '{"prompt": "ROMEO:", "max_new_tokens": 80, "temperature": 0.8}'
```

Interactive docs at `http://localhost:8000/docs`. `GET /health` reports whether the checkpoint loaded.

## Experiments and evaluation

```bash
python -m experiments.static_figures          # causal mask, PE heatmap, shape trace, model summary
python -m experiments.plot_curves --run mini_gpt
python -m experiments.attention_visualization --prompt "ROMEO:"
python -m experiments.evaluate                # validation perplexity, causality check, samples
python -m experiments.ablation --steps 1500   # depth, heads, context length, position method
```

## Reproducibility

- Seeds are fixed in `configs/tiny_gpt.yaml` (`train.seed`).
- The exact configuration of each run is saved to `outputs/logs/<run>/config.json`.
- The checkpoint stores weights, model config and tokenizer, so inference needs only that file.

## Deployment

A `Dockerfile` is included for Render (or any container host). See `GUIDE.md`, Part 6.

## Known limitations

- Character-level tokenization gives long sequences for a given amount of text, so the model sees
  a short span of context (128 characters by default). Long-range structure is weak.
- The model is small (under 1M parameters by default) and trained for a limited number of steps.
  Samples are locally plausible Shakespearean-looking text, not coherent long passages, and are
  not factual.
- Validation loss is measured on the last 10% of one book-like text. Generalisation beyond this
  style is not claimed.
- Dropout is applied to the residual and FFN outputs but not to attention weights.
- Generation re-runs the full forward pass for every new token (no KV cache), so it is O(n²) in
  output length. Fine for this size, slow for long outputs.
- The API has no authentication and no rate limiting. Do not expose it publicly with secrets or
  heavy traffic.
- Output is generated text from an experimental model. It is not a reliable source of facts.

## Attribution

The sinusoidal encoding follows Vaswani et al., "Attention Is All You Need" (2017). Pre-LayerNorm
follows Xiong et al. (2020). The Tiny Shakespeare corpus comes from Karpathy's char-rnn repository
(Hugging Face mirror: `tiny_shakespeare`). The code was written for this assignment; no
transformer implementation was copied. Cite any tutorial you adapted in `report/REPORT.md`.
