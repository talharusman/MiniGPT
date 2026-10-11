# Data

The corpus is **Tiny Shakespeare** (about 1.1 MB, about 40,000 lines of Shakespeare's plays).

- Hugging Face dataset card: https://huggingface.co/datasets/tiny_shakespeare
- Downloaded automatically by `src/dataset.py` from the Karpathy char-rnn mirror on GitHub
  (the same `input.txt` file) into `data/tinyshakespeare.txt` the first time you train.
- The file is not committed to git (see `.gitignore`). Delete it to force a fresh download.
