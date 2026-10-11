# Transformer & GPT from Scratch: Technical Report

> Fill every `TODO` with numbers and observations from YOUR run. Sections 1-5 and 9-10 are
> written as prompts. Section 6 (hand calculation) is worked out below as a reference; you must
> redo it yourself on paper and include your own photo of it.

**Team:** TODO names
**Repository:** TODO GitHub link
**Live API:** TODO Render link
**Demo video:** TODO link

---

## 1. Problem and scenario

TODO: In your own words, describe the scenario (a specialised organisation with a small private
corpus) and the question: can a small Transformer trained from scratch learn the next-token
structure of a domain corpus well enough to generate coherent text?

---

## 2. Transformer foundations

### 2.1 Why attention (Lesson 01)

TODO: Compare sequential depth and parallel depth of an RNN (O(T) sequential steps per layer) with
self-attention (O(1) sequential steps per layer, O(T^2) memory for the score matrix). Include one
measured or reasoned comparison of speed or memory. Your ablation on context length (Section 7)
can supply the measurement.

### 2.2 Scaled dot-product attention

    Q = X W_Q,   K = X W_K,   V = X W_V
    scores  = Q K^T / sqrt(d_k)
    weights = softmax(scores + mask)
    output  = weights V

Why divide by sqrt(d_k): TODO explain in your own words (dot products grow with d_k, pushing softmax
into saturated regions where gradients vanish).

### 2.3 Multi-head attention (Lesson 03)

    x in R^{B x T x D}  --split-->  (B, H, T, d)  with d = D/H
    each head attends independently
    concat (B, T, D)  --W_O-->  (B, T, D)

TODO: insert the shape trace (`outputs/logs/shape_trace.txt`) and explain each line.

### 2.4 Positional encoding (Lesson 04)

    PE[pos, 2i]   = sin(pos / 10000^{2i/D})
    PE[pos, 2i+1] = cos(pos / 10000^{2i/D})

TODO: explain why attention alone is permutation-equivariant and why position must be injected.

### 2.5 Transformer block (Lesson 05)

    x = x + Attn(LN(x))
    x = x + FFN(LN(x)),   FFN(x) = W_2 GELU(W_1 x)

TODO: explain residual paths and why we chose pre-LN over post-LN.

### 2.6 BERT masked LM vs GPT causal LM (Lesson 06 and 07)

TODO: compare the two attention masks. BERT: every position attends to every position, so a masked
token can use context on both sides. GPT: a lower-triangular mask, so position t only sees
positions <= t, which makes next-token prediction well defined. Include your mask-comparison
experiment if you ran one.

---

## 3. Implementation

TODO: tokenizer (character level, vocabulary size and special-token policy from `src/tokenizer.py`),
model configuration, attention implementation, masking, training loop, checkpointing. Cite any
code or tutorial you adapted.

---

## 4. Dataset and preprocessing

- **Source:** Tiny Shakespeare (Karpathy char-rnn mirror / Hugging Face `tiny_shakespeare`).
- **Size:** TODO characters, TODO lines (measure: `wc -c data/tinyshakespeare.txt`).
- **Vocabulary:** TODO (`vocab_size` printed by training).
- **Split:** contiguous, last 10% validation. TODO number of train and validation windows.
- **Context length:** TODO.
- **Filtering:** none (TODO confirm).

---

## 5. Training setup

| Item | Value (fill from `outputs/logs/mini_gpt/summary.json` and `config.json`) |
|---|---|
| Hardware | TODO `hardware` |
| Parameters | TODO `parameters` |
| Layers / heads / d_model / FFN | TODO |
| Context length | TODO |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 on matrices |
| Learning rate | TODO peak, min, warm-up steps, cosine schedule |
| Batch size | TODO |
| Steps | TODO |
| Dropout | TODO |
| Wall-clock training time | TODO `train_time_s` |
| Throughput | TODO `tokens_per_sec` |
| Overfit gate result | TODO first and final train loss from `outputs/logs/overfit/summary.json` |

Justify any change from the starting recommendation: TODO.

---

## 6. Hand calculation (required, graded separately)

### 6.1 Setup

Three tokens, d_k = d_v = 2:

    Q = [[1, 0],      K = [[1, 0],      V = [[1, 2],
         [0, 1],           [0, 1],           [3, 4],
         [1, 1]]           [1, 1]]           [5, 6]]

### 6.2 QK^T

Entry (i, j) is the dot product of query row i with key row j:

    QK^T = [[1, 0, 1],
            [0, 1, 1],
            [1, 1, 2]]

### 6.3 Divide by sqrt(d_k) = sqrt(2) = 1.4142

    scores = [[0.7071, 0.0000, 0.7071],
              [0.0000, 0.7071, 0.7071],
              [0.7071, 0.7071, 1.4142]]

### 6.4 Causal mask (future positions set to -inf)

Token 1 may see only token 1, token 2 may see tokens 1-2, and token 3 may see tokens 1-3.

    masked = [[ 0.7071,   -inf,   -inf],
              [ 0.0000,  0.7071,  -inf],
              [ 0.7071,  0.7071,  1.4142]]

### 6.5 Row-wise softmax

Row 1: only one allowed key, so weights = [1, 0, 0].

Row 2: exp(0) = 1, exp(0.7071) = 2.0281, sum = 3.0281, weights = [0.3302, 0.6698, 0].

Row 3: exp(0.7071) = 2.0281 (twice), exp(1.4142) = 4.1133, sum = 8.1695, weights = [0.2483, 0.2483, 0.5035].

    weights = [[1.0000, 0.0000, 0.0000],
               [0.3302, 0.6698, 0.0000],
               [0.2483, 0.2483, 0.5035]]

### 6.6 Multiply by V

    output_1 = 1.0000*[1,2]                                  = [1.0000, 2.0000]
    output_2 = 0.3302*[1,2] + 0.6698*[3,4]                   = [2.3395, 3.3395]
    output_3 = 0.2483*[1,2] + 0.2483*[3,4] + 0.5035*[5,6]    = [3.5105, 4.5105]

### 6.7 Explanation in words

- Token 1 can only use itself, so its output is its own value vector.
- Token 2 blends its own value with token 1's. The larger score for itself (0.7071 vs 0.0)
  puts more weight on... TODO: finish this sentence in your own words, and say what changes if the
  mask is removed.
- Token 3 can see the whole prefix, and its own score is the largest, so its output is a mixture
  of all three values. TODO: explain why its output differs from an unmasked calculation.

### 6.8 Check against the code

`tests/test_attention.py::test_hand_worked_example_matches_report` asserts these values with
`src.attention.scaled_dot_product_attention`.

---

## 7. Results

### 7.1 Loss curves (Figure 6)
TODO: insert `fig06_loss_curves.png`. Report observations (start, end, gap between train and
validation) and separate them from interpretation.

### 7.2 Generation samples (Figure 8)
TODO: include at least five samples from `outputs/samples/`. State which prompts you used and the
temperature.

### 7.3 Attention inspection (Figure 7)
TODO: insert `fig07_attention_trained.png`. Describe what one head attends to. Use the text
output from `experiments.attention_visualization`.

### 7.4 Evaluation
TODO: validation loss, perplexity (exp of loss), and the causality check result from
`outputs/logs/evaluation.json`.

---

## 8. Ablation study (Figure 9)

TODO: insert `outputs/ablation/ablation_results.md` as a table and a short plot if you like.

For each sweep (depth, heads, context length, position method) write:
- **Observation:** "validation loss went from X to Y" (numbers only).
- **Interpretation:** what you think caused it, and what evidence supports that.

Mention that each sweep changed one factor. Note any confound (for example, a larger context
changes the amount of data per window).

---

## 9. Failure analysis

TODO: at least two failure modes or limitations with evidence and likely causes. Candidates to
check: repetitive output at low temperature, overfitting (train-val gap widening), poor long-range
coherence due to the 128-character context, and the character-level vocabulary producing long
sequences.

---

## 10. Engineering packaging

TODO: explain the CLI (`src.generate`), the REST API (`src.api`, `/generate`, `/health`), how a
new user reproduces the results (README and GUIDE), and the Colab notebook. Include the Render URL
and one curl or browser screenshot.

---

## 11. Conclusion

TODO: summarise what your experiments show about Transformer-based causal language modelling,
keeping observation separate from interpretation. End with one concrete limitation and one next
step.

---

## Appendix: sources and attribution

- Vaswani et al., "Attention Is All You Need", 2017.
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019.
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers...", 2019.
- Xiong et al., "On Layer Normalization in the Transformer Architecture", 2020 (pre-LN).
- Karpathy, char-rnn repository (Tiny Shakespeare corpus).
- TODO: any tutorial or code you adapted, and what you changed.
