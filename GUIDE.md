# GUIDE: what to do, step by step

This guide takes you from the unzipped project to a trained model, a tested repository, a Colab
run, a deployed API, and the evidence for your report. Work through the parts in order.

Estimated total time: 1 day of work (most of it is waiting for training and writing the report).

---

## Part 0. What is in this zip

| Path | What it is | Do you edit it? |
|---|---|---|
| `src/` | The Transformer, attention, training, generation, API code | Read and understand it. You must explain it in the demo. |
| `tests/` | Automated checks | Run them. Do not edit them to pass. |
| `configs/tiny_gpt.yaml` | All hyper-parameters | Yes, if your hardware needs smaller settings. Document changes. |
| `experiments/` | Figures, ablations, evaluation scripts | Run them. |
| `notebooks/MiniGPT_Colab.ipynb` | The whole pipeline as a Colab notebook | Set your GitHub URL. |
| `report/REPORT.md` | Report template with the hand calculation already worked out | Fill in every `TODO`. |
| `README.md` | Repo documentation (setup, commands, limitations) | Update the example output with your own. |
| `Dockerfile`, `requirements.txt`, `pytest.ini`, `.gitignore` | Packaging | Leave as is. |
| `outputs/`, `checkpoints/` | Empty until you run the code | Filled by your runs. |

**Important:** the zip does not contain any trained model or results. Those come from your runs.
That is deliberate, because the assignment requires the numbers to come from your own training.

---

## Part 1. Accounts and tools you need

1. **Python 3.10 or newer**: check with `python --version` (Windows: `py --version`).
2. **Git**: check with `git --version`. Install from https://git-scm.com if missing.
3. **GitHub account**: https://github.com (free). Required for the Colab link and for the grader.
4. **Google account** for Google Colab: https://colab.research.google.com
5. **Render account** for hosting: https://render.com (free tier is enough for the demo).
6. **Medium account** for the report article (the assignment asks for a Medium article).
7. A screen recorder (OBS Studio, Loom, or the Windows/macOS built-in tool).

---

## Part 2. Local setup

Open a terminal inside the unzipped folder (the folder that contains `README.md`).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `pip install` is slow, that is normal: PyTorch is a large download (about 800 MB with CUDA
libraries on Windows and Linux). On a machine without a GPU, install the CPU build of PyTorch
first from https://pytorch.org (choose "CPU"), then run `pip install -r requirements.txt`.

**Checkpoint:** `python -c "import torch, fastapi; print(torch.__version__)"` prints a version
number with no error.

---

## Part 3. Run the tests (Milestones 1-3 evidence)

```bash
pytest -v
```

**Expected:** every test shows `PASSED`. Save the output as evidence (screenshot or
`pytest -v > outputs/logs/pytest.txt`).

If something fails, do not edit the test. Read the failure message. The most common causes:

- `ModuleNotFoundError: src`: run the command from the project root, the folder with `pytest.ini`.
- A shape error: check that the input has the expected dimensions and that `d_model` is divisible
  by `n_heads`.

---

## Part 4. Overfit gate (Milestone 4)

This checks that gradients and the target shift work before you spend hours on full training.

```bash
python -m src.train --overfit
```

The first run downloads Tiny Shakespeare (about 1.1 MB) to `data/tinyshakespeare.txt`. A line
like this is printed every 250 steps and at the end:

```
step      1 | lr 3.00e-03 | train 4.2xxx | val 4.2xxx | ...
step    300 | lr 3.00e-03 | train 1.xxxx | val 1.xxxx | ...
```

**Pass condition:** the loss falls substantially from step 1 to the last step (the assignment asks
for a clear drop). `summary.json` in `outputs/logs/overfit/` records `first_train_loss` and
`final_train_loss`. Keep this as evidence.

If the loss does not fall: check that `dropout` is 0 in overfit mode (the script sets it), that
`pytest tests/test_overfit.py` passes, and that you did not change the target shifting in
`src/dataset.py`.

---

## Part 5. Full training (Milestone 5)

```bash
python -m src.train --config configs/tiny_gpt.yaml --run-name mini_gpt
```

Timing guide (the default is about 0.8M parameters, 5,000 steps, batch 32, context 128):

| Hardware | Approximate time |
|---|---|
| Colab T4 GPU | 10-25 minutes |
| Laptop CPU (8+ cores) | 1-3 hours |

It prints progress every 250 steps. When it finishes, check:

- `checkpoints/mini_gpt_best.pt` exists (about 3-4 MB).
- `outputs/logs/mini_gpt/metrics.csv` has about 20 rows.
- `outputs/logs/mini_gpt/summary.json` contains `parameters`, `best_val_loss`, `train_time_s`, and
  `hardware`. Copy these numbers into your report (Section 5 of the template).
- `outputs/samples/mini_gpt_samples.md` contains five samples.

**If training is too slow:** reduce the model rather than skipping evidence. Example:

```bash
python -m src.train --run-name mini_small --set train.max_steps=2000 model.n_layers=2 model.d_model=64
```

Write down exactly which values you changed and why. The assignment explicitly allows this if you
document it (Section 11 of the handout).

**If you run out of GPU memory:** lower `train.batch_size` (for example 16).

---

## Part 6. Generate text, run the API, and deploy

### 6a. Command-line generation (Figure 10 evidence)

```bash
python -m src.generate --prompt "ROMEO:" --max-new-tokens 80 --temperature 0.8
python -m src.generate --prompt "The future of" --max-new-tokens 80 --temperature 0.8
python -m src.generate --prompt "To be, or not to be" --greedy
```

Screenshot one terminal run. It must show the prompt, the output, the parameter count, the context
length, and the temperature.

### 6b. Run the REST API locally

```bash
uvicorn src.api:app --port 8000
```

In a second terminal:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" \
     -d '{"prompt": "ROMEO:", "max_new_tokens": 80, "temperature": 0.8}'
```

On Windows PowerShell, use `Invoke-RestMethod` instead of `curl`, or open
`http://localhost:8000/docs` in a browser and click "Try it out". The browser route is easiest
and gives a good screenshot for the report.

### 6c. Deploy on Render

1. Push the repository to GitHub first (Part 8).
2. **Make sure the checkpoint is in git.** `.gitignore` allows `checkpoints/mini_gpt_best.pt`
   by name. Check with `git status`. If it is missing, run `git add -f checkpoints/mini_gpt_best.pt`.
   The file is small (a few MB), which is fine for this assignment.
3. On https://dashboard.render.com click **New -> Web Service** and connect your GitHub repository.
4. Settings:
   - **Runtime:** Docker (Render detects the `Dockerfile`).
   - **Instance type:** Free (or Starter if the free tier runs out of memory).
   - **Health check path:** `/health`.
   - **Environment variables:** none are required. Optional: `MINIGPT_CKPT=checkpoints/mini_gpt_best.pt`.
5. Click **Create Web Service** and wait for the build (5-15 minutes the first time).
6. Test the public URL:

```bash
curl https://YOUR-SERVICE.onrender.com/health
curl -X POST https://YOUR-SERVICE.onrender.com/generate -H "Content-Type: application/json" \
     -d '{"prompt": "ROMEO:", "max_new_tokens": 80}'
```

**Notes:** the free tier sleeps after about 15 minutes without traffic, so the first request after
a pause takes about 30-60 seconds. Open the URL once before your demo. If the build runs out of
memory, switch to a larger instance or remove unneeded packages from `requirements.txt`.

---

## Part 7. Experiments, figures, and evaluation (Milestone 6)

Run these in order. Each command prints where its outputs are saved.

```bash
python -m experiments.static_figures              # Figures 2, 3, 4, 5
python -m experiments.plot_curves --run mini_gpt   # Figure 6
python -m experiments.attention_visualization --prompt "ROMEO:"   # Figure 7
python -m experiments.evaluate                    # perplexity, causality check, Figure 8 samples
python -m experiments.ablation --steps 1500       # Figure 9 (about 12 short runs)
```

Ablation time: about 5-15 minutes on a GPU, 1-3 hours on CPU. To shorten it, use
`--steps 800` or run only some sweeps: `python -m experiments.ablation --steps 800 --only depth heads`.

**Check these results after each command:**

- `outputs/figures/` contains `fig02_causal_mask.png`, `fig04_positional_encoding.png`,
  `fig06_loss_curves.png`, `fig07_attention_trained.png`.
- `outputs/logs/shape_trace.txt` shows the tensor shapes through every stage.
- `outputs/logs/evaluation.json` says `"causality_check": true`. If it says false, stop and fix the
  mask before you write anything. It should never be false.
- `outputs/ablation/ablation_results.md` contains the table for Figure 9.

**Required ablation (Section 13 of the handout):** at least three. The script runs four:
depth, heads, context length, and position method. Greedy vs temperature is qualitative: compare
`outputs/samples/eval_samples.md` (the greedy and sampled blocks for each prompt) and describe
repetition and diversity in your report.

---

## Part 8. Put the project on GitHub

```bash
git init
git add .
git commit -m "MiniGPT assignment: implementation, tests, experiments"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/transformer-gpt-assignment.git
git push -u origin main
```

Before pushing, check that no secrets or large files are included:

```bash
git status
git check-ignore -v data/tinyshakespeare.txt     # should print a match
```

Make the repository public (Settings -> General -> Change visibility), or add your instructor as a
collaborator, as the handout requires.

---

## Part 9. Colab run (demo "from a clean environment")

1. Open https://colab.research.google.com -> **File -> Upload notebook** -> choose
   `notebooks/MiniGPT_Colab.ipynb` from the project.
2. In the first code cell, replace `YOUR-USERNAME` in `REPO_URL` with your GitHub user name.
3. **Runtime -> Change runtime type -> T4 GPU -> Save.**
4. **Runtime -> Run all.** Watch for errors. Each section prints its results.
5. At the end, `results_bundle.zip` downloads to your computer. Keep it as your evidence archive.

If you prefer not to use GitHub: skip the first cell, uncomment cell 2b, and upload the zip.

---

## Part 10. The report and figures (what to produce)

Open `report/REPORT.md`. It already contains:

- The **hand calculation** (Section 6 of the handout). The numbers are worked out for you. You
  must still understand each step, because it is graded separately, and you should redo it on
  paper once yourself before the demo.
- The required structure (10 sections).
- A figure checklist mapping each required figure to the file that produces it.

Fill every `TODO` with your own numbers from `outputs/logs/` and `outputs/ablation/`.
**Do not copy numbers from this guide or from the README example. Use your run.**

Figure mapping:

| Required figure | File you produce | Command |
|---|---|---|
| 1. Hand-calculated attention | `report/REPORT.md` Section 6 (write it on paper, photograph it, and add the photo) | manual |
| 2. Causal mask | `outputs/figures/fig02_causal_mask.png` | `experiments.static_figures` |
| 3. Multi-head shape trace | `outputs/logs/shape_trace.txt` | `experiments.static_figures` |
| 4. Positional encoding | `outputs/figures/fig04_positional_encoding.png` | `experiments.static_figures` |
| 5. Model summary and parameter count | `outputs/logs/model_summary.txt` | `experiments.static_figures` |
| 6. Train vs validation loss | `outputs/figures/fig06_loss_curves.png` | `experiments.plot_curves` |
| 7. Trained attention | `outputs/figures/fig07_attention_trained.png` | `experiments.attention_visualization` |
| 8. Generation samples | `outputs/samples/*.md` | `src.train` and `experiments.evaluate` |
| 9. Ablation table/plot | `outputs/ablation/ablation_results.md` | `experiments.ablation` |
| 10. CLI or API call | terminal or browser screenshot | `src.generate` / `/docs` |

Also include a **model architecture diagram**. Draw it yourself (draw.io, Excalidraw, or a
hand sketch) based on the pipeline in the handout, Section 1.

**Medium article:** copy the report sections into Medium. Keep the equations, the hand calculation,
and the figures. Cite Vaswani et al. (2017), Radford et al. (GPT-2, 2019), Devlin et al. (BERT,
2019), and the Tiny Shakespeare source. Add the GitHub and Render links at the top.

---

## Part 11. Demo (5-10 minutes, live or recorded)

Follow the demo script in the handout, Section 17. Suggested order:

1. Show the repo tree and `src/attention.py`. Explain Q, K, V, the scale, and the mask.
2. Run `pytest -v` and show it passing.
3. Show `outputs/logs/mini_gpt/summary.json` (parameters, hardware, time).
4. Show `fig06_loss_curves.png` and explain what you see (train vs val gap, when it flattens).
5. Run `python -m src.generate --prompt "ROMEO:"`.
6. Show `fig07_attention_trained.png` and say what one head looks at.
7. Open the Render URL `/docs` and make one live request.
8. State one ablation result and one limitation.

**Be ready to explain every model component** (the tests can serve as your guide). Typical oral
questions: why scale by sqrt(d_k), what happens to a row of the attention matrix after the mask,
why target t is token t+1, why pre-LN, and why BERT differs from GPT in its attention mask.

**Recording:** OBS -> Start Recording. Record your terminal full screen at 1080p. Run one command
at a time with a pause so each output is visible. Keep it between 5 and 10 minutes.

---

## Part 12. Final submission checklist

Tick each item before you submit.

- [ ] Repository is public or shared with the instructor, and the link works in a private window.
- [ ] `README.md` setup commands work from a clean clone (try in a new folder).
- [ ] `pytest -v` passes; output saved as evidence.
- [ ] Overfit gate loss fell substantially; `summary.json` saved.
- [ ] `checkpoints/mini_gpt_best.pt` exists and loads (`python -m src.generate` works).
- [ ] `evaluation.json` shows `"causality_check": true`.
- [ ] Hand calculation is in the report and redone by you on paper.
- [ ] Causal mask figure, shape trace, PE figure, model summary, loss curve, trained attention figure, samples, ablation table, CLI/API screenshot are all included (10 items in Part 10).
- [ ] At least five generation samples from fixed prompts are included.
- [ ] At least three ablations are reported with observation and interpretation kept separate.
- [ ] Final parameter count and training configuration (hardware, batch size, steps, LR, time) are documented.
- [ ] Render URL responds to `/health` and `/generate`.
- [ ] The Colab notebook runs from top to bottom with your `REPO_URL`.
- [ ] Failure analysis: at least two failure modes or limitations, with likely causes.
- [ ] All sources and any adapted code are cited.
- [ ] No API keys, tokens, or private data are committed (`git log -p | grep -i key` finds nothing suspicious).
- [ ] Medium article published (or draft link ready) with a link to the repo and the Render URL.
- [ ] Screen recording is 5-10 minutes and saved.

---

## Part 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root (where `pytest.ini` is). |
| `torch` install fails or is huge | Install the CPU build first from pytorch.org, then `pip install -r requirements.txt`. |
| `urllib` error downloading the corpus | Download `input.txt` from the URL in `configs/tiny_gpt.yaml` manually and save it as `data/tinyshakespeare.txt`. |
| Loss is `nan` | Lower `train.lr` to `3.0e-4`, keep `grad_clip` at 1.0. |
| Samples are only one repeated character | Training too short. Run more steps or check that the loss has fallen below about 2.0. |
| `checkpoint not found` from the API | Run training first, or set `MINIGPT_CKPT` to the path. |
| Colab "No GPU" | Runtime -> Change runtime type -> T4 GPU, then Run all again. |
| Render build fails on memory | Use a larger instance, or install the CPU-only torch wheel in the Dockerfile (already the default). |
| Ablation is too slow | Pass `--steps 800` or `--only depth heads`. Report the reduced setting. |
| `KeyError` loading a checkpoint | The checkpoint came from a different code version. Re-train with the current code. |

---

## Part 14. What this guide cannot do for you

Be honest about these in your report and demo:

- The trained checkpoint, the loss numbers, the samples, the ablation results, and the timings are
  produced only by running the code on your machine. Nothing in this zip contains results.
- The hand calculation in `report/REPORT.md` is worked out for you as a reference. The assignment
  grades your own hand calculation, so redo it yourself.
- The written explanations, interpretations, and oral answers must be your own. The template gives
  structure and prompts, not conclusions.
- Experiments on a different corpus, bigger models, or a different hardware setup need their own
  documented runs.
