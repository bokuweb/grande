# Ruri v3 next to grande

2026-09-21. [Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-130m)
(Nagoya University, Apache-2.0) is a family of Japanese sentence-embedding
models on ModernBERT-Ja: four sizes, mean pooling, 8,192-token context, a
100k-piece Japanese vocabulary. Measured exactly as
[multilingual-e5-small](e5.md) was — frozen vectors + heads trained in
seconds, one generic (state, option) head, and the whole encoder
fine-tuned — with the general prefix (`""`; `トピック: ` moved the 30m
heads by −1…+3 points, see below). Same JGLUE protocol: temperature on the
even-indexed valid records, everything reported on the odd-indexed ones
(n = 1,217 / 559), plus the first 400 rows raw.

| model | params | non-embedding | d | layers | JMTEB avg (published) |
|---|---|---|---|---|---|
| ruri-v3-30m | 37M | 10M | 256 | 10 | 74.51 |
| ruri-v3-70m | 70M | 31M | 384 | 13 | 75.48 |
| ruri-v3-130m | 132M | 80M | 512 | 19 | 76.55 |
| ruri-v3-310m | 315M | 236M | 768 | 25 | 77.24 |
| multilingual-e5-small (for scale) | 118M | 21.6M | 384 | 12 | – |

Rerun: `tools/ruri.sh` (features, heads and the generic head for every
size, then the fine-tunes; `frozen` for the first part only),
`tools/e5_table.py` prints the rows.
Latency is PyTorch fp16 on MPS at batch 1, tokenisation included, which
is launch-bound (a 25-layer model at 24 ms and a 19-layer one at 31 ms
are the same number: the run-to-run noise of that runtime); on grande's
wgpu engine the 30m / 70m would sit where e5-small does, ~10 ms a request.

## JNLI (odd half, n = 1,217)

| system | accuracy | ECE raw → scaled | NLL raw → scaled | T | first 400 acc / ECE | ms / record |
|---|---|---|---|---|---|---|
| ruri-v3-30m `cos` zero-shot | 0.601 | – | – | – | 0.618 / – | 7 |
| ruri-v3-30m `both-mlp` | 0.754 | 0.025 → 0.031 | 0.631 → 0.622 | 1.27 | 0.757 / 0.057 | 7 |
| ruri-v3-30m `generic` | 0.712 | 0.068 → 0.025 | 0.746 → 0.707 | 1.52 | 0.765 / 0.051 | 7 |
| ruri-v3-30m `ft` (whole encoder) | **0.875** | 0.027 → 0.030 | 0.385 → 0.373 | 1.26 | **0.853** / 0.037 | 7 |
| ruri-v3-70m `pair-lr` | 0.758 | 0.020 → 0.029 | 0.600 → 0.600 | 1.09 | 0.738 / 0.054 | 15 |
| ruri-v3-70m `both-mlp` | 0.758 | 0.036 → 0.030 | 0.630 → 0.618 | 1.30 | 0.755 / 0.043 | 15 |
| ruri-v3-70m `generic` | 0.744 | 0.043 → 0.021 | 0.655 → 0.642 | 1.31 | 0.760 / 0.062 | 15 |
| ruri-v3-130m `pair-lr` | **0.810** | 0.029 → 0.034 | 0.501 → 0.501 | 1.09 | 0.805 / 0.044 | 31 |
| ruri-v3-130m `both-mlp` | 0.771 | 0.034 → 0.024 | 0.563 → 0.553 | 1.19 | 0.772 / 0.049 | 31 |
| ruri-v3-130m `generic` | 0.785 | 0.038 → 0.032 | 0.554 → 0.543 | 1.32 | 0.782 / 0.046 | 31 |
| ruri-v3-310m `pair-lr` | 0.841 | 0.018 → 0.024 | 0.404 → 0.406 | 1.09 | 0.843 / 0.037 | 24 |
| ruri-v3-310m `both-mlp` | **0.842** | 0.023 → 0.045 | 0.446 → 0.446 | 1.22 | 0.845 / 0.025 | 24 |
| ruri-v3-310m `generic` | 0.807 | 0.021 → 0.025 | 0.501 → 0.497 | 1.26 | 0.805 / 0.044 | 24 |
| e5-small `both-mlp` ([e5.md](e5.md)) | 0.759 | 0.025 → 0.042 | 0.587 → 0.583 | 1.22 | 0.743 / 0.064 | 20 |
| e5-small `ft` | 0.831 | 0.026 → 0.020 | 0.448 → 0.448 | 0.94 | 0.835 / 0.050 | 22 |
| laya-multilingual | 0.702 | 0.207 → 0.046 | 1.188 → 0.745 | 2.83 | 0.670 / 0.242 | 18 |
| grande E2B zero-shot | 0.614 | 0.252 → 0.088 | 1.255 → 0.949 | 2.81 | 0.575 / – | 754 |
| grande E2B + trained pointer head | – | | | | **0.848** / 0.056 | 421 |

## JCommonsenseQA (odd half, n = 559)

| system | accuracy | ECE raw → scaled | NLL raw → scaled | T | first 400 acc / ECE | ms / record |
|---|---|---|---|---|---|---|
| ruri-v3-30m `cos` zero-shot | 0.753 | 0.543 → 0.412 | 1.567 → 1.172 | 0.08 | 0.715 / 0.505 | 14 |
| ruri-v3-30m `both-mlp` | 0.776 | 0.041 → 0.044 | 0.614 → 0.603 | 1.15 | 0.762 / 0.051 | 14 |
| ruri-v3-30m `generic` | 0.750 | 0.102 → 0.039 | 0.793 → 0.675 | 1.86 | 0.762 / 0.117 | 14 |
| ruri-v3-70m `cos` zero-shot | 0.787 | 0.578 → 0.452 | 1.567 → 1.166 | 0.08 | 0.772 / 0.563 | 19 |
| ruri-v3-70m `both-mlp` | 0.825 | 0.051 → 0.050 | 0.537 → 0.517 | 1.27 | 0.802 / 0.066 | 19 |
| ruri-v3-70m `generic` | 0.807 | 0.076 → 0.036 | 0.649 → 0.563 | 1.67 | 0.785 / 0.091 | 19 |
| ruri-v3-130m `cos` zero-shot | 0.826 | 0.616 → 0.484 | 1.564 → 1.133 | 0.08 | 0.818 / 0.608 | 25 |
| ruri-v3-130m `both-mlp` | **0.857** | 0.049 → 0.037 | 0.423 → 0.421 | 0.93 | 0.855 / 0.055 | 25 |
| ruri-v3-130m `generic` | 0.837 | 0.074 → 0.045 | 0.563 → 0.468 | 1.75 | 0.843 / 0.080 | 25 |
| ruri-v3-310m `cos` zero-shot | **0.864** | 0.654 → 0.520 | 1.563 → 1.122 | 0.08 | 0.860 / 0.650 | 33 |
| ruri-v3-310m `both-mlp` | **0.907** | 0.033 → 0.045 | 0.307 → 0.304 | 1.40 | 0.890 / 0.025 | 33 |
| ruri-v3-310m `generic` | 0.852 | 0.033 → 0.026 | 0.419 → 0.407 | 1.30 | 0.850 / 0.045 | 33 |
| e5-small `both-mlp` | 0.662 | 0.051 → 0.027 | 0.916 → 0.895 | 1.23 | 0.657 / 0.065 | 19 |
| e5-small `ft` | 0.671 | 0.075 → 0.048 | 0.850 → 0.830 | 1.23 | 0.635 / 0.095 | 19 |
| laya-multilingual | 0.551 | 0.041 → 0.043 | 1.177 → 1.173 | 1.23 | 0.5225 / 0.066 | 21 |
| grande E2B zero-shot | 0.853 | 0.044 → 0.046 | 0.447 → 0.438 | 1.19 | 0.855 / 0.046 | 702 |
| grande E4B zero-shot | – | | | | **0.932** / 0.017 | 735 |

Prefix (ruri-v3-30m, frozen heads): `トピック: ` instead of `""` gives
JNLI `cos` 0.612 / `sep-mlp` 0.704 / `pair-lr` 0.740 / `both-mlp` 0.754
(vs 0.601 / 0.673 / 0.728 / 0.754) and JCQA `both-mlp` 0.776 (same) — a
point or three on the separate-embedding heads, nothing on the best one.

## Reading

- **Japanese pre-training is what e5 was missing on JCQA.** The 37M
  ruri-v3-30m — 10M parameters without its embedding table, half of
  e5-small's — lands at 0.776 with a frozen head where e5-small stops at
  0.665 even fine-tuned; cos(question, choice) alone is 0.753. The
  commonsense of JCommonsenseQA is Japanese lexical knowledge (what a
  マザーボード is), which a model trained on Japanese pairs has in its
  geometry and a multilingual one has to share across 100 languages.
- **Frozen ruri-v3-310m + a 1M head passes the 2B LLM on JCQA**: 0.907
  against zero-shot E2B's 0.853 (first 400: 0.890 vs 0.855; E4B 0.932),
  and cos alone (0.864) already does. On JNLI it ties the frozen E2B +
  pointer head (0.845 vs 0.848 on the first 400) — with a linear head on
  one pair vector (`pair-lr` 0.841, 4k parameters), at 1/17 of the
  latency even in PyTorch. 130m is 5 points behind on both tasks, 70m 10.
- **The joint sequence wins as the model grows.** For e5, separate
  embeddings beat the pair sequence; for Ruri v3 it flips at 70m
  (`pair-lr` ≥ `sep-mlp` by 7 points at 70m, 12 at 130m, 13 at 310m):
  ModernBERT-Ja's 8k context and training on longer Japanese text make one
  sequence with two sentences in it something its pooling can read. That
  is good news for the `generic` head, which must read the state as one
  sequence: it gives up 2–4 points against the best task head instead of
  e5's 4–5, and at 310m it is 0.807 / 0.852.
- **Calibration** comes out at ECE 0.02–0.05 raw for every trained head,
  as with e5; the zero-shot `cos` needs T = 0.08 and stays at 0.4–0.5.
- **Fine-tuning** (whole encoder, 3 epochs, the e5 recipe; `tools/ruri.sh`,
  smallest first, the larger sizes still running when this was written):
  ruri-v3-30m fine-tuned on JNLI reaches **0.875** (first 400: 0.853) —
  a 37M model above the frozen E2B + pointer head (0.848) and e5-small
  fine-tuned (0.831), at 7 ms in PyTorch. The remaining rows land in this
  table as the queue finishes.

## What the head cannot do

The (state, option) head reads two vectors — the rendered state and an
option's description — and nothing else. A question's `instructions` are
never in its input, so two questions with the same options over the same
state get the same answer: in `examples/ticket-ja.json` the three noul
questions (escalate, refund_requested, churn_risk) come out at one
identical probability, and the contract preset's five nouls likewise. It
is a classifier for question families it was trained on (JNLI-shaped
pairs, JCQA-shaped choices, whatever else is in its training set), not a
System One that reads a new question — which is why these backends are
not offered in the browser demo and stay a native option for known
families. Reading the instructions takes a cross-encoder in Laya's shape
(instructions, marked options and the state in one sequence, a scorer on
the marker rows) trained on many question types; Ruri v3 is a ModernBERT
and could be trained that way (`tools/e5_finetune.py`'s JNLI mode is
already a one-task cross-encoder), the open question being how far 33
distinct instruction strings in `.cache/distill` generalise.

## On grande's wgpu engine

Done (this branch): `ruri-v3-130m-wgpu` and `ruri-v3-310m-wgpu`, on the
same `e5.rs` engine — it gained a ModernBERT path (pre-LN, RoPE with the
local / global layers, GeGLU, no biases: the Laya encoder's kernels, the
residual stream in one buffer, a final LayerNorm before the pool) beside
the BERT one, selected by the export's `model_type`. `tools/export_e5.py
--model cl-nagoya/ruri-v3-<size> --prefix ""` with the size's generic
head: Q8, `mlp.Wi` split into value / gate, the 102k-piece vocabulary
pruned to 86k (the corpus uses most of a Japanese vocabulary, so little
goes), 146 MB / 314 MB. `grande serve | probe --model <export>` pick them
by `config.json`; the browser loader (`kind: "e5"` in `web/engine.js`)
runs both (release `ruri-v1`), unlisted for the reason above.

Parity with the PyTorch shim (fp16 MPS, full vocabulary), first 400 rows
through `tools/http_eval.py`, generic head, T 1.5 (130m) / 1.3 (310m):

| | PyTorch shim | wgpu Q8, 86k vocabulary |
|---|---|---|
| ruri-v3-130m JNLI: acc / ECE / NLL | 0.783 / 0.059 / 0.522 | 0.790 / 0.064 / 0.525 |
| ruri-v3-130m JCQA | 0.843 / 0.027 / 0.468 | 0.845 / 0.038 / 0.465 |
| ruri-v3-310m JNLI | 0.805 / 0.046 / 0.506 | 0.815 / 0.036 / 0.508 |
| ruri-v3-310m JCQA | 0.850 / 0.028 / 0.408 | 0.855 / 0.022 / 0.407 |

Speed (`grande probe --repeat 20`, `examples/ticket-ja.json`, 5 questions,
M4, GPU otherwise idle; GPU time from `GRANDE_WGPU_PROFILE=1`):

| model | request | GPU time / dispatches | in the browser (Chromium, WebGPU) |
|---|---|---|---|
| multilingual-e5-small (12 × 384) | 13 ms | 11 ms / 142 | 34 ms |
| ruri-v3-130m (19 × 512) | **19 ms** | 18 ms / 198 | – |
| ruri-v3-310m (25 × 768) | **41 ms** | 41 ms / 258 | 112 ms; contract preset (8 questions, 187-token state) 197 ms (measured with a fine-tune sharing the GPU) |
| laya-multilingual (22 × 768, for scale) | 44 ms (3 questions) | 42 ms / 291 | 51 ms |
| grande E2B Q4_0 | ~700 ms | | ~1.5 s |

So ruri-v3-310m + head answers a 5-question Japanese request in 41 ms
with JNLI 0.81 / JCQA 0.85 — the zero-shot E2B's accuracy on JCQA and 20
points above it on JNLI, at 1/17 of its time and 1/4 of its download; the
130m gives up 2 points on each for half the time and 146 MB. Same caveat
as e5: at d = 512 / 768 the matmul tile is better filled than at 384, but
the pass is still dispatch-bound (258 dispatches for 41 ms), and a
fused / small-N tile is where the next 2× is.

## As a grande backend

`tools/e5_serve.py --model cl-nagoya/ruri-v3-<size> --prefix ""` with the
size's `runs/ruri/<size>/generic/head.pt` serves any of the four behind
`/v1/systemone` (the shim is model-agnostic); the 130m and 310m run on the
wgpu engine as above, and the 30m / 70m would export the same way
(`tools/export_e5.py`) if a smaller tier is wanted: the 30m is 10M
non-embedding parameters, e5-small's speed with JCQA 0.75 instead of 0.62.
