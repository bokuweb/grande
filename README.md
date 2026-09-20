# grande

A System One style decision model runtime in Rust. State in, typed questions
in, calibrated probabilities out, **one forward pass, no text generation**.

TypeSafe's [Jev](https://docs.typesafe.ai/) defines the contract; the
architecture follows Archer Hume's reconstruction and
[jaredpalmer/kev](https://github.com/jaredpalmer/kev): a shared state prefix,
one isolated branch per question, a readout at each branch's answer position.
grande targets Japanese, Gemma 4, quantized local inference, and (later) the
browser. Design notes live in `life/idea/local-jev`.

**Demo:** https://bokuweb.github.io/grande/ (WebGPU; nothing leaves the browser).
Gemma 4 E2B or E4B zero-shot on grande's own wgpu engine with a 25k-token
vocabulary: 1.2 GB / 2.5 GB streamed from the Hub once and cached in the
browser, 5 questions over a 90-token state in ~1.1 s (E2B) / ~2.5 s (E4B)
on an M4.

## Status

- [x] `grande-core`: TypeSafe-shaped request/response, renderer (label and
      pointer layouts), label readout, pointer head math, temperature
      scaling, ECE / Brier / NLL. No I/O, no backend dependency.
- [x] `grande-llama`: llama.cpp backend. Prefix decoded once into seq 0,
      every branch gets the prefix by `llama_memory_seq_cp` (zero copy) and
      all branches are decoded in one batch. Logits or hidden states at
      requested positions only.
- [x] `grande` CLI: `probe` (packed / separate / check, `--head`), `jglue`,
      `serve`, `bench`, `render`, `tokens`, `pieces`, `meta`.
- [x] pointer-head weights loader (`head.safetensors`, no deps).
- [x] `grande-server`: axum, `POST /v1/systemone`, `/separate`, `/permute`,
      `GET /v1/models`, `/health`, bearer auth, 422 `detail[]`, `X-Grande-*` headers.
- [x] `grande-eval`: JGLUE JNLI / JCommonsenseQA, temperature fit on the
      even half, accuracy / NLL / Brier / ECE / confident-error rate,
      `--permute` flip rate.
- [x] `python/grande_train`: LoRA + pointer head with the same layout
      (token-for-token parity with `grande render` verified), merge → GGUF →
      served by the Rust runtime. Smoke-tested on Gemma 3 270M (MPS).
- [x] trained 270M heads (3k / 12k records, 18 and 12 layers), vocab-pruned
      to 323 / 256 MB.
- [ ] trained Gemma 4 base weights (E2B base is 10 GB bf16; needs more than a
      16 GB laptop or a rented GPU)
- [x] `grande mechanism`: isolation, packed vs separate, boundary forgery.
- [x] browser demo (`web/`): `grande-core` as wasm + transformers.js on
      WebGPU, Gemma 3 270M / 1B and Gemma 4 E2B ONNX.
- [x] trained pointer head in the browser: `tools/export_browser.py` (ONNX
      q8, pruned embedding rows, id map), served from a GitHub release.
- [x] `grande suite` (kev-style frozen suites), `tools/http_eval.py`
      (any `/v1/systemone` server), `tools/prune_vocab.py`.
- [x] unified KV cache; resident prefix across requests over the same state.
- [x] state cache: the KV of every state seen is kept serialized (RAM LRU,
      `--state-cache-dir` for disk), so coming back to a document is a 2–10 ms
      restore instead of a prefill, across requests and restarts.
- [x] browser: state decoded once and resident, questions continue from its
      KV cache (`shared` mode); no state re-reading.
- [x] `grande-wgpu`: our own Gemma 3 / Gemma 4 forward pass in WGSL — state
      + every branch in one block-causal pass, native (Metal / Vulkan) and
      WebGPU from the same kernels. Parity with llama.cpp on the trained 270M
      and on E2B / E4B Q4_0; `grande --model <checkpoint dir>` and the
      `gemma-4-e2b-wgpu-ja` / `gemma-4-e4b-wgpu-ja` browser models (1.2 GB /
      2.5 GB after vocabulary pruning).
- [x] `grande jglue --shots N` (few-shot from the train split), `grande
      features` + `tools/train_head.py`: a pointer head on the frozen,
      quantized E2B / E4B read through the zero-shot chat prompt
      (`gemma_label_pointer`), trained on the engine's own hidden states.
- [ ] IIA test, permutation flip rate on a JGLUE sample
      and on E2B Q4_0; `grande --model <checkpoint dir>` and the
      `gemma-4-e2b-wgpu-ja` browser model (1.2 GB after vocabulary pruning).
- [x] `--orders N`: every Choice / Noul asked under N option orders in the
      same pass, logits averaged (position bias out); `order_spread` in the
      diagnostics. Permutation flip rate on JGLUE below.
- [x] two-stage Choice: more options than the label readout can letter
      (52) are asked in groups, the top of each group re-asked together in a
      second pass; banking77 (77-way) scored on kev's suite.
- [x] state rendered as `path: value` lines (`ticket.messages[0].text: …`),
      so questions can point at nested fields and conversation arrays the
      way TypeSafe's docs do; Python renderer mirrors it.
- [x] `grande iia`: kev's IIA test (log-odds shift from one irrelevant
      option) on JGLUE JCQA and kev's suite.

## First numbers (2026-09-19, M-series Mac, Metal)

Gemma 4 E2B it, Q4_0, zero-shot label readout, Japanese support ticket with 5
questions (`examples/ticket-ja.json`):

```
queue             packed [0.9953 0.0042 0.0004]  separate [0.9953 0.0043 0.0004]  Δmax 2.1e-5
escalate          packed [0.9769 0.0231]         separate [0.9770 0.0230]         Δmax 1.0e-5
urgency           packed [0.0194 0.0148 0.9658]  separate [0.0194 0.0148 0.9658]  Δmax 7.0e-5
refund_requested  packed [0.0368 0.9632]         separate [0.0368 0.9632]         Δmax 0
churn_risk        packed [0.9870 0.0130]         separate [0.9869 0.0131]         Δmax 2.8e-5
packed 839 ms (1 pass) vs separate 1625 ms (5 passes); max |Δp| = 7.0e-5
```

Packing does not change the answers (Q4 + fp16 noise level; kev reports
`4e-6` in fp32). Candidate mass (share of next-token probability on the
option letters) is 0.99+ on every question.

Isolation probe (`examples/isolation-ja.json`): a secret written only inside
a sibling question's text gives `P(合言葉は青い象) = 0.098`; the same secret
placed in the state gives `0.996`. Branches do not see each other.

Sliding-window attention: Gemma 4's SWA layers isolate correctly with the
default iSWA cache (`--swa-full false`), no full cache needed.

### JGLUE, zero-shot (Gemma 4 E2B it Q4_0, label readout, valid split)

Temperature fitted on even-indexed records, reported on odd-indexed ones.
Prompts are jev_local's, so the numbers compare across runtimes.

| task | n (test) | accuracy | ECE raw → scaled | NLL raw → scaled | T | p≥0.9 error rate raw → scaled | ms / record |
|---|---|---|---|---|---|---|---|
| JNLI (3-way) | 1,217 | 0.614 | 0.252 → **0.088** | 1.255 → 0.949 | 2.81 | 0.41 → 0.00 | 754 |
| JCommonsenseQA (5-way) | 559 | 0.853 | 0.044 → 0.046 | 0.447 → 0.438 | 1.19 | 0.04 → 0.02 | 702 |

The instruct model is badly overconfident on NLI (mean confidence 0.86 at
61% accuracy; 41% of its ≥0.9 answers are wrong) and one temperature
removes most of it. Commonsense QA is already calibrated. Timings were taken
while other GPU jobs ran; see `grande bench` for clean numbers.

Mechanism tests (`grande mechanism`): isolation sibling 0.098 / absent 0.098 /
state 0.996, packed vs separate 3.5e-4, forged delimiters add no options.

IIA (`grande iia`, kev's test): one irrelevant option is appended to a
3–10-way Choice ("紫 — 紫という色", "税金 — 無関係：四半期の税務申告", …)
and the log-odds between the original top-2 options are compared before
and after. A model that reads the options against the state should not
move; a letter-position or "pick the middle" heuristic does.

| Choices | n | mean \|Δ log-odds\| | p90 | argmax flips | mass on the distractor |
|---|---|---|---|---|---|
| JCQA valid, E2B zero-shot | 250 | 0.54 | 1.20 | 3.6% | 2.4% |
| JCQA valid, `--orders 3` | 250 | 0.47 | 1.03 | 4.4% | 1.5% |
| kev suite (agnews, mnli), E2B zero-shot | 160 | 0.46 | 1.10 | 8.1% | 1.2% |
| kev-0.5b (trained, kev's own number) | – | 0.13 | 0.34 | – | – |

The distractor itself is almost never chosen, but its presence moves the
odds between the real options by ~0.5 nats on average, four times kev's
trained readout. Order averaging trims the shift a little and removes most
of its sign (mean shift +0.39 → +0.14: unaveraged, the runner-up loses more
than the top option when a sixth line appears). This is a zero-shot
letter readout's weakness, not the packing's — the pointer head is what
should close it, once a Gemma 4 head is trained.

## Training loop

```bash
cd python && uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install torch transformers peft safetensors numpy gguf sentencepiece
python -m grande_train.train --base unsloth/gemma-3-270m \
  --jnli ../.cache/jglue/jnli-train.jsonl --jcqa ../.cache/jglue/jcommonsenseqa-train.jsonl \
  --n-per-source 1500 --epochs 2 --out ../runs/grande-270m
python -m grande_train.merge --base unsloth/gemma-3-270m --run ../runs/grande-270m --out ../runs/grande-270m/merged
PYTHONPATH=/path/to/llama.cpp python /path/to/llama.cpp/convert_hf_to_gguf.py ../runs/grande-270m/merged \
  --outfile ../runs/grande-270m/grande-270m-f16.gguf --outtype f16
cd .. && ./target/release/grande jglue --model runs/grande-270m/grande-270m-f16.gguf \
  --head runs/grande-270m/head.safetensors --task jnli --out runs/eval-270m-jnli
```

The renderer is defined twice (Rust for serving, Python for training) on
purpose; `grande render` dumps token ids and `grande_train.render.check_parity`
compares, so drift is caught before a model is trained on the wrong bytes.

## Comparison with kev, reflex, Jev

See [docs/comparison.md](docs/comparison.md). Short version, same M4:

- Japanese (JGLUE): grande E2B zero-shot JNLI 0.614 / JCQA 0.853; kev-0.5b
  0.450 / 0.577; a 270M grande head trained on 12k records **0.710 / 0.710**
  at 73–77 ms per record, and a 12-layer vocab-pruned 256 MB version
  0.685 / 0.630 at 26–34 ms.
- kev's English suite (identical questions): kev 0.797, Jev 0.808, grande
  E2B zero-shot 0.678 (banking77, two-stage, 0.575).
- Browser, same 5-question Japanese ticket: grande E2B 2.8 s cold / 2.5 s
  with the state resident (was 4.2 s re-reading the state per question) and
  all 5 right; reflex 0.8B 6.7 s cold / 3.5 s warm and 3 of 5 wrong.
- Vocabulary pruning cuts E2B Q4_0 from 2,841 MB to 1,247 MB with no JGLUE
  accuracy change (`tools/prune_vocab.py`); Q3_K_M on top reaches 1,181 MB
  at −6 pts JCQA, Q2_K collapses. The same token set takes the browser
  E2B from 2.8 GB to 1.2 GB (wgpu engine) and the ONNX export from 3.4 GB
  to 1.5 GB (`tools/prune_onnx_vocab.py`).
- Idle M4, 12 questions over a 500-token state: E2B Q4_0 1.94 s cold /
  1.01 s with the state resident; 270M 168 / 86 ms. A 2,000-token state
  that was seen before comes back from the state cache in 2–4 ms (RAM) or
  4–11 ms (file), so its request costs the same as a resident one.

## Usage

```bash
# Gemma 4 E2B it, Q4_0 (~2.8 GB)
mkdir -p models && curl -L -o models/gemma-4-E2B-it-Q4_0.gguf \
  https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_0.gguf

cargo build --release            # Metal on macOS; --features cuda / vulkan elsewhere
./target/release/grande probe --model models/gemma-4-E2B-it-Q4_0.gguf \
  --request examples/ticket-ja.json --mode check
```

`--mode packed` prints the TypeSafe-shaped response; `--mode check` runs
packed and per-question passes and reports the largest probability
difference.

```bash
./target/release/grande serve --model models/gemma-4-E2B-it-Q4_0.gguf \
  --state-cache-mb 512 --state-cache-dir .cache/states
```

`--baseline` (probe, serve, jglue) turns on contextual calibration: every
question is also asked over a content-free state (`N/A` by default, or the
text you pass) and the option logits that yields — the model's prior over
the options — are subtracted from the live ones before the softmax (Zhao et
al. 2021). It depends on the question alone, so it is cached per rendered
branch: a fixed question set over changing states pays one extra pass. The
raw readout is `logits + baseline`; both are in the diagnostics.

What it does and does not fix (E2B Q4_0, JGLUE valid, first 200; 100
calibration / 100 test):

| | JNLI acc | JNLI ECE raw → T | JNLI NLL raw | JCQA acc | JCQA ECE raw → T |
|---|---|---|---|---|---|
| raw | 0.540 | 0.326 → 0.063 (T 2.64) | 1.364 | **0.860** | 0.067 → 0.078 |
| `--baseline` | 0.540 | **0.174** → 0.083 (T 1.42) | **1.028** | 0.770 | 0.097 → 0.077 |

It removes a bias that lives in the *labels*: the letter pull on a fixed
3-way question (JNLI ECE halves with no temperature fitted) and the yes /
no lean on a Noul the state says nothing about (E2B reads "is the password
blue elephant" over a memo about a meeting as 2–18% *yes* across seven such
questions; with the baseline all are ≤ 0.3%, and the letter *A* stops
pulling a Choice). It does not know the difference between "no evidence"
and "evidence against", so a Noul the model already answers weakly is
flattened too (資料は共有済み 0.998 → 0.747), and when the *options* carry
the content — JCQA's five answer strings — the content-free prior is part
of the answer and subtracting it costs 9 points. Use it for a fixed
question set with letter labels or yes / no over documents; leave it off
for commonsense-style choices. A fitted temperature is still the better
tool where a labelled split exists.

The server keeps the current state resident and every other state it has
seen serialized: an LRU in RAM (`--state-cache-mb`, 512 MB ≈ 55k tokens of
E2B state) and, with `--state-cache-dir`, a file per state keyed by model
and token ids, so a document answered before a restart still restores in
milliseconds. The `X-Grande-State` response header says which path a
request took: `resident`, `ram`, `disk` or `decoded`. `grande bench`
reports the restore (`restored_ms`, `restored_from`) next to cold and warm.

## Layout

```
crates/grande-core    types, renderer, readout, math, calibration, Backend trait
crates/grande-llama   llama-cpp-2 backend
crates/grande-wgpu    wgpu backend: Gemma 3 / Gemma 4 in WGSL, native and WebGPU
crates/grande-web     wasm surface for the browser demo (+ the wgpu engine)
crates/grande-cli     `grande` binary
examples/             request files
```

The `Backend` trait is two methods (`tokenize`, `evaluate`) plus token
lookups. There is no `generate`; an engine only has to implement prefix +
isolated branches → rows at positions.

## The wgpu engine

`crates/grande-wgpu` implements that contract without a third-party
runtime: nine WGSL kernels (embedding, RMSNorm, matmul and gated MLP over
f16 / Q8_0 / Q4_0 weights, a skinny logits projection, q/k norm + RoPE,
block-causal attention, and the two Gemma 4 per-layer-embedding steps) and
one command buffer per request.
Every token carries a (position, sequence) pair — prefix tokens are
sequence 0, branch b's tokens are sequence b restarting at the prefix length
— and the attention kernel's visibility rule (prefix or own sequence,
causal, sliding window) *is* the isolation. No KV cache to manage, no
padding, no runtime constraint on batching a continuation.

```bash
# package a trained run for it (config, tokenizer, head, f16 safetensors)
python tools/export_wgpu.py --run runs/grande-270m-12k --out web/models/grande-270m-ja-wgpu
# a directory instead of a GGUF selects the engine
./target/release/grande probe --model web/models/grande-270m-ja-wgpu \
  --head runs/grande-270m-12k/head.safetensors --request examples/ticket-ja.json

# Gemma 4 E2B / E4B: repack the llama.cpp GGUF (Q4_0 / Q8_0 codes kept as they are)
python tools/export_wgpu_gguf.py models/gemma-4-E2B-it-Q4_0.gguf --out models/gemma-4-e2b-wgpu-q4
./target/release/grande probe --model models/gemma-4-e2b-wgpu-q4 --request examples/ticket-ja.json
python tools/export_wgpu_gguf.py models/gemma-4-E4B-it-Q4_0.gguf --out models/gemma-4-e4b-wgpu-q4
```

Two model families run on the same kernels. Gemma 3 (the trained 270M, f16
from the HF checkpoint) and Gemma 4 E2B / E4B: sliding layers at head_dim
256 and global layers at 512 with partial RoPE, the last 20 (E2B) / 18
(E4B) layers attending over the last own-K/V layer of their type,
RMS-normalized V, double-wide MLPs, the per-layer token embeddings (the
1.3 GB table is gathered on the host per request, not uploaded), per-layer
output scalars and softcapped logits. `Config` describes every layer and
`Config::tensors()` is the catalogue both loaders fill. Grouped-query
attention: one K/V head on the 270M and E2B, two on E4B (8 query heads,
4 per group); the attention workgroup handles the rows of one K/V head.

All match llama.cpp: the 270M to ~5e-4 in probability on the ticket and
isolation examples, E2B on the same Q4_0 weights to ≤ 5e-4 (ticket,
isolation, contract), E4B to ≤ 7e-5 (ticket, isolation).
`crates/grande-wgpu/tests/reference.rs` checks the shaders against a
plain-Rust forward on random models of all three shapes with f16, Q8 and
Q4 weights.

E2B on this engine, M4, one pass and no resident state: the 5-question
ticket (313 tokens) in 1.16 s and the 8-question contract (639 tokens) in
2.5 s in the browser, against 2.5–2.8 s and 3.1–5.2 s on the ONNX Runtime
path (`shared` mode, state resident); natively 1.47 s for the ticket where
llama.cpp takes ~1.0 s (its Metal matmuls use simdgroup matrix ops, WGSL
has none — the quantized matmul runs at ~1 TFLOPS).

Where the 270M stands (M4, measured while another training job held the
GPU, so only the ratios mean anything): in the browser the same checkpoint
answers the 5-question ticket in 0.17–0.55 s on this engine against
0.86–1.35 s on the ONNX Runtime path run interleaved with it, and the 8-question contract
in 0.41–0.74 s against 2.9–4.0 s; natively it is still 2–3× behind
llama.cpp's Metal kernels at ~500–900 tokens and ~10× at 2,500 (the
attention over a long prefix is where the kernel is weakest). Native stays
on llama.cpp; the browser is where this engine pays off.
## Order averaging and two-stage Choice

Position bias is real on a zero-shot label readout: the same Noul reads
differently with `true` listed first or second (on the ticket example,
`churn_risk` moves by 0.33 between orders). `--orders N` (probe, jglue,
suite, serve) asks every Choice and Noul under N option orders — the
identity, its rotations, then their reversals — as extra branches of the
same pass and averages the option logits, so the state is still read once.
`order_spread` in the diagnostics (`X-Grande-Order-Spread-Max` from the
server) is the largest |Δp| any option showed between two orders, i.e. the
bias that was averaged out. Score levels are never permuted.

JGLUE valid, first 300 records, E2B Q4_0, label readout, no temperature
(`grande jglue --limit 300 [--orders 3]`):

| | JNLI acc | JNLI ECE | JNLI NLL | JCQA acc | JCQA ECE | JCQA NLL | ms / record |
|---|---|---|---|---|---|---|---|
| 1 order | 0.530 | 0.341 | 1.493 | 0.857 | 0.051 | 0.471 | ~400 / 319 |
| 3 orders | **0.567** | **0.275** | **1.184** | **0.860** | **0.041** | **0.385** | 980 / 720 |

Under 3 orders (`--permute 3`) the JNLI argmax flips on 14% of records and
the distributions differ by 0.34 in L1 on average, which is what the
average removes. The extra branches cost their tokens: on the wgpu engine
(native, E2B pruned) the 5-question ticket goes from 315 to 542 tokens and
0.80 to 1.36 s at 3 orders; on llama.cpp the JGLUE runs above took
2–2.5× per record.

A Choice with more options than the label readout can letter (52) no
longer errors: the options are asked in groups of at most 52 in the first
pass, the top options of every group (as many as fit under 52 together)
are asked once more against each other in a second pass, and the answer's
probabilities are the second pass's, with the eliminated options at 0.
`two_stage` in the diagnostics lists the finalists. The browser demo runs
both through the same code (`grande-core::plan` via wasm: **Orders** in the
page; a 77-way Choice takes two passes there too). On kev's banking77
(77 intents) this scores 0.575 zero-shot against kev 0.800 / Jev 0.838;
the group stage loses the gold intent in 6 of 80 records, the rest are
second-stage misses (see [docs/comparison.md](docs/comparison.md)). The
pointer readout has no cap and never needs this.

## Raising E2B / E4B accuracy

What moves the zero-shot numbers, measured on the same first 400 records of
JGLUE valid (E2B Q4_0, label readout, no temperature: JNLI 0.575, JCQA
0.855). `grande jglue --limit 400 [--orders N] [--shots N]`:

| | JNLI acc | JNLI ECE | JCQA acc | JCQA ECE | ms / record |
|---|---|---|---|---|---|
| E2B Q4_0 | 0.575 | 0.279 | 0.855 | 0.046 | 750 |
| E2B Q4_0, `--orders 3` / `5` | 0.598 | 0.239 | 0.873 | 0.021 | 1,990 / 890 |
| E2B Q4_0, `--shots 3` / `6` / `12` | 0.562 / 0.570 / 0.583 | 0.339 / 0.302 / 0.249 | 0.830 (3) | 0.060 | 860–1,400 |
| E2B Q8_0 | 0.585 | 0.414 | | | 370 |
| E4B Q4_0 | 0.595 | 0.366 | **0.932** | **0.017** | 735 / 466 |
| E4B Q4_0, `--shots 3` / **`6`** / `12` | 0.723 / **0.775** / 0.715 | 0.077 / **0.055** / 0.136 | 0.887 (6) | 0.038 | 1,360 / 1,770 / 2,920 |
| E4B Q4_0, `--shots 6 --orders 3` | 0.693 | 0.155 | | | 2,710 |

`--shots N` puts N labelled train-split records (balanced over labels,
fixed seed; `--shots-seed` picks another draw) in front of every state
under an `例` key, as a resident prefix would in production. What the
table says:

- Quantization and order averaging are worth a point or two on JNLI;
  order averaging is the cheap win on JCQA (+1.8, ECE halves).
- **E2B cannot use few-shot examples**: 3, 6 or 12 of them leave JNLI at
  the zero-shot level and cost 2.5 points on JCQA. **E4B can**: 6 examples
  take JNLI from 0.595 to 0.775 and the raw ECE from 0.37 to 0.055 — the
  instruct model stops being overconfident without a fitted temperature.
  Another draw of 6 gives 0.752; 12 is worse (0.715) and 3 is 0.723.
- Few-shot is for the skill task only: on JCQA (knowledge) it costs E4B
  4.5 points, and combining it with order averaging loses 8 points on JNLI.
- JCQA is decided by the backbone: E4B zero-shot 0.932 against E2B 0.855.

So without training: E4B, `--shots 6` on NLI-shaped questions, zero-shot
elsewhere, `--orders` when the question is a fixed letter choice.

For scale: chance is 0.33 on JNLI and 0.20 on JCQA; encoders fine-tuned on
the JGLUE train splits reach about 0.90 on JNLI and 0.80–0.90 on JCQA, and
the JGLUE paper's human estimate is about 0.93 / 0.98. E4B zero-shot on
JCQA is past the fine-tuned encoders; JNLI with few-shot is still some 15
points under them, which is what the trained head below is for.

### Frozen backbone, trained head

The zero-shot label readout is the weak part on JNLI (E4B barely beats
E2B), and a trained readout is what lifted the 270M to 0.71. The same can
be done on E2B / E4B without touching the weights: `Renderer::Label` with
`pointer` on renders the **same chat prompt** as the zero-shot layout but
marks the last token of every option line (`OptEnd`) and the model-turn
position (`Decide`), so a pointer head reads the served, quantized model's
own hidden states. No LoRA, no re-export, no `<unused*>` delimiters the
base model never saw; the head is two `d → 256` affine maps in
`head.safetensors` and `--head` picks the layout from its metadata.

```bash
# hidden states of the train split, 2 shuffled option orders per record (F16 safetensors, ~0.5 GB / 6k)
./target/release/grande features --model models/gemma-4-E2B-it-Q4_0.gguf --task jnli --limit 6000 --out runs/feat/e2b-jnli.safetensors
./target/release/grande features --model models/gemma-4-E2B-it-Q4_0.gguf --task jcqa --limit 4000 --out runs/feat/e2b-jcqa.safetensors
# the head trains in seconds on the cached rows (last 10% of records held out for model selection)
python tools/train_head.py --features runs/feat/e2b-jnli.safetensors runs/feat/e2b-jcqa.safetensors --out runs/head-e2b
./target/release/grande jglue --model models/gemma-4-E2B-it-Q4_0.gguf --head runs/head-e2b/head.safetensors --task jnli --out runs/eval-head-jnli
```

Extraction runs at ~0.7 s per record on the M4 (the second order reuses
the resident state); 6k JNLI records are 148 MB of F16 rows and the head
trains in under 30 s. Same first 400 valid records as the table above,
E2B Q4_0, no temperature:

| readout | JNLI acc | JNLI ECE | JNLI NLL | JCQA acc | JCQA ECE | ms / record |
|---|---|---|---|---|---|---|
| zero-shot letters | 0.575 | 0.279 | 1.36 | 0.855 | 0.046 | 750 |
| head trained on 6k JNLI / 4k JCQA (one per task) | **0.848** | **0.056** | 0.44 | 0.835 | 0.037 | 421 / 280 |
| one head trained on both | 0.708 | 0.216 | | 0.843 | 0.040 | |
| for reference: E4B `--shots 6` | 0.775 | 0.055 | | | | 1,770 |
| for reference: 270M + LoRA + head, 12k records | 0.710 | 0.160 | | 0.710 | 0.050 | 75 |

- **JNLI 0.575 → 0.848 with the weights untouched**, and the readout is
  calibrated on its own (ECE 0.056; 6% of the ≥0.9 answers are wrong,
  against 41% zero-shot). It beats E4B with few-shot by 7 points at a
  quarter of the latency (no example prefix; hidden rows are cheaper
  than a vocabulary projection) and the 270M LoRA model by 14.
- JCQA does not want a head: 0.835 against 0.855 zero-shot. The options
  carry the content there and the letter logits already read it; the
  head can only lose information. Use the zero-shot readout (or E4B) for
  knowledge questions and a head for skill questions.
- One head for both tasks costs JNLI 14 points: a single 256-wide
  pointer cannot serve fixed-label NLI and content-bearing options at
  once. Heads are per question family; `--head` is per request, so a
  server can hold one per question type.

The head-only path is the cheap half of the training story: no HF
checkpoint, no bf16 memory, no re-quantization, and the wgpu engine
serves it unchanged since the hidden rows already come out of
`Want::Hidden`. LoRA on the backbone remains the next step for the
last points; distillation only makes sense with a teacher stronger
than E4B.

## Notes

- Gemma's tokenizer splits digits (`10` → 2 tokens, `254` → 3), so the label
  readout uses `A–Z a–z` (52 options max). The pointer readout has no such
  limit and no vocabulary head.
- Gemma 4 turn markers are `<|turn>` / `<turn|>` (ids 105 / 106), not
  `<start_of_turn>`; thinking is off unless a system turn carries `<|think|>`.
- User text is tokenized with control-token surface forms broken up
  (`<unused0>` → `<‌unused0>`), so option delimiters cannot be forged.
