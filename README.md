# grande

A System One style decision model runtime in Rust. State in, typed questions
in, calibrated probabilities out, **one forward pass, no text generation**.

TypeSafe's [Jev](https://docs.typesafe.ai/) defines the contract; the
architecture follows Archer Hume's reconstruction and
[jaredpalmer/kev](https://github.com/jaredpalmer/kev): a shared state prefix,
one isolated branch per question, a readout at each branch's answer position.
grande targets Japanese, Gemma 4, quantized local inference, and (later) the
browser. Design notes live in `life/idea/local-jev`.

**Demo:** https://bokuweb.github.io/grande/ (WebGPU; pick a model, load, run — nothing leaves the browser)

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
- [ ] trained Gemma 4 base weights (E2B base is 10 GB bf16; needs more than a
      16 GB laptop or a rented GPU)
- [x] `grande mechanism`: isolation, packed vs separate, boundary forgery.
- [x] browser demo (`web/`): `grande-core` as wasm + transformers.js on
      WebGPU, Gemma 3 270M / 1B and Gemma 4 E2B ONNX.
- [x] `grande suite` (kev-style frozen suites), `tools/http_eval.py`
      (any `/v1/systemone` server), `tools/prune_vocab.py`.
- [x] unified KV cache; resident prefix across requests over the same state.
- [x] state cache: the KV of every state seen is kept serialized (RAM LRU,
      `--state-cache-dir` for disk), so coming back to a document is a 2–10 ms
      restore instead of a prefill, across requests and restarts.
- [x] browser: state decoded once and resident, questions continue from its
      KV cache (`shared` mode); no state re-reading.
- [ ] IIA test, permutation flip rate on a JGLUE sample

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
  0.450 / 0.577; a 270M grande head trained on 3k records 0.540 / 0.670 at
  47–77 ms per record.
- kev's English suite (identical questions): kev 0.797, Jev 0.808, grande
  E2B zero-shot 0.677 (banking77 excluded).
- Browser, same 5-question Japanese ticket: grande E2B 2.8 s cold / 2.5 s
  with the state resident (was 4.2 s re-reading the state per question) and
  all 5 right; reflex 0.8B 6.7 s cold / 3.5 s warm and 3 of 5 wrong.
- Vocabulary pruning cuts E2B Q4_0 from 2,841 MB to 1,273 MB with no JGLUE
  accuracy change (`tools/prune_vocab.py`); Q3_K_M on top reaches 1,181 MB
  at −6 pts JCQA, Q2_K collapses.
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
crates/grande-cli     `grande` binary
examples/             request files
```

The `Backend` trait is two methods (`tokenize`, `evaluate`) plus token
lookups. There is no `generate`; a wgpu or Metal engine only has to implement
prefix + isolated branches → rows at positions.

## Notes

- Gemma's tokenizer splits digits (`10` → 2 tokens, `254` → 3), so the label
  readout uses `A–Z a–z` (52 options max). The pointer readout has no such
  limit and no vocabulary head.
- Gemma 4 turn markers are `<|turn>` / `<turn|>` (ids 105 / 106), not
  `<start_of_turn>`; thinking is off unless a system turn carries `<|think|>`.
- User text is tokenized with control-token surface forms broken up
  (`<unused0>` → `<‌unused0>`), so option delimiters cannot be forged.
