# grande

A System One style decision model runtime in Rust. State in, typed questions
in, calibrated probabilities out, **one forward pass, no text generation**.

TypeSafe's [Jev](https://docs.typesafe.ai/) defines the contract; the
architecture follows Archer Hume's reconstruction and
[jaredpalmer/kev](https://github.com/jaredpalmer/kev): a shared state prefix,
one isolated branch per question, a readout at each branch's answer position.
grande targets Japanese, Gemma 4, quantized local inference, and (later) the
browser. Design notes live in `life/idea/local-jev`.

## Status

- [x] `grande-core`: TypeSafe-shaped request/response, renderer (label and
      pointer layouts), label readout, pointer head math, temperature
      scaling, ECE / Brier / NLL. No I/O, no backend dependency.
- [x] `grande-llama`: llama.cpp backend. Prefix decoded once into seq 0,
      every branch gets the prefix by `llama_memory_seq_cp` (zero copy) and
      all branches are decoded in one batch. Logits or hidden states at
      requested positions only.
- [x] `grande` CLI: `probe` (packed / separate / check), `tokens`, `pieces`, `meta`.
- [ ] pointer-head weights loader (`head.safetensors`)
- [ ] `grande-server` (axum, `/v1/systemone`)
- [ ] `grande-eval` (JGLUE, isolation / permutation / IIA / boundary tests)
- [ ] fine-tuned Gemma 4 pt weights, wasm / WebGPU backend

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
