# Laya next to omg

2026-09-20. [Laya](https://github.com/NandhaKishorM/laya) (Convai
Innovations, Apache-2.0) is a System One model built the other way round
from omg: a small bidirectional encoder with a trained decision head,
instead of a decoder LLM read at the answer position.
[laya-mlx](https://github.com/mizorewww/laya-mlx) is its MLX port and
reports 13 ms (English) / 7 ms (multilingual) per short question on an M3
Max. This note measures the multilingual checkpoint on omg's Japanese
evals and on JevBench, says what of Laya is worth taking, and describes
its port onto omg's wgpu engine (native and browser).

## What Laya is

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
```

One sequence per question, question first, the state truncated to fit
(`max_len` 512 or 1,024 minus `head_max_len` 192 / 256 for the question).
The encoder output goes through a type embedding and two more Transformer
layers; a 1-wide scorer reads the hidden row at each `[MASK]` and a softmax
over those rows is the answer. `noul` is a two-option choice
(`false: …`, `true: …`), `score` is a choice over `level i: …` lines. No LM
head, no KV cache, no generation. A second head (`act_head`) reads the
`[CLS]` row plus (top-p, margin, entropy, k) and emits an "escalate"
probability trained against a cost. Temperature is per question type with
optional `(type, option count)` buckets. Trained with RL against proper
scoring rules ("RLCD").

| checkpoint | encoder | non-embedding params | context | state budget |
|---|---|---|---|---|
| `laya` | ModernBERT-large, 28 × 1024 | ~390M | 512 | ~320 tokens |
| `laya-multilingual` | mmBERT-base, 22 × 768, Gemma 2 256k vocabulary | ~110M (322M with the embedding) | 1,024 | ~768 tokens |

So the speed is the model: ~20× fewer FLOPs per token than Gemma 4 E2B,
and no read-out over a vocabulary. laya-mlx's own README puts MLX graph
compilation at 1.03–1.08×. Laya re-encodes the state for every question
(cost O(Q × (S + q))); omg shares the state prefix (O(S + Q × q)) and
keeps it resident, so the two cross over as states get longer and question
counts grow — but for short states Laya's constant wins outright.

## Measurements

`tools/laya_serve.py` wraps laya-mlx in omg's `/v1/systemone` shape, so
`tools/http_eval.py` and the JevBench harness run unchanged. M4 (base),
fp16, batch 1, caller wall time over localhost.

### JGLUE (laya-multilingual, valid split, same prompts and protocol as the README)

Temperature fitted on even-indexed records, reported on odd-indexed ones.
omg rows are the README's / [comparison.md](comparison.md)'s.

| task | system | n | accuracy | ECE raw → scaled | NLL raw → scaled | T | ms / record |
|---|---|---|---|---|---|---|---|
| JNLI | **laya-multilingual** | 1,217 | **0.702** | 0.207 → 0.046 | 1.188 → 0.745 | 2.83 | **18** |
| JNLI | omg E2B zero-shot | 1,217 | 0.614 | 0.252 → 0.088 | 1.255 → 0.949 | 2.81 | 754 |
| JNLI (first 400) | laya-multilingual | 400 | 0.670 | 0.242 | 1.251 | – | 18 |
| JNLI (first 400) | omg E2B + trained pointer head | 400 | **0.848** | 0.056 | – | – | 421 |
| JCQA | laya-multilingual | 559 | 0.551 | 0.041 → 0.043 | 1.177 → 1.173 | 1.23 | **21** |
| JCQA | omg E2B zero-shot | 559 | **0.853** | 0.044 → 0.046 | 0.447 → 0.438 | 1.19 | 702 |
| JCQA (first 400) | omg E4B zero-shot | 400 | **0.932** | – | – | – | – |

Laya beats zero-shot E2B on JNLI — a reading task, exactly what its RL
training targets — at 40× the speed, and is as overconfident as E2B before
scaling (22 % of its ≥0.9 answers wrong). omg's trained head on the
frozen E2B is still 15 points ahead. On JCommonsenseQA, which needs world
knowledge rather than reading, the 110M encoder is near chance-plus
(0.55 on 5-way) against 0.85 / 0.93 for E2B / E4B.

### JevBench public items (231, English)

Same harness and ranking as [jevbench.md](jevbench.md). The English `laya`
row through our shim reproduces the published one (95.8 / 70.8 / 35.1
vs 95.8 / 69.4 / 35.1: one standard-tier item flips at fp16), which
validates the pipeline. Summaries: `docs/jevbench/laya-{en,ml}-*-summary.json`.

| system | easy 48 | std 72 | hard 111 | I(pub) | p50 / p95 |
|---|---|---|---|---|---|
| omg, Gemma 4 E4B it Q4_0, zero-shot | 100.0 | 94.4 | 51.4 | 77.6 (#10) | 1.47 s / 17.5 s |
| omg, Gemma 4 E2B it Q4_0 pruned, zero-shot | 97.9 | 73.6 | 34.2 | 61.9 (#17) | – |
| Laya English (ours, laya-mlx fp16) | 95.8 | 70.8 | 35.1 | 60.8 | 0.05 s / 0.2 s |
| Laya English (published) | 95.8 | 69.4 | 35.1 | 60.3 (#18) | – |
| **laya-multilingual** (ours) | 89.6 | **40.3** | 33.3 | **47.0** (#21 of 23) | 0.07 s / 0.29 s |

The multilingual checkpoint is not the English one with more languages: on
the standard tier it drops to 40 % (adequacy 2/12, ordinal 3/12, policy
3/12) and lands below every published system except Needle. Its estimated
JevBench score with the official formulas is 53.6 (I 47 / C 34 / S 87 /
K 59) against 59 for omg E4B — the speed term cannot carry it.

## What to take from Laya

1. **Temperature per (question type, option count).** Laya's
   `temperature[3]` + `temperature_by_options{"choice:5": …}`; omg has
   one `engine.temperature`. The JNLI / JCQA fits above (2.8 vs 1.2 for
   the same model) say the bucket matters; this is the cheapest gain on
   JevBench's calibration term. → `omg-core::calibration` bucket fit,
   `serve --temperature-file`.
2. **An "escalate" signal.** Laya trains `act_head` on (top-p, margin,
   entropy, k) against a cost of being wrong. omg measures the
   confident-error rate but does not expose anything per answer; a
   rule on the same features in the diagnostics / `X-Omg-*` headers is
   most of the value without training.
3. **A no-growth test.** laya-mlx checks 100 repeated calls for zero
   active-memory growth; `omg-wgpu` should have the same after the
   stranded-resident-state fix (#15).
4. **Question-token cache.** Laya keeps the tokenized question prefix in a
   128-entry LRU. Negligible natively; worth measuring in the browser.

Not worth taking: shape specialization / `pad_to_multiple` (omg-wgpu
already runs on a capacity-sized workspace), the question-first layout with
the state cut from the right (Laya's Banking77 0.425 comes from this;
omg's two-stage Choice handles 77 options), fp16 unquantized weights
(bandwidth-bound; Q4 is faster on the same GPU).

## Laya on omg's wgpu engine

Done (this branch): `crates/omg-wgpu/src/laya.rs` runs the mmBERT /
ModernBERT encoder and the two decision-head layers in WGSL, natively
(Metal / Vulkan / DX12) and on WebGPU, and the scorer + act head on the host.

- Every question is one sequence; all of a request's sequences are packed
  into one token stream and one command buffer, isolated by a
  bidirectional attention mask (`attention_bi.wgsl`: same sequence, and
  |Δposition| ≤ 64 on the local layers). New kernels: LayerNorm, an
  elementwise bias / ReLU / residual tail, in-place RoPE, the attention;
  `matmul`, the gated matmul (with an exact-GELU variant) and the
  embedding gather are the Gemma ones.
- `prompt.rs` is laya's `build_sequence` token for token (budgets,
  truncations, Python's `json.dumps` spacing for object states); the
  tokenizer is the caller's — the HF tokenizer natively, transformers.js in
  the browser — so both pack the same ids (checked: identical token counts
  and probabilities to 1e-6).
- `Decider` (omg-core) is the trait one level above `Backend`
  (`answer`, `answer_many`, `distributions`); `Engine<B>` and `LayaBackend`
  both implement it, and `omg serve` / `omg probe` pick the Laya
  runtime when `--model` is a Laya directory (Hub snapshot or export).
- `tools/export_laya.py` writes the browser layout: Q8 weights, the
  vocabulary pruned to a corpus's tokens plus their merge closure and every
  single-character token (256k → 56k; tokenizer.json rewritten), 180 MB.
  `web/engine.js` lists it as `laya-multilingual-wgpu`; the page's E2B
  default is unchanged.

Parity with laya-mlx (fp16), M4:

| | laya-mlx | wgpu f16 (snapshot) | wgpu Q8 + 56k vocab (export) |
|---|---|---|---|
| JNLI first 400: accuracy / ECE / NLL | 0.670 / 0.242 / 1.251 | 0.6725 / 0.244 / 1.249 | 0.6625 / 0.243 / 1.247 |
| JCQA first 400 | 0.5225 / 0.066 / 1.212 | 0.5225 / 0.071 / 1.212 | 0.5275 / 0.041 / 1.214 |
| JNLI-style request, 3 questions, 175 tokens | 24 ms | 70 ms | 44 ms native, **51 ms browser** |
| ticket-ja, 5 questions, 637 tokens | 78 ms | 211 ms | – |
| contract preset, 8 questions, 2,361 tokens (browser) | – | – | 765 ms |

The answers match; the speed is within 2× of MLX and stays there. Where the
time goes (`OMG_WGPU_PROFILE=1 omg probe --repeat 20`, Q8 export,
3-question request, 175 tokens): 41.6 ms of GPU time over 291 dispatches,
the four encoder matmuls 25 ms (`mm_qkv` 0.44 ms a layer = 1.4 TFLOPS on
Q8, `mm_o` 1.2 TFLOPS), attention 4.4 ms, everything else under 2 ms.
Two compute-shaped tiles were tried against the shipped kernel (f32 tiles
staged k-major in workgroup memory, no unpacking, register blocks unrolled
by hand) and both lost on the M4:

| matmul tile (Q8 export) | 3 q / 175 tok | 5 q / 637 tok |
|---|---|---|
| **matmul.wgsl** (32 × 128, 4 × 8 per thread, f16-pair tiles) | **43 ms** | **140 ms** |
| 64 × 64, 8 × 8 per thread, f32 tiles | 55 ms | 184 ms |
| 32 × 32, 4 × 4 per thread, f32 tiles | 61 ms | 208 ms |
| laya-mlx (MLX steel GEMM, fp16) | 24 ms | 78 ms |

f32 staging costs more workgroup-memory bandwidth than the unpacks save
(the same result the Gemma kernel's notes record), so a plain WGSL tile
sits at ~1.3 TFLOPS here; MLX's 2× comes from the simdgroup matrix units,
which WGSL cannot reach in shipping wgpu / browsers.

Two things that were not the matmul (2026-09-21, same request):

- `attention_bi` staged each 8-key tile's positions and sequence ids and
  had one invocation scan all 128 (query, key) pairs serially to decide
  whether the tile could be skipped, with the other 127 waiting. A packed
  sequence's rows are contiguous and its positions count from its first
  row, so the keys a tile of queries can see are an arithmetic range
  (`[max(lo, tok0 − window), min(hi, tok0 + ROWS + window))` of the
  sequences' span) and a pair's mask is its sequence ids plus the row
  distance. The kernel now loops over that range with four barriers per
  tile instead of five and no per-tile scan: 1.2 → 0.55 ms a layer in the
  profile (2.2×). 16 × 8 stays the tile at HD 64 (16 × 16 is 1.6× slower,
  32 × 8 and 8 × 16 within the noise).
- The residual adds now ride in the LayerNorm that follows them
  (`layernorm.wgsl` gets `residual` / `has_rbias` / `in_place`): an
  encoder layer is 8 dispatches instead of 10, the tail (final norm, copy,
  type embedding, add) is one in-place norm plus a residual gather, and a
  head layer's `add_l2` carries the next layer's `norm1`. 247 → 198
  dispatches. Outputs are unchanged to 1e-7 on the three example requests.

Still open, roughly in order: the N = 768 matmuls (`mm_o`, `mm_wo`,
`mm_l2`: 42 workgroups on 10 cores, ~30 % below the wide layers'
throughput — a 32 × 64 tile or split-K), RoPE into the attention's Q / K
staging (22 dispatches of ~50 µs), and the browser's tokenizer time.
Cross-request
batching (`answer_many` packs queued requests into one pass, `X-Omg-Batch`)
is in, and worth +25 % throughput at 16 concurrent clients (23 → 29 req/s)
— the pass is compute-bound, so it cannot do more. What would move the
number now is fewer tokens, not a faster kernel: Laya re-reads the state
for every question, so an 8-question request over a 245-token state is
2,361 tokens (765 ms in the browser) where omg's shared prefix would
read it once.

## Rerun

```bash
# omg's own runtime (wgpu), from the Hub snapshot laya-mlx downloads
hf download aac6fef/laya-multilingual-mlx
./target/release/omg serve --model ~/.cache/huggingface/hub/models--aac6fef--laya-multilingual-mlx/snapshots/<rev> --port 8791
python tools/http_eval.py --url http://127.0.0.1:8791 --model grande-latest --task jnli

# browser export (Q8, pruned vocabulary) into the demo's model directory
python tools/export_laya.py <snapshot> --dtype q8 --out web/models/laya-multilingual-wgpu \
    --corpus .cache/jglue/*-train.jsonl .cache/kev/*.jsonl examples/*.json

# the MLX reference (Apple Silicon only), behind the same API, for parity checks
uv venv -p 3.13 .laya && uv pip install -p .laya/bin/python git+https://github.com/mizorewww/laya-mlx
.laya/bin/python tools/laya_serve.py --model aac6fef/laya-multilingual-mlx --port 8790
python -m jevbench.cli run --adapter typesafe --endpoint http://127.0.0.1:8790 --key-env '' --model laya-multilingual --reserve-usd 0 --cap-usd 0 --tasks datasets/public/original.jsonl --results …
```
