# grande vs kev vs reflex (vs Jev where numbers exist)

2026-09-19. Machine for every "here" number: Apple M4 (base, 10-core GPU), 16 GB,
macOS. grande = llama.cpp Metal; kev = PyTorch fp32 MPS; reflex browser =
transformers.js WebGPU in the same Chromium pane as grande's browser demo.

Three things are being compared and they should not be collapsed into one
score: **accuracy on Japanese** (what grande is for), **accuracy on kev's
English suite** (the only place all three of grande, kev and Jev were scored
on identical questions), and **latency** on one machine.

## 1. Accuracy, Japanese (JGLUE valid, same prompts everywhere)

| model | trained on | JNLI acc | JNLI ECE | JCQA acc | JCQA ECE | ms / record (M4) |
|---|---|---|---|---|---|---|
| **grande** Gemma 4 E2B it Q4_0, zero-shot label readout | – | **0.614** (n=1,217) | 0.252 → 0.088 (T=2.81) | **0.853** (n=559) | 0.044 → 0.046 | 700–750 |
| grande Gemma 4 E2B, **vocab-pruned** (1.27 GB) | – | 0.580 (first 400; unpruned 0.575 on the same rows) | 0.279 | 0.855 (unpruned 0.855) | 0.046 | 570–840 |
| grande Gemma 3 270M + LoRA + pointer head | 1,500 JNLI + 1,500 JCQA train, 2 ep | 0.540 (n=200) | 0.112 → 0.091 | 0.670 (n=200) | 0.074 | 47–77 |
| grande 270M + head, **12k records** (6,000 + 6,000, 1 ep) | 12k | **0.710** (n=200) | 0.160 → 0.086 | 0.710 (n=200) | 0.050 | 73–77 |
| grande 270M **12 of 18 layers** + head, 6k records, vocab-pruned (**256 MB**) | 6k | 0.685 (n=200) | 0.063 | 0.630 (n=200) | 0.064 | **26–34** |
| **kev-0.5b** (Qwen2.5-0.5B, English-trained), via its `/v1/systemone` | 6 English datasets | 0.450 (n=300) | 0.168 | 0.577 (n=300) | 0.059 | 136–234 |
| reflex | – | not run: Python engine needs CUDA; browser 0.8B below | | | | |
| jev_local (LFM2.5-1.2B) / Jev | – | not run here | | | | |

Reading: zero-shot Gemma 4 E2B is the most accurate on commonsense QA and
badly overconfident on NLI until one temperature is applied. kev is an
English model and it shows (JNLI 0.45 is near the 3-way chance of a
label-skewed set). **A 270M head trained on 12k records beats the 2B
zero-shot on JNLI (0.710 vs 0.614) at a tenth of the latency**, and a
12-layer / 256 MB version keeps 0.685 at 26–34 ms per record — kev's own
finding (data and a trained readout beat zero-shot size in-distribution)
reproduced in Japanese. JCQA still favours the bigger backbone (0.853 vs
0.71): commonsense is knowledge, NLI is a skill.

## 2. Accuracy, kev's frozen English suite (identical questions)

`evals/decision-v1/development.jsonl`, clean variant. kev and Jev numbers are
kev's published `docs/kev-vs-jev-summary.json`. grande here is the zero-shot
E2B label readout with no English tuning at all.

| task | n | kev-0.5b (trained on these sources) | Jev | grande E2B zero-shot |
|---|---|---|---|---|
| agnews (4-way) | 80 | 0.913 | 0.813 | 0.787 |
| agnews_yn (noul) | 160 | 0.925 | 0.906 | 0.831 |
| boolq (noul) | 80 | 0.788 | **0.925** | 0.838 |
| mnli (3-way) | 80 | 0.775 | 0.825 | 0.500 |
| sst5 (5 levels) | 80 | 0.575 | 0.613 | 0.388 |
| yelp (5 levels) | 80 | 0.613 | 0.650 | 0.450 |
| yelp_yn (noul) | 80 | 0.863 | 0.825 | 0.800 |
| banking77 (77-way) | 80 | 0.800 | 0.838 | 0.575 (two-stage: 2 groups of 39 / 38, 52 finalists, 4.7 s) |
| **all but banking77** | 640 | **0.797** | **0.808** | **0.678** |
| all | 720 | – | – | 0.667 |

grande ECE on this suite: 0.197 raw (no temperature), NLL 1.337 (banking77's
NLL is 4.04: in 6 of 80 records the gold intent was cut in the group stage
and gets probability 0; the other 28 misses are second-stage errors).
The nested and conversation states of the suite (`[{"role": …}]`) are
rendered as `path: value` lines since the state-flattening change; the
per-task numbers moved by at most ±1.3 points and the 640-question total
by +0.1.

Transfer (`transfer-v2` dev, clean, 560 questions; kev/Jev published numbers
are on the older transfer-v1 sample of the same sources, so only rough):

| task | grande E2B zero-shot | kev-0.5b (v1) | Jev (v1) |
|---|---|---|---|
| emotion | **0.625** | 0.413 | 0.500 |
| mmlu | 0.438 | 0.475 | 0.900 |
| qnli | 0.762 | 0.788 | 0.875 |
| tweet_offensive | 0.688 | 0.525 | 0.800 |
| sciq | 0.938 | – | – |
| paws | 0.550 | – | – |
| contrastive | 0.625 | – | – |
| all | 0.661 | 0.633 (v1, 8 tasks) | 0.823 (v1) |

Reading: untuned E2B lands between kev and Jev out of kev's training
distribution and well below both inside it. Scores (5-level) are where the
zero-shot label readout is weakest, exactly what a trained pointer head fixes
(kev: sst5 0.575 vs base zero-shot 0.313 in its own table).

## 3. Latency, same machine

All grande native numbers below were re-measured with the GPU otherwise idle
(the first pass of this document had other jobs running and was ~2× slower;
this laptop is sensitive to contention and heat, so compare within a table,
not across sessions). `cold` = state and branches evaluated; `warm` = same
state again, branches only (the prefix stays resident in the KV cache).

| runtime | model | request | tokens | cold | warm |
|---|---|---|---|---|---|
| grande native (Metal) | Gemma 3 270M f16 + head | 12 Q, 500-tok state | 1,072 | **168 ms** | **86 ms** |
| grande native | Gemma 4 E2B Q4_0 | 12 Q, 500-tok state | 1,072 | 1.94 s | 1.01 s |
| grande native | Gemma 4 E2B Q4_0 | 12 Q, 2,000-tok state | 2,620 | 4.69 s | 1.11 s |
| grande native | Gemma 4 E2B Q4_0 | 5 Q ticket | 313 | ~0.6 s | |
| grande native | Gemma 3 270M + head | JGLUE, 1 Q | ~100 | 47–77 ms | |
| kev-0.5b (PyTorch MPS fp32) | Qwen2.5-0.5B | 5 Q ticket | 261 | 211–313 ms | |
| kev-0.5b | JGLUE, 1 Q | | | 136–234 ms | |
| grande browser (WebGPU, `shared`) | Gemma 3 270M q4f16 | 8 Q contract | 639 | 1.7 s | 0.9 s |
| grande browser (`shared`) | Gemma 4 E2B q4f16 | 5 Q ticket | 313 | 2.8 s | 2.5 s |
| grande browser (`shared`) | Gemma 4 E2B q4f16 | 8 Q contract | 639 | 5.2 s | 3.1 s |
| grande browser (`batched`, state re-read per row) | Gemma 4 E2B q4f16 | 5 Q ticket | 673 | 4.6 s | |
| grande browser (`batched`) | Gemma 4 E2B q4f16 | 8 Q contract | 2,263 | 12.1 s | |
| **grande browser, trained** | grande-270m-ja q8 (211 MB) | 5 Q ticket | 571 | 449 ms | **267 ms** |
| grande browser, trained, **wgpu engine** (GPU shared with a training job; ONNX path interleaved: 0.86–1.35 s) | grande-270m-ja-wgpu f16 (320 MB) | 5 Q ticket | 240 | 0.17–0.55 s | |
| grande browser, trained, wgpu engine (same conditions; ONNX 2.9–4.0 s) | grande-270m-ja-wgpu f16 | 8 Q contract | 522 | 0.41–0.74 s | |
| grande browser, trained | grande-270m-ja q8 | 8 Q contract | 2,102 | | 923 ms |
| **grande browser, wgpu engine** (one pass, no resident state) | gemma-4-e2b-wgpu Q4_0 (2.8 GB) | 5 Q ticket | 313 | 1.16 s | |
| grande browser, wgpu engine | gemma-4-e2b-wgpu Q4_0 | 8 Q contract | 639 | 2.5 s | |
| grande native, wgpu engine | gemma-4-e2b-wgpu Q4_0 | 5 Q ticket | 313 | 1.47 s | |
| grande browser, wgpu engine, **f16-tile matmul** (pruned model; same session: previous kernels 1.18–1.42 s / 2.56–2.90 s) | gemma-4-e2b-wgpu-ja Q4_0 (1.2 GB) | 5 Q ticket / 8 Q contract | 313 / 639 | **1.02 s / 2.3 s** | |
| grande native, wgpu engine, f16-tile matmul (previous kernels 1.53 s) | gemma-4-e2b-wgpu-ja Q4_0 | 5 Q ticket | 313 | **0.88 s** (1.12 s with `GRANDE_WGPU_CHECKED=1`, naga's bounds clamps on) | |
| grande native, wgpu engine, + attention rewrite and 32-row matmul tiles | gemma-4-e2b-wgpu-ja Q4_0 | 5 Q ticket | 313 | **0.82 s** (GPU 782 ms: gate_up 393, down 202, qkv 55, o 51, attention 46, rest 35) | |
| grande browser, wgpu engine, **E4B** (2 K/V heads; machine shared with a llama.cpp build, so an upper bound) | gemma-4-e4b-wgpu-ja Q4_0 (2.5 GB) | 5 Q ticket / 8 Q contract | 313 / 674 | 2.5 s / 7–10 s | |
| grande native, wgpu engine, E4B (llama.cpp 2.2 s on the same GGUF, same conditions) | gemma-4-e4b-wgpu-ja Q4_0 | 5 Q ticket | 313 | 1.8 s | |
| grande native, wgpu engine, **resident state** | gemma-4-e2b-wgpu-ja Q4_0 | 12 Q, 500-tok state | 1,123 | 3.02 s | **1.66 s** |
| grande native, wgpu engine, resident state | gemma-4-e2b-wgpu-ja Q4_0 | 12 Q, 2,000-tok state | 2,663 | 8.1 s | **2.03 s** |
| grande browser, wgpu engine, resident state | gemma-4-e2b-wgpu-ja Q4_0 | 5 Q ticket / 8 Q contract | 313 / 674 | 1.0 s / 2.3 s | **0.77 s / 1.5 s** |
| grande native, wgpu engine, **state cache** (2,000-tok state seen before, another state in between) | gemma-4-e2b-wgpu-ja Q4_0 | 12 Q, 2,000-tok state | 2,663 | 8.2 s | RAM **2.05 s**, file 2.08 s (save 23 ms, 19 MB) |
| grande browser, wgpu engine, state cache | gemma-4-e2b-wgpu-ja Q4_0 | 8 Q contract, after a ticket request | 674 | | restored 1.50–1.75 s (resident 1.5–1.73) |
| **reflex browser** (WebGPU) | Qwen3.5-0.8B q4f16 | 5 Q ticket (same JSON) | 1,114 / 324 warm | 6.7 s | 3.5 s |
| reflex Python (published) | Qwen3.5-4B bf16, GB10 | 4 Q | | | ~100 ms |
| kev (published) | 0.5B, M5 | 6 Q | | ~160 ms | |
| Jev (published) | ~10B-active MoE (inferred) | 30k tokens | | ~160 ms | |

Raw llama.cpp `llama-bench pp1024`, idle M4: E2B Q4_0 **569 tok/s**, 270M
f16 **6,181 tok/s**. grande's packed pass: 553 and 6,380 tok/s. The runtime
adds nothing; the wall is prefill compute of the backbone on a base M4 GPU.
Flash attention on/off and n_ubatch 128–2048 change nothing (compute-bound).

State cache (native): a state that is not resident but was seen before is
restored from its serialized KV instead of prefilled. E2B Q4_0, 2,071-token
state, 12 questions, same session (numbers from a busier session than the
table above, so read them against each other): cold 6.5 s, resident 1.49 s,
restored from RAM 1.54 s (restore itself 2–4 ms for 19 MB), restored from a
file in a fresh process 2.46 s total (restore 4–11 ms). Restored answers
match the decoded ones to max |Δp| 1.5e-4 for states longer than the
sliding window (the SWA cache holds only the window after a restore, so the
attention kernel reduces over fewer cells) and exactly for shorter ones.

Browser: the `shared` mode is the same layout as native — state decoded
once into a resident KV cache, one continuation per question — and cuts the
tokens per request from B × (state + question) to state + Σ question. The
branches cannot be batched into one forward because ORT's
GroupQueryAttention requires `batch_size == 1` for a multi-token
continuation from a cache, so each question is its own forward (~90 ms
fixed + ~7 ms/token on E2B); that per-dispatch cost, not the layout, is
now the browser's floor.
reflex's browser answers on the Japanese ticket were wrong on 3 of 5
questions (queue=account 62%, refund 71%); grande E2B got all 5.

## 4. What moves each axis

Size (E2B). Vocabulary pruning first (`tools/prune_vocab.py`: 262k → 29,180
tokens from JGLUE train + kev's suites + the examples, BPE merge closure
kept), then requantized from the pruned Q8_0 with `llama-quantize
--allow-requantize`. Accuracy on the first 300 JGLUE valid records (not in
the pruning corpus); the unpruned Q4_0 scores 0.530 / 0.857 on the same rows.

| variant | size | JNLI | JCQA | 12 Q / 500-tok, warm |
|---|---|---|---|---|
| Q4_0, unpruned | 2,841 MB | 0.530 | 0.857 | 1.06 s |
| Q8_0, pruned | 2,355 MB | 0.533 | 0.860 | 1.15 s |
| Q4_K_M, pruned | 1,408 MB | 0.517 | 0.843 | 1.14 s |
| **Q4_0, pruned** | **1,273 MB** | 0.533 (0.580 at n=400) | 0.847 (0.855 at n=400) | 1.10 s |
| Q4_0, pruned v2 (exact merge paths, 25,392 tokens) | 1,247 MB | 0.513 | 0.840 | |
| IQ4_XS + Japanese imatrix, pruned | 1,226 MB | 0.513 | 0.867 | |
| Q3_K_M, pruned | 1,181 MB | 0.520 | 0.793 | 1.18 s |
| Q2_K, pruned | 969 MB | 0.223 | 0.240 | – (model collapses) |
| Q3_K_M, embeddings kept at Q8 | 1,255 MB | 0.520 | 0.790 | |
| Q2_K, embeddings kept at Q8 | 1,043 MB | 0.213 | 0.243 | (still collapses: it is the blocks) |
| IQ3_XXS + Japanese imatrix, emb Q8 | 1,080 MB | 0.570 | 0.793 | |
| Q2_K + Japanese imatrix, emb Q8 | 1,043 MB | 0.543 | 0.710 | |
| IQ2_M + Japanese imatrix, emb Q8 | 985 MB | 0.603 | 0.557 | |

Reading: pruning is free (Q8_0-pruned ≡ unpruned); Q4 costs nothing
measurable on JNLI and ~1 pt on JCQA. Below Q4 the transformer blocks are
what breaks, not the embeddings: Q2_K collapses with or without Q8
embeddings, and an importance matrix computed on Japanese text
(`llama-imatrix`, 120 × 512 tokens of Wikipedia + JGLUE) brings Q2_K back to
0.54 / 0.71 and gives IQ3_XXS Q3_K_M quality at 1,080 MB. Gemma 4's shared
K/V layers have no activations for the imatrix, so `attn_k` / `attn_v` are
pinned to Q4_0 for the IQ types. Nothing under ~1.0 GB keeps JCQA above
0.79; the next 20% below Q4_0-pruned buys −6 pts, so **Q4_0-pruned
(1,273 MB) is the sweet spot** and the lever below it is layer dropping, not
bits. Quantization level does not change speed (compute-bound).

Pruning corpus: with JGLUE + kev suites + examples (29k tokens kept) text
outside the corpus tokenizes into more pieces (synthetic contract state:
549 → 612 branch tokens, +11%). Adding 3,000 Japanese + 1,500 English
Wikipedia articles keeps 86,897 tokens (33%), the overhead drops to +6.6%
(585 tokens), and the Q4_0 file would be ~1.6 GB. Pick by workload.

Pruned v2 fixes two things in `prune_vocab.py` that made even *in-corpus*
text tokenize longer than the full vocabulary (ticket example: 341 vs 313
tokens, and `refund_requested` moved 0.037 → 0.035): the merge closure kept
one producing pair per token instead of the path BPE actually takes
(`Answer` came out as `An`+`sw`+`er`, `責任` fell to bytes), and the corpus
saw the raw JSON strings, not the rendered `key: {"nested":"json"}` lines
and the `Question: / A: / Answer with one letter.` scaffolding. With the
real path replayed per token the set is *smaller* (25,392) and the examples
tokenize byte-for-byte like the full model; the contract preset's overhead
drops from +9.7% to +5.5% (639 → 674 tokens). JGLUE valid is out of corpus
either way; on the same 300 rows v2 is −2.0 / −0.7 pts against v1, inside
the n=150 noise but not in v2's favour, so the JGLUE train rows that v1 kept
tokens for by accident may be worth keeping on purpose (a wider corpus).

The same token set in the browser: the wgpu export of the pruned GGUF
(`tools/export_wgpu_gguf.py`) is 1,245 MB instead of 2.8 GB — the per-layer
table alone 1.3 GB → 128 MB — with the same answers to four decimals on the
ticket; the transformers.js export pruned by `tools/prune_onnx_vocab.py`
is 3,381 → 1,465 MB, logits bit-identical to the full export on identically
tokenized text (`web/README.md`).

E4B through the same pipeline (`gemma-4-E4B-it-Q4_0.gguf` 4,591 MB): the
pruning corpus keeps the identical 25,392 token ids (same tokenizer), the
per-layer table drops 1,585 → 154 MB and the embedding 713 → 69 MB, so the
pruned GGUF is 2,500 MB and the wgpu export 2,498 MB; llama.cpp answers on
the ticket are equal to the unpruned file to the last digit, and the wgpu
engine matches llama.cpp to ≤ 7e-5 (ticket, isolation).

What is left in the 1,247 MB: FFN 876 MB (69%), attention q/o 149 MB,
per-layer embeddings 147 MB, token embeddings 48 MB, `per_layer_model_proj`
27 MB (bf16; llama-quantize hard-codes it unquantized), the rest 2%. So
"embeddings at Q3" is worth ~40 MB, not 250: below this the lever is the
blocks. IQ4_XS on the blocks buys −47 MB for free; each dropped layer is
~30 MB *and* ~3% of prefill compute (needs the PLE columns sliced and
`shared_kv_layers` adjusted, `llama-quantize --prune-layers` does neither).

The 270M backbone, same levers (`--keep-layers` trains the head on layer N
and exports an N-layer GGUF; vocab pruning with the Wikipedia-broadened
corpus keeps 91k tokens):

| variant | size | JNLI | JCQA | 12 Q / 500-tok cold / warm | ms / record |
|---|---|---|---|---|---|
| 18 layers, 3k records | 551 MB | 0.540 | 0.670 | 168 / 86 ms | 47–77 |
| 18 layers, 12k records | 551 MB | 0.710 | 0.710 | 177 / 92 ms | 73–77 |
| 18 layers, 12k, vocab-pruned | 323 MB | ≈ same | ≈ same | 173 / 84 ms | |
| **12 layers, 6k, vocab-pruned** | **256 MB** | 0.685 | 0.630 | **99 / 59 ms** | **26–34** |

Dropping a third of the layers costs 2–8 pts and buys 1.5–1.7× on speed;
the embedding table is 60% of the small model, so pruning matters more here
than on E2B in relative terms.

Speed (E2B, M4):

| lever | effect | status |
|---|---|---|
| unified KV cache (no per-stream copy) | correctness + budget; needed for long states | done |
| resident prefix across requests over the same state | 500-tok state: 1.94 → 1.01 s; 2,000-tok: 4.69 → 1.11 s | done |
| state cache (RAM LRU + files): any previously seen state restores in ms | 2,000-tok state, second visit or after restart: cold → resident cost | done |
| browser: state resident, questions continue from its KV | ticket 4.2 → 2.5 s, contract 11.4 → 3.1 s (E2B, warm) | done |
| flash attention on/off, n_ubatch 128–2048 | no change (compute-bound) | measured |
| quantization level | no change (compute-bound) | measured |
| shorter label template (`--terse`: no "Question:" / "Answer with one letter.") | −15% branch tokens, but JCQA collapses: 0.857 → 0.480, candidate mass 0.90 → 0.001; JNLI 0.530 → 0.517 with mass 0.99 → 0.83 (first 300 valid rows, E2B Q4_0) | measured, rejected |
| smaller backbone | 270M is 11× faster than E2B (86 vs 1,010 ms warm) | done |
| early exit (train with `--keep-layers`) | 270M 18 → 12 layers: 86 → 59 ms warm, −2 to −8 pts | done on 270M; E2B needs a GPU to train |
| bigger GPU | prefill is compute-bound; a 4090-class GPU is ~30× an M4 | |
| wgpu matmul: f16-pair tiles, K step 32 (one Q4 block), 128-invocation workgroups, vec loads | native 1.53 → 1.12 s, browser 1.2–1.4 → 1.0 s (ticket); GPU time is 90% matmul, gate_up 714 → 533 ms, down 402 → 263 ms | done |
| wgpu matmul: register prefetch of the next tile; f32 X tile | prefetch +20% (matmul) to +140% (gated: 64 accumulators + 44 staged registers spill); f32 X tile +30% | measured, rejected |
| wgpu native: skip naga's per-index bounds clamps (`create_shader_module_trusted`; `GRANDE_WGPU_CHECKED=1` puts them back) | 1.12 → 0.88 s, same answers; the browser cannot (Tint's clamps are not optional) | done, default |
| wgpu attention: K and V staged as f16 and read as vec4, exp once per (row, key), 4 score accumulators, 16 x 8 tiles | 88 → 46 ms over 35 layers (16 x 16: 67, 32 x 8: 101, 8 x 16: 68 at HD 256; the HD 512 layers gained most, 64 → 46 total) | done |
| wgpu matmul: 32-row tiles (twice the workgroups on the small-N layers) | down 222 → 202 ms, o 56 → 51, pl_mm_gate 13 → 9; gated kernel unchanged at 64x64 / 32x128 / 256-invocation 4x4x2 (±3%) | done |
| wgpu matmul: row tiles as the fast dispatch axis (weight-tile reuse); `enable f16` tiles with f16 FMAs and per-step f16 partials | +7%; +5% (the M4 has no double-rate f16, and unpack2x16float was already free) | measured, rejected |
| wgpu matmul: X tile in registers, fetched per k step with `subgroupShuffle` from the lane that loaded it (no xs staging, 2 KB less workgroup memory; wgpu `Features::SUBGROUP`, Metal 32-lane simdgroups) | +40–55% on every matmul (gate_up 11.5 → 17.5 ms a layer, down 5.9 → 9.9, qkv 1.6 → 2.6; three interleaved runs, same answers): four shuffles a step cost far more than the one broadcast threadgroup load they replace. The kernels are issue-bound (64 FMAs + 12 unpacks + 3 loads a step), not workgroup-bandwidth-bound, so moving the W tile through shuffles (8 a step) would lose more | measured, rejected |
| wgpu engine: resident state (the prefix's K/V and positions stay in the cache; the same prefix again runs only the branches, appended past it) | native 12 Q: 500-tok state 3.0 → 1.7 s, 2,000-tok 8.1 → 2.0 s, identical rows (bitwise, `tests/reference.rs`); browser ticket 1.0 → 0.77 s, contract 2.3 → 1.5 s | done |
| wgpu engine: state cache (every decoded prefix's K/V read back as f16 — exact, the attention tile rounds to f16 anyway — with sliding layers window-only; RAM LRU + `--state-cache-dir` files natively, RAM in the browser) | 2,000-tok E2B state = 19 MB, saved in 23 ms, restored in ~50 ms (RAM) / ~80 ms (file); same rows as decoded | done |
| **cross-request batching** (`serve` queues concurrent requests and hands them to the engine as one `answer_many`; the wgpu engine lays every request's state + branches in one pass, each under its own group number in the sequence id, the attention kernel scanning only that request's key rows; llama.cpp optional, `--llama-batch`). The idea DiffusionGemma-as-Jev gets its 162 decisions/s from: 32 concurrent reads in one vLLM step on a DGX Spark, 0.12 s alone → 0.58 s for 32 | `grande bench --batch 16`, fresh state per request, state cache off, 16 requests one by one → together: 270M wgpu 3 Q / 60-tok state 1.21 → 1.04 s (+16%), 12 Q / 500-tok 6.6 → 6.4 s; E2B wgpu 3 Q 15.4 → 16.4 s, 12 Q / 500-tok 29.0 → 29.7 s; E2B llama.cpp 3 Q 13.1 → 12.8 s with `--llama-batch`, 270M llama.cpp 1.7 → 2.6 s (the unified cache: every ubatch attends over all 16 requests' cells). HTTP, `tools/http_bench.py`, 270M wgpu ticket: concurrency 1 → 16 gives 9.5 → 10.8 req/s, batches of 14–15, every answer identical. **No gain on the M4**: one 190-token request already runs the wgpu kernels at 1.5–1.9 TFLOPS, about what they reach on 3,000 tokens, so there is nothing left to fill — the lever is a GPU one request does not saturate (the DGX Spark's 32-way sweep scales 6×; a 4090 would look the same for E2B). Numbers taken with another E4B server resident and the laptop warm, so ~1.5–2× the idle figures above; compare within the row | done, neutral here |
