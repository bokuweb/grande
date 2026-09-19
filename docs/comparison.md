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
| grande Gemma 3 270M + LoRA + pointer head | 1,500 JNLI + 1,500 JCQA train, 2 ep | 0.540 (n=200) | 0.112 → 0.091 | 0.670 (n=200) | 0.074 | **47–77** |
| **kev-0.5b** (Qwen2.5-0.5B, English-trained), via its `/v1/systemone` | 6 English datasets | 0.450 (n=300) | 0.168 | 0.577 (n=300) | 0.059 | 136–234 |
| reflex | – | not run: Python engine needs CUDA; browser 0.8B below | | | | |
| jev_local (LFM2.5-1.2B) / Jev | – | not run here | | | | |

Reading: zero-shot Gemma 4 E2B is the most accurate on Japanese by a wide
margin but badly overconfident on NLI until one temperature is applied. kev is
an English model and it shows (JNLI 0.45 is near the 3-way chance of a
label-skewed set). A 270M model with 3,000 training records already beats
kev on Japanese at a quarter of its latency; that is the size/speed lever, not
the accuracy ceiling.

## 2. Accuracy, kev's frozen English suite (identical questions)

`evals/decision-v1/development.jsonl`, clean variant. kev and Jev numbers are
kev's published `docs/kev-vs-jev-summary.json`. grande here is the zero-shot
E2B label readout with no English tuning at all.

| task | n | kev-0.5b (trained on these sources) | Jev | grande E2B zero-shot |
|---|---|---|---|---|
| agnews (4-way) | 80 | 0.913 | 0.813 | 0.775 |
| agnews_yn (noul) | 160 | 0.925 | 0.906 | 0.844 |
| boolq (noul) | 80 | 0.788 | **0.925** | 0.838 |
| mnli (3-way) | 80 | 0.775 | 0.825 | 0.512 |
| sst5 (5 levels) | 80 | 0.575 | 0.613 | 0.375 |
| yelp (5 levels) | 80 | 0.613 | 0.650 | 0.425 |
| yelp_yn (noul) | 80 | 0.863 | 0.825 | 0.800 |
| banking77 (77-way) | 80 | 0.800 | 0.838 | – (label readout caps at 52 options) |
| **all but banking77** | 640 | **0.797** | **0.808** | **0.677** |

grande ECE on this suite: 0.198 raw (no temperature), NLL 0.994.

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
| **Q4_0, pruned** | **1,273 MB** | 0.580 (n=400) | 0.855 (n=400) | 1.10 s |
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

Beyond this: PLE / embeddings at Q3 with the blocks at Q4 (~1.0 GB), early
exit / layer drop (needs training, RFC 07 §2.4), or the 270M backbone
(0.55 GB f16, JNLI 0.54 / JCQA 0.67 with 3k records; data is the lever).

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
| early exit at layer 24/35 | ~−30% | needs training |
| bigger GPU | prefill is compute-bound; a 4090-class GPU is ~30× an M4 | |
