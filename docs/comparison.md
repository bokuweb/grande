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

| runtime | model | request | tokens | latency |
|---|---|---|---|---|
| grande native (Metal) | Gemma 3 270M f16 + head | 12 Q, 500-tok state | 1,072 | **174 ms** (6,160 tok/s) |
| grande native | Gemma 4 E2B Q4_0 | 12 Q, 500-tok state | 1,072 | 4,450 ms cold / **2,480 ms** state resident |
| grande native | Gemma 4 E2B Q4_0 | 5 Q ticket | 313 | 785 ms |
| grande native | Gemma 4 E2B Q4_0 | JGLUE, 1 Q | ~100 | 700 ms |
| grande native | Gemma 3 270M + head | JGLUE, 1 Q | ~100 | 47–77 ms |
| kev-0.5b (PyTorch MPS fp32) | Qwen2.5-0.5B | 5 Q ticket | 261 | 211–313 ms |
| kev-0.5b | JGLUE, 1 Q | | | 136–234 ms |
| grande browser (WebGPU) | Gemma 3 270M q4f16 | 5 Q ticket | 673 | 2.0 s |
| grande browser | Gemma 4 E2B q4f16 | 5 Q ticket | 673 | 4.2 s |
| grande browser | Gemma 4 E2B q4f16 | 8 Q contract | 2,263 | 11.4 s |
| **reflex browser** (WebGPU) | Qwen3.5-0.8B q4f16 | 5 Q ticket (same JSON) | 1,114 / 324 warm | 6.7 s cold / 3.5 s state cached |
| reflex Python (published) | Qwen3.5-4B bf16, GB10 | 4 Q | | ~100 ms warm |
| kev (published) | 0.5B, M5 | 6 Q | | ~160 ms |
| Jev (published) | ~10B-active MoE (inferred) | 30k tokens | | ~160 ms |

Raw llama.cpp `llama-bench pp1024` on this M4: E2B Q4_0 **247 tok/s**, 270M
f16 **6,181 tok/s**. grande's packed pass runs at 242–291 and 6,160 tok/s,
i.e. the runtime adds nothing; the wall is prefill compute of the backbone on
a base M4 GPU. reflex's browser answers on the Japanese ticket were wrong on
3 of 5 questions (queue=account 62%, refund 71%); grande E2B got all 5.

## 4. What moves each axis

Size (E2B Q4_0, 2,841 MB):

| step | size | accuracy | status |
|---|---|---|---|
| vocab pruning 262k → 29k (JGLUE train + kev suites + examples) | **1,273 MB** | unchanged on JGLUE valid | `tools/prune_vocab.py`, done |
| + PLE / embeddings at Q2–Q3 instead of Q4/Q8 | ~1.1 GB | untested | quantize from the Q8_0 source |
| + transformer blocks Q3_K / IQ3 | ~0.9 GB | untested | |
| + early exit / layer drop (needs training) | ~0.6 GB | | RFC 07 §2.4 |
| 270M + head (different backbone) | 0.55 GB f16, ~0.3 GB Q8 | JNLI 0.54 / JCQA 0.67 with 3k records | done; more data is the lever |

Speed (E2B, M4):

| lever | effect | status |
|---|---|---|
| unified KV cache (no per-stream copy) | correctness + budget; needed for long states | done |
| resident prefix across requests over the same state | 4.45 s → 2.48 s on a 500-token state, 12 Q | done |
| flash attention on/off, n_ubatch 128–2048 | no change (compute-bound) | measured |
| shorter label template | ~−15% branch tokens | not done |
| smaller backbone | 270M is 25× faster than E2B | done |
| early exit at layer 24/35 | ~−30% | needs training |
| bigger GPU | prefill is compute-bound; a 4090-class GPU is ~30× an M4 | |
