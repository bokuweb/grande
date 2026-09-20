# grande on JevBench (public items)

2026-09-20. [JevBench](https://benchmarkheaven.com/jev-models) is Benchmark
Heaven's benchmark for Jev-class decision models: 534 decisions per system
(easy 72, standard 96, judge 146, hard 220), one question per request, scored
as the geometric mean of Intelligence, Calibration, Speed and Cost. grande is
not on the published v1.2.2 table (21 systems); this is our own run of the
231 public items with the published harness
([fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench), MIT)
against `grande serve`, whose `/v1/systemone` the harness's `typesafe`
adapter speaks unchanged.

Machine: Apple M4 (base), 16 GB, Metal, llama.cpp backend, zero-shot label
readout, no temperature (T = 1), no contextual calibration. Latency is
caller wall time on localhost, one request at a time.

Held-out items (24 easy, 24 standard, 109 hard) and the whole judge tier
(146 imported items) are not public, so nothing here is the official score;
the published per-task outcomes of every ranked system *are* public, so the
same 231 items can be compared exactly.

## Accuracy on the same 231 public items

`I(pub)` = 100 × (0.14 easy + 0.28 standard + 0.30 hard) / 0.72, i.e. the
official Intelligence weights with the judge tier left out. Published rows
come from `results/v1.2/jevbench-v1.2-per-task.json`; the two grande rows
are ours (`tools/jevbench_compare.py`).

| # | system | easy 48 | std 72 | hard 111 | I(pub) |
|---|---|---|---|---|---|
| 1 | DeepSeek V4.1 Flash (thinking) | 100.0 | 98.6 | 96.4 | 98.0 |
| 2 | GPT-5.6 Luna (low) | 100.0 | 97.2 | 96.4 | 97.4 |
| 3 | Gemini 3.1 Flash-Lite | 100.0 | 98.6 | 73.9 | 88.6 |
| 4 | Jev 1.13.0 | 100.0 | 98.6 | 73.0 | 88.2 |
| 5 | classifier.dev (fast tier) | 100.0 | 98.6 | 70.3 | 87.1 |
| 6 | openjev-sglang (Qwen3.6-35B-A3B) | 100.0 | 94.4 | 73.0 | 86.6 |
| 7 | djev (diffusion-gemma) | 100.0 | 98.6 | 67.6 | 85.9 |
| 8 | OpenJev (DiffusionGemma 26B-A4B) | 100.0 | 97.2 | 64.0 | 83.9 |
| 9 | SemIf (Qwen3.5-4B, zero-shot) | 100.0 | 98.6 | 61.3 | 83.3 |
| **10** | **grande, Gemma 4 E4B it Q4_0, zero-shot** | **100.0** | **94.4** | **51.4** | **77.6** |
| 11 | system-one-open (Gemma 4 E2B LoRA) | 100.0 | 93.1 | 48.6 | 75.9 |
| 12 | open-alternative-jev (Qwen3.5-4B) | 100.0 | 83.3 | 56.8 | 75.5 |
| 13 | Qwen3.8 27B (partial) | 100.0 | 98.6 | 42.3 | 75.4 |
| 14 | system-one (Qwen3-8B) | 100.0 | 88.9 | 48.6 | 74.3 |
| 15 | Bespoke Nimble 9B | 100.0 | 93.1 | 36.9 | 71.0 |
| 16 | jeff (GLiFormer 400M) | 100.0 | 75.0 | 38.7 | 64.8 |
| **17** | **grande, Gemma 4 E2B it Q4_0 vocab-pruned (1.2 GB), zero-shot** | 97.9 | 73.6 | 34.2 | 61.9 |
| 18 | Laya (ModernBERT-large 421M) | 95.8 | 69.4 | 35.1 | 60.3 |
| – | laya-multilingual (mmBERT-base 322M) through laya-mlx, our run — see [laya.md](laya.md) | 89.6 | 40.3 | 33.3 | 47.0 |
| 19 | GLiNER2 (gliner2.5-base) | 97.9 | 63.9 | 36.9 | 59.3 |
| 20 | openJev Verdict (ModernBERT-base 151M) | 85.4 | 62.5 | 37.8 | 56.7 |
| 21 | open-jev-deberta-v3-large | 100.0 | 43.1 | 37.8 | 52.0 |
| 22–23 | Needle 3 (two modes, partial) | 66.7 / 47.9 | 26.4 / 16.7 | 0.0 / 15.3 | 23.2 / 22.2 |

Standard tier, E2B variants (same 72 items): vocab-pruned Q4_0 **73.6**,
unpruned Q4_0 **79.2**, unpruned Q8_0 **80.6**, E4B Q4_0 **94.4**. The
Japanese-corpus vocabulary pruning costs ~5 points on English; quantization
does not matter; the base model does. Zero-shot E4B lands just above the
LoRA-trained E2B of system-one-open (94.4 / 51.4 vs 93.1 / 48.6) and below
zero-shot Qwen3.5-4B (SemIf, 98.6 / 61.3).

## E4B Q4_0 in detail

Summaries as the harness writes them: `docs/jevbench/e4b-q4_0-{easy,original,hard}-summary.json`.

| tier | n | accuracy | ECE | Brier | p50 | p95 |
|---|---|---|---|---|---|---|
| easy (public 48) | 48 | 1.000 | 0.000 | 0.000 | 0.45 s | 0.56 s |
| standard (public 72) | 72 | 0.944 | 0.050 | 0.104 | 0.46 s | 0.59 s |
| hard (public 111) | 111 | 0.514 | 0.327 | 0.717 | 3.59 s | 19.5 s |

Standard by family: adequacy 0.83, policy 0.83, extraction / intent /
ordinal / routing 1.00.

Hard by family, against the published outcomes of the same items:

| family | n | grande E4B | SemIf | system-one-open | Jev | GPT-5.6 |
|---|---|---|---|---|---|---|
| adversarial | 6 | 0.83 | 0.67 | 0.67 | 1.00 | 1.00 |
| ambiguous | 7 | 0.57 | 0.71 | 0.43 | 0.86 | 1.00 |
| judge_hard | 17 | 0.47 | 0.76 | 0.82 | 0.76 | 0.94 |
| long_policy | 19 | 0.53 | 0.53 | 0.32 | 0.63 | 1.00 |
| multi_hop | 18 | 0.56 | 0.67 | 0.39 | 0.83 | 0.94 |
| probability | 10 | 0.30 | 0.40 | 0.40 | 0.70 | 0.90 |
| routing_hard | 5 | 1.00 | 1.00 | 0.80 | 1.00 | 1.00 |
| temporal_numeric | 15 | **0.07** | 0.20 | 0.33 | 0.27 | 0.93 |
| tradeoff | 6 | 0.50 | 0.67 | 0.00 | 0.83 | 1.00 |
| trap | 8 | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 |

The holes are date / number arithmetic (temporal_numeric), matching a gold
probability distribution (probability: mean TVD 0.46) and judging long
answers (judge_hard). long_policy items are 2,000–6,000-token states and
are where the p95 latency comes from.

## What the official score would roughly be

Official formulas (`jevbench/composite_v12.py`) on our numbers, with the
parts we cannot measure made explicit: judge tier taken as standard − 3
points (the published systems lose 1–5 there), Speed from the standard
tier's latency treated as a self-hosted endpoint (×2 + 0.15 s), Cost at the
hosted-provider estimate JevBench uses for the 4B class (~$0.023 per 1,000
decisions).

| variant | I | C | S | K | score |
|---|---|---|---|---|---|
| E4B, raw probabilities, self-hosted | 81.5 | 44.5 | 78.4 | 59.1 | **64.0** |
| + one temperature fitted on the standard tier (T = 1.3) | 81.5 | 51.2 | 78.4 | 59.1 | 66.3 |
| + counted as a production API (no latency adjustment) | 81.5 | 51.2 | 85.6 | 59.1 | 67.8 |

That is the system-one-open (68.7) / OpenJev (67.6) / jeff (66.9) band,
#7–#11 on the v1.2.2 table. Calibration is the cheap axis: on the hard tier
the readout is badly overconfident (ECE 0.33, temperature ≈ 3–4 by a 2-fold
cross-fit inside the public hard items, which brings ECE to 0.13 and C to
~71 — but a temperature fitted on the benchmark itself is not a number to
publish; it says what a temperature fitted on comparable long-state data
would be worth).

## What the standard-tier misses looked like (E2B)

The E2B failures were not option-order bias (`/v1/systemone/permute` gives
the same distribution under both orders). Two things:

- **noul wording.** JevBench sends `criteria: {true: …, false: …}` and grande
  renders the keys literally (`A: true — Every required condition is
  established…`). Over a policy state that plainly permits the action, E2B
  answers `false` at 0.96; the same state with a plain yes / no question is
  `yes` at 0.92. A `yes` / `no` rendering with the criteria in the
  instructions was the best of four variants on the standard tier (0.67 vs
  0.58 on its 24 nouls) but not on the hard tier; E4B is much less
  sensitive. Worth switching the rendering to `yes` / `no` regardless.
- **Shallow reading at 2B.** "Where is the parcel? Keep the delivery address
  as it is." → `change_address` 0.99; "Suggest three names for a pet
  dragon" → `coding` 0.53. E4B gets these right.

## Reading

- JevBench is one question per request over a fresh state every time.
  grande's packing, resident state and state cache do nothing here; a
  ranking on it says nothing about the N-questions-per-document case grande
  is built for.
- For a JevBench submission the model to send is E4B (unpruned, or pruned
  on an English rendered corpus), with a temperature fitted on long-state
  data that is not JevBench, and a `yes` / `no` noul rendering.
- JevBench's own runs are on a 4-GB CPU box or a rented GPU; E4B Q4_0
  (4.6 GB) needs the GPU route, the pruned E2B (1.2 GB) fits the CPU route
  but scores like the 400M-class systems there.

## Reproduce

```bash
git clone https://github.com/fstandhartinger/jevbench.git
./target/release/grande serve --model models/gemma-4-E4B-it-Q4_0.gguf --port 8787
cd jevbench && for t in easy original hard; do
  python3 -m jevbench.cli run --tasks datasets/public/$t.jsonl --adapter typesafe \
    --endpoint http://127.0.0.1:8787 --key-env '' --model grande-latest \
    --cost-basis no_billable_account_local --reserve-usd 0 --cap-usd 0 \
    --results ../runs/jevbench/e4b/$t/results.jsonl --raw-dir ../runs/jevbench/e4b/$t/raw \
    --ledger ../runs/jevbench/ledger.jsonl
  python3 -m jevbench.cli summarize --tasks datasets/public/$t.jsonl \
    --results ../runs/jevbench/e4b/$t/results.jsonl --public-export ../runs/jevbench/e4b/$t/summary.json
done
python3 ../tools/jevbench_compare.py . ../runs/jevbench   # rank + score estimate
```

The harness refuses `--results` / `--raw-dir` inside its own checkout and
needs only the Python standard library.
