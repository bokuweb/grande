# A Laya-shaped cross-encoder on Ruri v3 (70m, 310m)

2026-09-23. The (state, option) embedding head of [e5.md](e5.md) /
[ruri.md](ruri.md) never sees a question's instructions, so every noul over
one state gets the same probability. This trains the other shape — Laya's
([laya.md](laya.md)): one question is one sequence

```
<s> <type> question: <instructions> </s> <mask> opt0 <mask> opt1 … </s> <state json> </s>
```

read by the whole encoder, and a scorer (LayerNorm → Linear → GELU →
Linear(1)) on each `<mask>` row gives the option's logit. Backbone
`cl-nagoya/ruri-v3-70m` (70M, 31M without the embedding table), trained
end to end on the M4 (MPS, 2 epochs, 38 min). Script:
`tools/cross_train.py`; preset comparison: `tools/cross_compare.py`;
results in `runs/cross/70m/` (`summary.json`, `presets.json`).

## Data

| source | sequences / epoch | target |
|---|---|---|
| `.cache/distill/labeled.jsonl`: 1,806 synthetic states (tickets, contract clauses, memos, reviews, captions, wiki), their questions minus 8 held-out instructions, twice with fresh option orders | 12,070 | Gemma 4 E4B's probabilities (soft) |
| JGLUE JNLI + JCommonsenseQA train, in `tools/http_eval.py`'s request shape | 29,012 | gold |

The distill corpus has **33 distinct instructions** (25 trained on here).
Held out entirely — never in training, any state — are 8 of them, one to
three per family and every question type (the table below).

## Results

**Seen instructions, unseen states** (194 held-out states, 640 questions),
agreement with E4B's answer:

| | all | choice | score | noul |
|---|---|---|---|---|
| cross-encoder | **0.938** | 0.897 | 0.763 | 0.988 |
| E4B's majority answer per instruction (oracle prior) | 0.691 | | | |

ECE against E4B 0.03. For question families it was trained on, the 70M
encoder reproduces the 4B teacher.

**Held-out instructions** (never trained on, 2,116 questions over all
states):

| instruction | type | n | agreement with E4B |
|---|---|---|---|
| この文章に数値が含まれているか | noul | 668 | 0.771 |
| 顧客が求めている対応はどれか | choice | 146 | 0.760 |
| この条項は契約終了後も効力が続く義務を定めているか | noul | 174 | 0.684 |
| セキュリティに関わる問い合わせか | noul | 130 | 0.538 |
| 乙（受託者）から見たこの条項のリスクはどの程度か | score | 150 | 0.467 |
| レビュアーは製品を他人に勧めているか | noul | 277 | 0.347 |
| このレビューの評価はどれか | score | 281 | 0.331 |
| 会議の場所が明記されているか | noul | 290 | 0.190 |
| **all** | | 2,116 | **0.534** (oracle prior 0.610) |

It does not generalise from 25 instructions to new ones: below the oracle
prior overall, near chance on several, and confidently wrong on some (ECE
0.23; "会議の場所が明記されているか" at 0.19 means it answers the opposite
of E4B most of the time — it has learned what memos usually look like, not
what the question asks).

**The demo presets** (17 requests, 99 questions; every instruction unseen
by both Ruri models; `ticket-ja` and the ticket preset are the same
request). There is no gold here, so the reference is omg on Gemma 4 E2B
(`omg probe`, zero-shot): choice = same argmax, noul = same side of 0.5,
score = same rounded level.

| | agreement with E2B | noul answers that differ within a request |
|---|---|---|
| **cross-encoder (70m)** | **0.667** (66 / 99) | 91% |
| (state, option) embedding head (70m, docs/ruri.md) | 0.374 (37 / 99) | 37% (all nouls of a request equal) |
| E2B itself | – | 91% |

Against Laya (laya-multilingual on omg's wgpu engine, `omg probe`, the 16
presets without the duplicate ticket, 94 questions): Laya agrees with E2B
on **0.596** (56 / 94), the cross-encoder on **0.649** (61 / 94) — five
questions apart, within noise at this size. They fail differently: Laya
gets the calibration preset 7 / 7 (16:00? 0.12, 会議室 1? 0.38 — it reads
the detail), the cross-encoder 5 / 7; the cross-encoder is ahead where its
training families are close (incident 5 vs 2, expense 4 vs 2, ticket 5 vs
4). On JGLUE it is far ahead (0.898 / 0.818 vs Laya's 0.702 / 0.551), but
it was trained on JGLUE's train split and Laya was not.

The answers now depend on the question — the contract preset's five nouls
come out 0.99 / 0.87 / 0.98 / 0.97 / 0.99 instead of 0.41 × 5 — and it
agrees with E2B twice as often as the embedding head (7/8 on the contract,
5/5 on the ticket). But depending on the question is not the same as
reading it closely. The calibration preset's memo says the meeting is at
15:00 in 会議室 2; the cross-encoder answers "starts at 16:00?" 0.99 and
"in 会議室 1?" 0.98 (E2B: 0.02 / 0.01), and gets only the questions whose
answer is not in the memo right (合言葉 / 社長の名前: 0.09–0.11). It
matches the shape of a question to what such states usually say, not the
number or name in this one. (Both it and E2B also say yes to
"遅延損害金の利率は年14.6%を超えているか" when the clause says exactly
14.6%.) Far from the training families (news, review aspects, RAG
grounding) it drops to 2–3 of 6.

**JGLUE valid** (odd half, the same model — multi-task, not a JNLI
specialist):

| | JNLI | JCQA |
|---|---|---|
| cross-encoder 70m | 0.898 (ECE 0.061 → 0.013) | 0.818 |
| ruri-v3-70m fine-tuned on the task alone (ruri.md) | 0.913 | 0.837 (1 epoch) |
| frozen ruri-v3-70m + head | 0.758 | 0.825 |
| omg E2B zero-shot / + pointer head | 0.614 / 0.848 (first 400) | 0.853 |

One model serves both tasks within 1.5–2 points of the single-task
fine-tunes.

**Speed**: 23–116 ms for a 3–8-question preset in PyTorch on MPS (each
question re-reads the state; the contract preset is 8 sequences of ~200
tokens). On omg's wgpu engine this is the Laya path (`laya.rs` already
runs a ModernBERT encoder with a scorer on marker rows), so the port is a
config away; expect Laya's ~15–20 ms per question.

## Reading

- **The shape works for trained families.** Instructions in the sequence
  are what separates the questions: 91% of noul answers differ within a
  request, like E2B's, and on the seen families the 70M model matches E4B
  94% of the time.
- **It does not read details of new questions.** On unseen instructions it
  answers from the question's shape (16:00 vs 15:00, 会議室 1 vs 2 both
  "yes"), which is exactly the failure a System One must not have.
- **Coverage is the data.** 25 instruction strings teach the families,
  not "reading a question": on held-out instructions it is at or below
  the answer prior. Laya bought generalisation with many datasets and RL;
  here that means many more instructions — synthesise hundreds of
  question strings per family (paraphrases, new predicates), label with
  E4B, and hold out whole families to measure it.
- **As a tier today**: route the question families it was trained on to
  it (E4B-level answers at a fraction of the cost), everything else to
  E2B / E4B. A confidence or act head (Laya's `act_head`, trained on
  disagreement with the teacher) would make that routing automatic —
  the raw confidence is not enough, since the held-out failures are
  confident.

## ruri-v3-310m

The same recipe on `cl-nagoya/ruri-v3-310m` (315M, 236M without the
embedding table): token embeddings frozen, bf16 autocast, lr 3e-5, 8
sequences per 256 tokens — to fit the 16 GB M4 (it still swapped to
11 GB) — 2 epochs, 4.1 h. `runs/cross/310m/`.

| | 70m | **310m** | reference |
|---|---|---|---|
| seen instructions, held-out states: agreement with E4B | 0.938 | **0.948** | oracle prior 0.691 |
| **held-out instructions**: agreement with E4B | 0.534 | **0.669** | oracle prior 0.610 |
| … ECE vs E4B on held-out instructions | 0.232 | 0.119 | |
| 16 Japanese presets: agreement with E2B | 0.649 (61 / 94) | **0.713** (67 / 94) | Laya 0.596 (56 / 94) |
| noul answers that differ within a request | 91% | 99% | E2B 91% |
| JNLI valid (odd half) | 0.898 | **0.928** | E2B + pointer head 0.848 (first 400) |
| JCommonsenseQA valid (odd half) | 0.818 | **0.909** | E2B 0.853, E4B 0.932 (first 400; 310m first 400: 0.915) |

Held-out instructions, one by one (70m → 310m): 会議の場所が明記されているか
0.19 → 0.69, セキュリティに関わる問い合わせか 0.54 → 0.79, 契約終了後も効力が続く
義務 0.68 → 0.81, レビュアーは製品を他人に勧めているか 0.35 → 0.56,
このレビューの評価 0.33 → 0.52, 乙から見たリスク 0.47 → 0.60, 顧客が求めている
対応 0.76 → 0.73, 数値が含まれているか 0.77 → 0.71.

On the calibration preset (memo: 15:00, 会議室 2) the 310m says "16:00?"
0.64 and "会議室 1?" 0.76 — less sure than the 70m's 0.99 / 0.98, still on
the wrong side (E2B 0.02 / 0.01, Laya 0.12 / 0.38); the unanswerable ones
stay low (0.08–0.23). On the contract it now leans "no" on
"年14.6%を超えているか" (0.64 vs the 70m's 0.98 and E2B's 0.92 — the
clause says exactly 14.6%, so lower is right).

Reading: scale is what moved the held-out number — the same 25 training
instructions, and the 310m is above the oracle prior (0.669 vs 0.610) where
the 70m was below it, with half the calibration error. It reads new
questions better but not reliably (0.52–0.81 per instruction, the
15:00 / 16:00 detail still wrong), so the conclusion about coverage stands;
the 310m is the backbone to run the "many more instruction strings"
experiment on. JCQA 0.909 is past zero-shot E2B, JNLI 0.928 past E2B + a
trained head, in one multi-task model. Latency in PyTorch was not
measured cleanly (the M4 was swapping); on omg's wgpu engine a 310m
ModernBERT is the laya.rs path, 41 ms for a 5-question request as an
embedder (ruri.md), so roughly 5× that per request as a cross-encoder,
which re-reads the state per question.

## In the demo (omg's wgpu engine)

Both models run on `laya.rs`, which now takes a checkpoint without Laya's
question-type embedding, decision-head layers and act head
(`laya_agent.type_emb` / `head_layers` / `act_head` in the export's
config). `tools/export_cross.py` writes the Laya export layout (Q8
encoder, f16 norms and scorer, the full 102k vocabulary, `<s>` / `</s>`
as the separators the trainer used): `ruri-v3-70m-cross-wgpu` 78 MB,
`ruri-v3-310m-cross-wgpu` 326 MB, release `cross-v1`, listed in the demo
and picked up by `omg serve | probe --model <dir>`.

Parity with the PyTorch model on the 16 Japanese presets (94 questions):
the same decision on 94 / 94 for both sizes, largest probability
difference 0.009 (Q8). The browser gives the same answers (310m: 16:00?
0.644, 会議室 1? 0.756, 14.6%? 0.632).

| request time, M4 | 70m | 310m |
|---|---|---|
| native (`omg probe`), 16 presets: min / median / max | 13 / 54 / 98 ms | 69 / 335 / 630 ms |
| browser (Chromium, WebGPU): ticket (5 q) / contract (8 q) | – / 145 ms | 289 / 987 ms |

Every question re-reads the state (the contract preset is 8 sequences of
~240 tokens, 1,970 tokens in all), so the 310m in the browser costs about
what E2B does on the same request; the 70m is the fast one.

## Rerun

```bash
python tools/cross_train.py --model cl-nagoya/ruri-v3-70m --out runs/cross/70m \
    --probe examples/ticket-ja.json runs/cross/preset-*_ja_.json      # ~40 min on an M4
# E2B references for the presets (presets dumped from web/presets.js to runs/cross/preset-*.json)
for f in examples/ticket-ja.json runs/cross/preset-*_ja_.json; do
  ./target/release/omg probe --model models/gemma-4-E2B-it-Q4_0.gguf --request $f > runs/cross/e2b/$(basename $f .json).json
done
python tools/cross_compare.py --requests examples/ticket-ja.json runs/cross/preset-*_ja_.json
# 310m (~4 h on an M4; resumes after the last finished epoch)
python tools/cross_train.py --model cl-nagoya/ruri-v3-310m --out runs/cross/310m --bs 8 --lr 3e-5 --freeze-embed --bf16
python tools/export_cross.py --model cl-nagoya/ruri-v3-310m --weights runs/cross/310m/model.pt --out web/models/ruri-v3-310m-cross-wgpu
python tools/cross_compare.py --model cl-nagoya/ruri-v3-310m --cross runs/cross/310m \
    --embed-head runs/ruri/310m/generic/head.pt --embed-temperature 1.3 --requests examples/ticket-ja.json runs/cross/preset-*_ja_.json
```
