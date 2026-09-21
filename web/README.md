# grande web demo

Static page: `index.html` + `app.js` + `engine.js` + `cache.js` +
`presets.js` + `pkg/` (wasm-bindgen output of `crates/grande-web`).
transformers.js is loaded from jsDelivr; model weights stream from the
Hugging Face Hub once and are kept in IndexedDB (`cache.js`, plugged in as
`env.customCache`: Chromium's Cache API rejects the large `.onnx_data`
shards, so the default cache only kept the small files). Loaded models also
stay resident for the page's lifetime, so switching back is instant.
Nothing is uploaded.

```bash
# build the wasm (once per grande-core change)
cargo build --release -p grande-web --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg --out-name grande target/wasm32-unknown-unknown/release/grande_web.wasm
wasm-opt -Oz -o web/pkg/grande_bg.wasm web/pkg/grande_bg.wasm

# serve
python3 -m http.server 8765 --directory web
open "http://localhost:8765/"
```

The page offers three models: `gemma-4-e2b-wgpu-ja` (the default) and
`gemma-4-e4b-wgpu-ja`, Gemma 4 E2B / E4B on grande's own wgpu engine, and
`laya-multilingual-wgpu`, Convai's Laya (mmBERT encoder + decision head,
docs/laya.md) on the same engine — 180 MB (Q8, 56k-token vocabulary,
`tools/export_laya.py`), one bidirectional sequence per question, ~50 ms
for a 3-question request; fetched from the `laya-v1` release into
`./models/` by `fetch-models.sh` (small enough for the Pages site). The
sentence-embedder backends (multilingual-e5-small, Ruri v3 130m / 310m
with a trained (state, option) head; docs/e5.md, docs/ruri.md; releases
`e5-v1`, `ruri-v1`) run on the engine too (`kind: "e5"` in engine.js) but
are not listed: the head never reads a question's instructions, so every
noul question over one state gets the same answer — they are a native
option (`grande serve --model <export>`) for question families a head was
trained on, not a general System One. The
Gemma models share the same 25k-token vocabulary, 1.2 GB / 2.5 GB, streamed from the Hugging
Face repos `bokuweb/gemma-4-E2B-it-grande-wgpu-ja` /
`bokuweb/gemma-4-E4B-it-grande-wgpu-ja` (or from `./models/<id>/` when
that directory exists). Requires WebGPU (Chrome / Edge, Safari 26+). `engine.js` still carries the loaders
for everything else this page has run — the Gemma 3 / Gemma 4 ONNX exports
through transformers.js, the full-vocabulary `gemma-4-e2b-wgpu` (2.8 GB),
the trained 270M pointer model — so an entry can be put back in `MODELS`
(specs in git history); the sections below describe those paths too.
Gemma 4 ONNX models are loaded text-only (`Gemma4ForCausalLM`): the audio
and vision encoder shards (270 MB for E2B) are never fetched.

What runs where:

- `grande-core` (wasm): request validation, the rendered layout (Gemma 3 and
  Gemma 4 turn markers), the branch plan (`plan` / `plan_second`: option
  orders to average, the group and finalist passes of a Choice with more
  than 52 options), option labels, temperature / softmax / confidence, the
  TypeSafe-shaped response. Same crate as the native runtime, so **Orders**
  in the page is `--orders` and a 77-way Choice runs the same two stages as
  `grande serve` (the page shows `2-stage N` and the order `spread` per
  question).
- `engine.js`: tokenizes the rendered segments (control tokens in caller text
  are neutralized the same way as native), decodes the state once into a KV
  cache, continues every question from it with `num_logits_to_keep = 1`, and
  reads the label logits and candidate mass. **Calibrate** (on by default)
  runs the same questions over the content-free state `N/A` once (cached per question,
  never through the resident cache) and hands those logits to
  `grande.answer`, which subtracts them before the softmax — contextual
  calibration, the same `--baseline` as native.

Modes:

- `shared` (default): the state is decoded once and its KV cache stays
  resident in the engine; every question continues from that cache as its
  own forward, so a branch sees the state and itself only, exactly the
  native layout. The next request over the same state skips the state pass
  (`usage.state_resident: true`). One forward per question because ORT's
  GroupQueryAttention requires `batch_size == 1` when a multi-token input
  continues from a cache — and a refused run leaves the session unusable, so
  the tiled-cache single forward is not attempted. A branch forward is
  short (only its own tokens), so what remains is ORT's per-dispatch cost.
- `batched`: one forward where every row is `state + question`, right- or
  left-padded. The state is re-read once per question.
- `sequential`: one forward per `state + question`, for comparison.

The trained pointer model (`grande-270m-ja`, a hidden-state export without
a KV cache) always runs batched, so it still re-reads the state per question.

`grande-270m-ja-wgpu` is the same checkpoint on grande's own engine
(`crates/grande-wgpu`, compiled into `pkg/grande_bg.wasm` and run on WebGPU
through wgpu): state and every question in one forward pass with a
block-causal mask, no ONNX Runtime. Its files (`config.json`,
`tokenizer.json`, `head.safetensors`, `model.safetensors` f16, 320 MB) come
from `tools/export_wgpu.py` and live in `models/grande-270m-ja-wgpu/`
(release `wgpu-v1`; neither 270M model is in the page's list or fetched by
`fetch-models.sh` any more). Measured with the
GPU shared with a training job, interleaved with the ONNX model: ticket
0.17–0.55 s vs 0.86–1.35 s, contract 0.41–0.74 s vs 2.9–4.0 s.

`gemma-4-e2b-wgpu` is Gemma 4 E2B on the same engine: the llama.cpp Q4_0
GGUF repacked by `tools/export_wgpu_gguf.py` into a `manifest.json`
directory (one file per layer, `embed.bin`, and the 1.3 GB
`per_layer_table.bin` of per-layer token embeddings, 2.8 GB in all). The
loader streams the files into the engine one at a time (`WgpuLoader`);
the per-layer table stays in JS and its rows are gathered and dequantized
per request. Zero-shot label readout, one pass, no resident state: ticket
1.16 s, contract 2.5 s on an idle M4 (the ONNX path: 2.5–2.8 s / 3.1–5.2 s).
The files are served from `./models/gemma-4-e2b-wgpu/` when present (local
development: export there or symlink) and otherwise from the Hugging Face
repo `bokuweb/gemma-4-E2B-it-grande-wgpu` — GitHub Pages caps a site at
1 GB and release assets are not CORS-enabled — uploaded with
`tools/upload_wgpu_hf.py`. Until the repo exists the entry is listed
disabled ("not published yet"), so the site deploys either way.

`gemma-4-e2b-wgpu-ja` (the one the page ships) is the same export from
the vocabulary-pruned GGUF (`tools/prune_vocab.py`: 262k → 25,392 tokens
from JGLUE train, kev's suites and the examples; `docs/comparison.md` §4):
`per_layer_table.bin` 1.3 GB → 128 MB, `embed.bin` 455 → 69 MB, 1.2 GB in
all, and the same answers to four decimals on the ticket, natively and in
the browser (from the Hub: 1.14 s warm on an idle M4). Text inside the
pruning corpus tokenizes exactly as with the full vocabulary; outside it
~5% more pieces (contract preset 639 → 674 tokens).

`gemma-4-e4b-wgpu-ja` is Gemma 4 E4B through the same pipeline
(`gemma-4-E4B-it-Q4_0.gguf` → `prune_vocab.py` with the same corpus, which
keeps the identical 25,392 token ids since both sizes share the tokenizer →
`export_wgpu_gguf.py`): 4.6 GB → 2.5 GB, answers equal to the unpruned GGUF
on llama.cpp to the last digit on the ticket, wgpu vs llama.cpp ≤ 7e-5.
E4B has two K/V heads (grouped-query attention, 4 query heads each), which
the attention kernel dispatches as one workgroup column per K/V head.
Ticket ~2.5 s warm on an M4 in the browser (E2B ~1.0 s).

The ONNX counterpart exists too: `tools/prune_onnx_vocab.py` applied to the
onnx-community export with the same token set drops the embedding,
per-layer-embedding and `lm_head` rows of the other 237k tokens
(embed_tokens 1,591 → 154 MB, decoder 1,520 → 1,310 MB) and renumbers
`tokenizer.json` to match. Logits are bit-identical to the full export for
any text that tokenizes the same (checked on CPU ORT; `shared` on the
ticket preset gives the same numbers to the last digit), 1.5 GB, published
as `bokuweb/gemma-4-E2B-it-ONNX-ja` (spec: `kind: "gemma4"`, `hub` = the
repo id, `padding: "right"`). `?base=http://host/path/` fetches a Hub
model's files from `${base}${id}/` instead (and skips the published-or-not
probe), e.g. a `python3 -m http.server` over a directory of exports while a
freshly pruned one is being checked.

Padding (`batched` only): Gemma 3 causal-LM exports honour `attention_mask`
/ `position_ids`, so rows are left-padded and only one logits position is
kept. The Gemma 4 multimodal export does not — padded rows read the pads as
context and the answers are wrong (noul questions collapsed to ~0.97 with
candidate mass 0.001). For it the batch is right-padded and logits are kept
at every position (capped at 64M elements, above that it falls back to
sequential). `shared` never pads, so it is not affected.

Measured (M4, 16 GB, Chromium WebGPU, Gemma 4 E2B q4f16; `shared` cold =
state decoded in this request, warm = state already resident):

| request | tokens (shared / batched) | shared cold | shared warm | batched | sequential |
|---|---|---|---|---|---|
| ticket, 5 questions, 90-token state | 313 / 673 | 2.8 s | 2.5 s | 4.6 s | 4.2 s |
| contract, 8 questions, 232-token state | 639 / 2,263 | 5.2 s | 3.1 s | 12.1 s | 13.1 s |

gemma-3-270m on the contract: shared 1.7 s cold / 0.9 s warm, sequential
3.2 s. The remaining cost per branch on E2B is ~90 ms fixed + ~7 ms per
token (ORT WebGPU at small batch), which is why the ticket's five ~45-token
branches take 2.5 s even with the state resident.

`shared` and `sequential` agree to max |Δp| 1e-5 on the contract (logits
differ by up to 0.8 in fp16 where the probability is already saturated);
Gemma 4 E2B answers in the browser match the native llama.cpp run
(`refund_requested` 0.010 vs 0.037 Q4_0; the rest within a few 1e-3).
`batched` on Gemma 4 is the exception: the right-padded batch moves noul
logits by up to ~2 (`refund_requested` 0.002), same answers. With the
262k vocabulary the ticket's 673 × 262,144 logits exceeded the cap, so
`batched` silently ran as `sequential`; the 25k-token export is the first
to actually run it in one forward.
