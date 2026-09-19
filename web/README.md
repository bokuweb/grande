# grande web demo

Static page: `index.html` + `app.js` + `engine.js` + `presets.js` + `pkg/`
(wasm-bindgen output of `crates/grande-web`). transformers.js is loaded from
jsDelivr; model weights stream from the Hugging Face Hub and are cached by
the browser. Nothing is uploaded.

```bash
# build the wasm (once per grande-core change)
cargo build --release -p grande-web --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg --out-name grande target/wasm32-unknown-unknown/release/grande_web.wasm
wasm-opt -Oz -o web/pkg/grande_bg.wasm web/pkg/grande_bg.wasm

# serve
python3 -m http.server 8765 --directory web
open "http://localhost:8765/?model=gemma-3-270m"
```

`?model=` picks `gemma-3-270m` (0.27 GB, smoke test), `gemma-3-1b` (0.76 GB,
default) or `gemma-4-e2b` (3.1 GB; half of it is the per-layer embedding
table). Requires WebGPU (Chrome / Edge, Safari 26+).

What runs where:

- `grande-core` (wasm): request validation, the rendered layout (Gemma 3 and
  Gemma 4 turn markers), option labels, temperature / softmax / confidence,
  the TypeSafe-shaped response. Same crate as the native runtime.
- `engine.js`: tokenizes the rendered segments (control tokens in caller text
  are neutralized the same way as native), runs one batched forward with
  `num_logits_to_keep = 1`, reads the label logits and candidate mass.

Modes: `batched` is one forward where every row re-reads the state (ORT's
GroupQueryAttention refuses a batched multi-token continuation from a cache,
so state-once + one-forward is not available in the browser yet);
`sequential` is one forward per question, for comparison.
