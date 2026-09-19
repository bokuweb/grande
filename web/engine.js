// grande in the browser: transformers.js (ONNX Runtime Web, WebGPU) does the forward
// pass, the wasm build of grande-core does everything that must match the native
// runtime (validation, layout, labels, softmax / temperature / confidence, response).
//
//   const engine = await loadEngine({ transformers, model: "gemma-3-1b", onProgress });
//   const resp = await engine.answer(request, { temperature: 1, mode: "shared", calibrate: true });
//
// Zero-shot label readout: the state is decoded once into a resident KV cache, every
// question continues from it as an isolated branch ("state, then this question"), and
// only the label-token logits at each branch's last position are read. No generation.
// Modes: "shared" (above), "batched" (one forward where every row re-reads the
// state) and "sequential" (one forward per question) are kept for comparison.
// The trained pointer model (a hidden-state export without a KV cache) always
// runs batched. `calibrate` subtracts the model's prior over the options (the
// same questions over a content-free state) before the softmax.

import init, * as grande from "./pkg/grande.js";
import { idbCache } from "./cache.js";

const GEMMA3 = { layout: "label", turn_start: "<start_of_turn>", turn_end: "<end_of_turn>", user: "user", model: "model" };
const GEMMA4 = { layout: "label", turn_start: "<|turn>", turn_end: "<turn|>", user: "user", model: "model" };

export const MODELS = {
  // The same trained checkpoint on grande's own wgpu engine (crates/grande-wgpu):
  // state + every question in ONE forward pass with a block-causal mask, no
  // ONNX Runtime. f16 safetensors served from this site.
  "grande-270m-ja-wgpu": { id: "grande-270m-ja-wgpu", local: true, kind: "wgpu", readout: "pointer", dtype: "f16", layout: { layout: "pointer", state: "<unused0>", question: "<unused1>", opt: "<unused2>", opt_end: "<unused3>", decide: "<unused4>" }, size: "0.32 GB", note: "trained, wgpu engine: one pass" },
  // Gemma 4 E2B (Q4_0, from the GGUF via tools/export_wgpu_gguf.py) on the same
  // engine: zero-shot label readout, state + every question in one pass. The
  // 1.3 GB per-layer token table stays in JS memory and is gathered per request.
  // Served from ./models/ when present (local development), otherwise from
  // the Hugging Face repo (GitHub Pages caps a site at 1 GB and release
  // assets are not CORS-enabled).
  "gemma-4-e2b-wgpu": { id: "gemma-4-e2b-wgpu", local: true, hub: "bokuweb/gemma-4-E2B-it-grande-wgpu", kind: "wgpu", readout: "label", manifest: true, dtype: "q4", layout: GEMMA4, size: "2.8 GB", note: "wgpu engine: one pass" },
  // Trained pointer head (JNLI 0.71 / JCQA 0.71, 12k records) on a Gemma 3 270M
  // backbone with embedding rows pruned to a Japanese + English corpus. Served
  // from this site (./models/, fetched from a GitHub release at build time).
  "grande-270m-ja": { id: "grande-270m-ja", local: true, kind: "pointer", layout: { layout: "pointer", state: "<unused0>", question: "<unused1>", opt: "<unused2>", opt_end: "<unused3>", decide: "<unused4>" }, dtype: "q8", size: "0.21 GB", note: "default, trained, fast" },
  "gemma-3-270m": { id: "onnx-community/gemma-3-270m-it-ONNX", kind: "causal", layout: GEMMA3, dtype: "q4f16", size: "0.27 GB", note: "smoke test only" },
  "gemma-3-1b": { id: "onnx-community/gemma-3-1b-it-ONNX", kind: "causal", layout: GEMMA3, dtype: "q4f16", size: "0.76 GB", note: "fast" },
  "gemma-4-e2b": { id: "onnx-community/gemma-4-E2B-it-ONNX", kind: "gemma4", layout: GEMMA4, dtype: "q4f16", size: "3.4 GB", note: "default", padding: "right" },
  "gemma-4-e4b": { id: "onnx-community/gemma-4-E4B-it-ONNX", kind: "gemma4", layout: GEMMA4, dtype: "q4f16", size: "5.2 GB", note: "larger", padding: "right" },
};

const ZWNJ = "‌";
// Mirror of grande-llama's neutralize_specials: caller text can never tokenize
// into a control token, so option boundaries cannot be forged.
export function neutralize(text) {
  return text.replace(/<(?=[A-Za-z|/])/g, "<" + ZWNJ);
}

function segmentsToText(segments, bos) {
  let out = "";
  for (const s of segments) {
    if (s.kind === "bos") out += bos;
    else if (s.kind === "special") out += s.value;
    else out += neutralize(s.value);
  }
  return out;
}

function logSumExp(row, ids) {
  let m = -Infinity;
  if (ids) for (const i of ids) m = Math.max(m, row[i]);
  else for (let i = 0; i < row.length; i++) if (row[i] > m) m = row[i];
  let s = 0;
  if (ids) for (const i of ids) s += Math.exp(row[i] - m);
  else for (let i = 0; i < row.length; i++) s += Math.exp(row[i] - m);
  return m + Math.log(s);
}

// Minimal safetensors reader for the pointer head (q.weight [dp,d], q.bias, k.weight, k.bias).
async function loadHead(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const n = Number(new DataView(buf).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  const base = 8 + n;
  const tensor = (name) => {
    const t = header[name];
    if (!t) throw new Error(`head: missing ${name}`);
    if (t.dtype !== "F32") throw new Error(`head: ${name} is ${t.dtype}, expected F32`);
    const [a, b] = t.data_offsets;
    return { shape: t.shape, data: new Float32Array(buf.slice(base + a, base + b)) };
  };
  const wq = tensor("q.weight"), bq = tensor("q.bias"), wk = tensor("k.weight"), bk = tensor("k.bias");
  return { dp: wq.shape[0], d: wq.shape[1], wq: wq.data, bq: bq.data, wk: wk.data, bk: bk.data };
}

function project(head, w, b, h, off) {
  const { dp, d } = head;
  const out = new Float32Array(dp);
  for (let r = 0; r < dp; r++) {
    let acc = b[r];
    const wr = r * d;
    for (let c = 0; c < d; c++) acc += w[wr + c] * h[off + c];
    out[r] = acc;
  }
  return out;
}

// Pointer readout: logits_i = (W_k h_opt_i + b_k) · (W_q h_decide + b_q) / sqrt(dp).
function pointerLogits(head, hidden, decideOff, optOffs) {
  const q = project(head, head.wq, head.bq, hidden, decideOff);
  const scale = 1 / Math.sqrt(head.dp);
  return optOffs.map((o) => {
    const k = project(head, head.wk, head.bk, hidden, o);
    let dot = 0;
    for (let i = 0; i < head.dp; i++) dot += k[i] * q[i];
    return dot * scale;
  });
}

// fetch through the same IndexedDB cache transformers.js uses, with progress.
async function cachedFetch(url, onProgress) {
  const hit = await idbCache.match(url);
  if (hit) return hit;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.({ status: "progress", file: url.split("/").slice(-2).join("/"), loaded, total });
  }
  const blob = new Blob(chunks);
  const out = new Response(blob, { status: 200, headers: { "content-length": String(blob.size) } });
  await idbCache.put(url, out.clone()).catch(() => {});
  return out;
}

// An exported model directory (tools/export_wgpu_gguf.py): manifest.json lists
// tensors per file; each file is fetched (and cached) once and pushed into the
// engine tensor by tensor, so only one file is in memory at a time. The
// per-layer token table (Gemma 4) is kept in JS and gathered per request.
async function loadManifest(base, config, onProgress) {
  const manifest = await (await fetch(`${base}manifest.json`)).json();
  const loader = await grande.WgpuLoader.open(config);
  const total = manifest.files.length + (manifest.per_layer_table ? 1 : 0);
  let done = 0;
  for (const file of manifest.files) {
    const buf = new Uint8Array(await (await cachedFetch(`${base}${file.path}`, onProgress)).arrayBuffer());
    for (const t of file.tensors) {
      loader.push(t.name, t.dtype, Uint32Array.from(t.shape), buf.subarray(t.offset, t.offset + t.nbytes),
        t.scales_nbytes ? buf.subarray(t.scales_offset, t.scales_offset + t.scales_nbytes) : new Uint8Array(0));
    }
    onProgress?.({ status: "upload", file: file.path, loaded: ++done, total });
  }
  let plTable = null;
  const pl = manifest.per_layer_table;
  if (pl) {
    const buf = new Uint8Array(await (await cachedFetch(`${base}${pl.path}`, onProgress)).arrayBuffer());
    plTable = { dtype: pl.dtype, rows: pl.shape[0], width: pl.shape[1], data: buf.subarray(pl.offset, pl.offset + pl.nbytes),
      scales: buf.subarray(pl.scales_offset, pl.scales_offset + pl.scales_nbytes) };
    onProgress?.({ status: "upload", file: pl.path, loaded: ++done, total });
  }
  onProgress?.({ status: "ready" });
  return { gpu: loader.finish(4096, 256), plTable };
}

// f16 <-> f32 without Float16Array (Chrome < 135, Firefox < 129).
function f16ToF32(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
const f32Buf = new Float32Array(1), u32Buf = new Uint32Array(f32Buf.buffer);
function f32ToF16(v) {
  f32Buf[0] = v;
  const x = u32Buf[0], sign = (x >>> 16) & 0x8000;
  let e = ((x >>> 23) & 0xff) - 127 + 15, m = x & 0x7fffff;
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) { if (e < -10) return sign; m = (m | 0x800000) >> (1 - e); return sign | ((m + 0x1000) >> 13); }
  return sign | (e << 10) | ((m + 0x1000) >> 13);
}

// Gather + dequantize the per-layer token table rows for `ids`: f16
// little-endian [ids][width], what WgpuEngine.evaluate_rows expects.
function gatherPerLayer(table, ids) {
  const { width, data, scales, dtype } = table;
  const blocks = width / 32;
  const hasF16 = typeof Float16Array !== "undefined";
  const out = hasF16 ? new Float16Array(ids.length * width) : new Uint16Array(ids.length * width);
  const put = hasF16 ? (i, v) => { out[i] = v; } : (i, v) => { out[i] = f32ToF16(v); };
  const sc = new Uint16Array(scales.buffer, scales.byteOffset, scales.byteLength / 2);
  const bpb = dtype === "q8" ? 32 : 16; // payload bytes per block
  for (let t = 0; t < ids.length; t++) {
    const row = ids[t];
    let o = t * width;
    for (let b = 0; b < blocks; b++) {
      const d = f16ToF32(sc[row * blocks + b]);
      const p = (row * blocks + b) * bpb;
      if (dtype === "q8") {
        for (let j = 0; j < 32; j++) put(o++, ((data[p + j] << 24) >> 24) * d);
      } else {
        for (let j = 0; j < 16; j++) put(o + j, ((data[p + j] & 0xf) - 8) * d);
        for (let j = 0; j < 16; j++) put(o + 16 + j, ((data[p + j] >> 4) - 8) * d);
        o += 32;
      }
    }
  }
  return new Uint8Array(out.buffer);
}

// Where a `hub`-backed model's files are right now: "here" (./models/),
// "hub" (the Hugging Face repo) or null (not published yet). Cached models
// count as available.
export async function whereIs(spec) {
  const head = (url) => fetch(url, { method: "HEAD", cache: "no-store" }).then((r) => r.ok).catch(() => false);
  if (await head(new URL(`./models/${spec.id}/config.json`, location.href).href)) return "here";
  if (!spec.hub) return null;
  if (await head(`https://huggingface.co/${spec.hub}/resolve/main/config.json`)) return "hub";
  return null;
}

let wasmReady = null;

export async function loadEngine({ transformers, model = "gemma-3-1b", device = "webgpu", onProgress } = {}) {
  // Fetch the wasm with a cache-busting query: GitHub Pages caches for 10 min
  // and a stale wasm with a fresh grande.js fails at Table.grow.
  wasmReady ??= init({ module_or_path: new URL(`./pkg/grande_bg.wasm?v=${Date.now()}`, import.meta.url) });
  await wasmReady;
  const spec = MODELS[model];
  if (!spec) throw new Error(`unknown model ${model}`);
  // Weights persist in IndexedDB across visits (see cache.js); ask the browser
  // not to evict them under storage pressure.
  transformers.env.useCustomCache = true;
  transformers.env.customCache = idbCache;
  navigator.storage?.persist?.().catch(() => {});
  const { AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Gemma4ForConditionalGeneration, Tensor, DynamicCache } = transformers;

  let tok, net, head = null, idMap = null, gpu = null, plTable = null;
  if (spec.kind === "wgpu") {
    let base = new URL(`./models/${spec.id}/`, location.href).href;
    let here = true;
    if (spec.hub) {
      const probe = await fetch(`${base}config.json`, { method: "HEAD", cache: "no-store" }).catch(() => null);
      if (!probe?.ok) {
        base = `https://huggingface.co/${spec.hub}/resolve/main/`;
        here = false;
        const hub = await fetch(`${base}config.json`, { method: "HEAD", cache: "no-store" }).catch(() => null);
        if (!hub?.ok) throw new Error(`${spec.id}: not in ./models/ and https://huggingface.co/${spec.hub} is not published (see web/README.md)`);
      }
    }
    if (here) {
      transformers.env.allowLocalModels = true;
      transformers.env.localModelPath = "./models/";
      transformers.env.allowRemoteModels = false;
      tok = await AutoTokenizer.from_pretrained(spec.id, { progress_callback: onProgress });
      transformers.env.allowRemoteModels = true;
    } else {
      tok = await AutoTokenizer.from_pretrained(spec.hub, { progress_callback: onProgress });
    }
    const config = await (await fetch(`${base}config.json`)).text();
    if (spec.manifest) {
      ({ gpu, plTable } = await loadManifest(base, config, onProgress));
    } else {
      head = await loadHead(`${base}head.safetensors`);
      const weights = new Uint8Array(await (await cachedFetch(`${base}model.safetensors`, onProgress)).arrayBuffer());
      onProgress?.({ status: "ready" });
      gpu = await grande.WgpuEngine.load(config, weights, 4096, 256);
    }
    await gpu.warmup();
  } else if (spec.local) {
    // Same-origin model directory. transformers.js only probes local files when
    // localModelPath is NOT an absolute URL (its metadata check skips http(s)
    // paths), so keep it page-relative.
    transformers.env.allowLocalModels = true;
    transformers.env.localModelPath = "./models/";
    transformers.env.allowRemoteModels = false;
    const base = new URL(`./models/${spec.id}/`, location.href).href;
    tok = await AutoTokenizer.from_pretrained(spec.id, { progress_callback: onProgress });
    net = await transformers.AutoModel.from_pretrained(spec.id, { dtype: spec.dtype, device, progress_callback: onProgress });
    transformers.env.allowRemoteModels = true;
    head = await loadHead(`${base}head.safetensors`);
    idMap = new Int32Array(await (await fetch(`${base}id_map.bin`)).arrayBuffer());
  } else if (spec.kind === "gemma4") {
    const processor = await AutoProcessor.from_pretrained(spec.id, { progress_callback: onProgress });
    tok = processor.tokenizer;
    net = await Gemma4ForConditionalGeneration.from_pretrained(spec.id, { dtype: spec.dtype, device, progress_callback: onProgress });
  } else {
    tok = await AutoTokenizer.from_pretrained(spec.id, { progress_callback: onProgress });
    net = await AutoModelForCausalLM.from_pretrained(spec.id, { dtype: spec.dtype, device, progress_callback: onProgress });
  }
  const bos = tok.bos_token ?? "<bos>";
  const LABELS = grande.labels();
  const labelIds = [];
  for (const ch of LABELS) {
    const ids = tok.encode(ch, { add_special_tokens: false });
    if (ids.length !== 1) break;
    labelIds.push(ids[0]);
  }

  function readRow(logits, dims, b, position, k) {
    const [, K, V] = dims;
    const last = logits.subarray((b * K + position) * V, (b * K + position + 1) * V);
    const ids = labelIds.slice(0, k);
    const z = ids.map((i) => Number(last[i]));
    const mass = Math.exp(logSumExp(last, ids) - logSumExp(last));
    return { logits: z, candidate_mass: mass };
  }

  function rowsFromLogits(logits, dims, keysPerRow) {
    // logits [B, K, V]; read the last kept position per row.
    return keysPerRow.map((k, b) => readRow(logits, dims, b, dims[1] - 1, k));
  }

  // One batched forward. Two flavours:
  //  - left padding + num_logits_to_keep = 1: the 262k-vocab projection runs at one
  //    position per row. Works when the graph honours attention_mask / position_ids
  //    for padded rows (Gemma 3 causal LM).
  //  - right padding + logits at every position: pads sit after the real tokens, so
  //    causal attention never sees them even if the graph ignores the mask (the
  //    Gemma 4 multimodal export). Costs B × L × V logits, so it is capped.
  const RIGHT_PAD_MAX_LOGITS = 64 * 1024 * 1024; // elements; ~128 MB in fp16

  // Every output (logits and the present.* cache, which lives on the GPU) is
  // released once read; only the resident state cache outlives a request.
  async function disposeOutputs(out) {
    for (const t of Object.values(out)) if (t?.dispose) await t.dispose();
  }
  async function batched(texts, keysPerRow) {
    if (spec.padding === "right") return batchedRight(texts, keysPerRow);
    tok.padding_side = "left";
    const inputs = tok(texts, { padding: true, truncation: false, add_special_tokens: false });
    const [B, L] = inputs.attention_mask.dims;
    const mask = inputs.attention_mask.data;
    const pos = new BigInt64Array(B * L);
    for (let b = 0; b < B; b++) {
      let c = 0n;
      for (let i = 0; i < L; i++) {
        pos[b * L + i] = Number(mask[b * L + i]) ? c : 0n;
        if (Number(mask[b * L + i])) c += 1n;
      }
    }
    const position_ids = new Tensor("int64", pos, [B, L]);
    const out = await net.forward({ ...inputs, position_ids, num_logits_to_keep: new Tensor("int64", [1n], []) });
    let tokens = 0;
    for (let i = 0; i < mask.length; i++) if (Number(mask[i])) tokens++;
    const rows = rowsFromLogits(out.logits.data, out.logits.dims, keysPerRow);
    await disposeOutputs(out);
    return { rows, tokens };
  }

  async function batchedRight(texts, keysPerRow) {
    tok.padding_side = "right";
    const inputs = tok(texts, { padding: true, truncation: false, add_special_tokens: false });
    const [B, L] = inputs.attention_mask.dims;
    const V = net.config.text_config?.vocab_size ?? net.config.vocab_size ?? 262144;
    if (B * L * V > RIGHT_PAD_MAX_LOGITS) return sequential(texts, keysPerRow);
    const mask = inputs.attention_mask.data;
    const lens = [];
    let tokens = 0;
    for (let b = 0; b < B; b++) {
      let n = 0;
      for (let i = 0; i < L; i++) if (Number(mask[b * L + i])) n++;
      lens.push(n);
      tokens += n;
    }
    const out = await net.forward({ ...inputs, num_logits_to_keep: new Tensor("int64", [BigInt(L)], []) });
    const rows = keysPerRow.map((k, b) => readRow(out.logits.data, out.logits.dims, b, lens[b] - 1, k));
    await disposeOutputs(out);
    return { rows, tokens };
  }

  async function sequential(texts, keysPerRow) {
    const rows = [];
    let tokens = 0;
    for (let b = 0; b < texts.length; b++) {
      const inputs = tok(texts[b], { add_special_tokens: false });
      const out = await net.forward({ ...inputs, num_logits_to_keep: new Tensor("int64", [1n], []) });
      tokens += inputs.input_ids.dims[1];
      rows.push(rowsFromLogits(out.logits.data, out.logits.dims, [keysPerRow[b]])[0]);
      await disposeOutputs(out);
    }
    return { rows, tokens };
  }

  // Shared state: the browser counterpart of grande-llama's resident prefix.
  // The state is decoded once into a KV cache and stays resident; every branch
  // then continues from that cache, so the state is never re-read and a second
  // request over the same state skips it entirely. A branch attends to the
  // state and to its own tokens only, exactly the native block-causal layout.
  //
  // Branches run one forward each. ORT's GroupQueryAttention requires
  // "batch_size must be 1 when sequence_length > 1 and past context is given",
  // and a refused run leaves the session unusable, so the tiled-cache single
  // forward is not attempted. Each branch forward is short (its own tokens
  // only), so the cost is the per-dispatch overhead, not compute.
  let resident = null; // { text, n, kv: DynamicCache }

  function cacheFromOutput(out) {
    const entries = {};
    for (const name in out) {
      if (!name.startsWith("present")) continue;
      entries[name.replace("present", "past_key_values")] = out[name];
    }
    return new DynamicCache(entries);
  }

  async function ensureResident(prefixText) {
    if (resident?.text === prefixText) return { ...resident, warm: true };
    if (resident) { await resident.kv.dispose(); resident = null; }
    const inputs = tok(prefixText, { add_special_tokens: false });
    const out = await net.forward({ ...inputs, num_logits_to_keep: new Tensor("int64", [1n], []) });
    out.logits.dispose?.();
    resident = { text: prefixText, n: inputs.input_ids.dims[1], kv: cacheFromOutput(out) };
    return { ...resident, warm: false };
  }

  async function shared(prefixText, branchTexts, keysPerRow) {
    const { kv, n: P, warm } = await ensureResident(prefixText);
    const rows = [];
    let tokens = 0;
    for (let b = 0; b < branchTexts.length; b++) {
      const inputs = tok(branchTexts[b], { add_special_tokens: false });
      const Q = inputs.input_ids.dims[1];
      const attention_mask = new Tensor("int64", new BigInt64Array(P + Q).fill(1n), [1, P + Q]);
      const out = await net.forward({ input_ids: inputs.input_ids, attention_mask, past_key_values: kv, num_logits_to_keep: new Tensor("int64", [1n], []) });
      tokens += Q;
      rows.push(rowsFromLogits(out.logits.data, out.logits.dims, [keysPerRow[b]])[0]);
      await disposeOutputs(out);
    }
    return { rows, tokens: tokens + (warm ? 0 : P), forwards: branchTexts.length + (warm ? 0 : 1), warm, state_tokens: P };
  }

  // Tokenize rendered segments one by one (mirror of grande-core's pack): text
  // segments never parse control tokens, specials resolve to one id, and the
  // wanted positions (each </opt>, then <decide>) are the last token of their
  // segment.
  function packSegments(segments) {
    const ids = [];
    const ends = [];
    for (const s of segments) {
      if (s.kind === "bos") ids.push(...tok.encode(bos, { add_special_tokens: false }));
      else if (s.kind === "special") {
        const t = tok.encode(s.value, { add_special_tokens: false });
        if (t.length !== 1) throw new Error(`${s.value} is not one token`);
        ids.push(t[0]);
      } else ids.push(...tok.encode(neutralize(s.value), { add_special_tokens: false }));
      ends.push(ids.length - 1);
    }
    return { ids, ends };
  }

  // Per-layer embedding rows for a packed request (Gemma 4), or undefined.
  function perLayerRows(prefix, branches) {
    if (!plTable) return undefined;
    const ids = prefix.concat(...branches.map((b) => b.tokens));
    return gatherPerLayer(plTable, ids);
  }

  // Zero-shot label readout on the wgpu engine: one pass, the full-vocabulary
  // logits at each branch's last token come back, and the option labels'
  // logits plus the candidate mass are read from them (same as the ONNX path).
  async function labelWgpu(rendered) {
    const prefix = packSegments(rendered.prefix).ids;
    const branches = rendered.branches.map((b) => {
      const { ids } = packSegments(b.segments);
      return { tokens: ids, want: [ids.length - 1], k: b.keys.length };
    });
    const flat = await gpu.evaluate_rows(Uint32Array.from(prefix), JSON.stringify(branches.map(({ tokens, want }) => ({ tokens, want }))), "logits",
      perLayerRows(prefix, branches));
    const V = flat.length / branches.length;
    const rows = branches.map((b, i) => readRow(flat, [branches.length, 1, V], i, 0, b.k));
    const tokens = prefix.length + branches.reduce((n, b) => n + b.tokens.length, 0);
    return { rows, tokens, prefixTokens: prefix.length };
  }

  // Pointer readout on the wgpu engine: prefix once, every branch isolated by
  // the mask, one pass. Rows come back branch by branch, each option end then
  // the decide token, as `d`-wide hidden states.
  async function pointerWgpu(rendered) {
    const prefix = packSegments(rendered.prefix).ids;
    const branches = rendered.branches.map((b) => {
      const { ids, ends } = packSegments(b.segments);
      const want = b.marks.map(([seg, mark]) => (mark === "Last" ? ids.length - 1 : ends[seg]));
      return { tokens: ids, want, k: b.keys.length };
    });
    const flat = await gpu.evaluate_rows(Uint32Array.from(prefix), JSON.stringify(branches.map(({ tokens, want }) => ({ tokens, want }))), "hidden",
      perLayerRows(prefix, branches));
    const D = head.d;
    let off = 0;
    const rows = branches.map((b) => {
      const offs = b.want.map((_, i) => (off + i) * D);
      off += b.want.length;
      const decide = offs[offs.length - 1];
      return { logits: pointerLogits(head, flat, decide, offs.slice(0, -1)), candidate_mass: null };
    });
    const tokens = prefix.length + branches.reduce((n, b) => n + b.tokens.length, 0);
    return { rows, tokens, prefixTokens: prefix.length };
  }

  // Pointer readout over one batched forward (right padding; pads sit after the
  // real tokens so causal attention never sees them). Every row re-reads the state.
  async function pointerBatched(rendered) {
    const prefix = packSegments(rendered.prefix).ids;
    const rows = rendered.branches.map((b) => {
      const { ids, ends } = packSegments(b.segments);
      const want = b.marks.map(([seg, mark]) => (mark === "Last" ? ids.length - 1 : ends[seg]) + prefix.length);
      return { ids: [...prefix, ...ids], want, k: b.keys.length };
    });
    const B = rows.length;
    const L = Math.max(...rows.map((r) => r.ids.length));
    const input = new BigInt64Array(B * L);
    const mask = new BigInt64Array(B * L);
    let tokens = 0;
    rows.forEach((r, b) => {
      r.ids.forEach((id, i) => {
        input[b * L + i] = BigInt(idMap[id]);
        mask[b * L + i] = 1n;
      });
      tokens += r.ids.length;
    });
    const out = await net({ input_ids: new Tensor("int64", input, [B, L]), attention_mask: new Tensor("int64", mask, [B, L]) });
    const hs = out.last_hidden_state;
    const [, , D] = hs.dims;
    const data = hs.data instanceof Float32Array ? hs.data : Float32Array.from(hs.data, Number);
    const result = rows.map((r, b) => {
      const offs = r.want.map((p) => (b * L + p) * D);
      const decide = offs[offs.length - 1];
      const logits = pointerLogits(head, data, decide, offs.slice(0, -1));
      return { logits, candidate_mass: null };
    });
    hs.dispose?.();
    return { rows: result, tokens, prefixTokens: prefix.length };
  }

  let queue = Promise.resolve();
  const enqueue = (job) => { const p = queue.then(job, job); queue = p.catch(() => {}); return p; };

  // Rows for every branch of a rendered request. Label readout honours
  // `mode`; the pointer readouts have one path each.
  async function rowsFor(rendered, mode) {
    if (spec.kind === "pointer" || spec.kind === "wgpu") {
      for (const b of rendered.branches) if (spec.readout === "label" && b.keys.length > labelIds.length) throw new Error(`a question has ${b.keys.length} options; this tokenizer supports ${labelIds.length} single-token labels`);
      const { rows, tokens, prefixTokens } = spec.kind !== "wgpu" ? await pointerBatched(rendered) : spec.readout === "label" ? await labelWgpu(rendered) : await pointerWgpu(rendered);
      return { rows, tokens, forwards: 1, state_tokens: prefixTokens, mode: spec.kind === "wgpu" ? "packed" : "batched" };
    }
    const prefix = segmentsToText(rendered.prefix, bos);
    const branchTexts = rendered.branches.map((b) => segmentsToText(b.segments, bos));
    const texts = branchTexts.map((t) => prefix + t);
    const keys = rendered.branches.map((b) => b.keys.length);
    for (const k of keys) if (k > labelIds.length) throw new Error(`a question has ${k} options; this tokenizer supports ${labelIds.length} single-token labels`);
    if (mode === "sequential") return { ...(await sequential(texts, keys)), forwards: texts.length, mode };
    if (mode === "batched") return { ...(await batched(texts, keys)), forwards: 1, mode };
    return { ...(await shared(prefix, branchTexts, keys)), mode };
  }

  // Contextual calibration (Zhao et al. 2021): the same branches over the
  // content-free state "N/A" give the model's prior over the options, which
  // grande.answer subtracts in logit space. The prior depends on the question
  // alone, so it is cached per rendered branch; a fixed question set over
  // changing states pays for it once. Never runs through `shared`, so the
  // live state stays resident.
  const CONTENT_FREE = grande.content_free_state();
  const baselineCache = new Map();
  async function baselineRows(request, rendered, mode, contentFree) {
    const keys = rendered.branches.map((b) => JSON.stringify([contentFree, b.segments, b.keys]));
    const missing = keys.map((k, i) => (baselineCache.has(k) ? -1 : i)).filter((i) => i >= 0);
    let forwards = 0, tokens = 0;
    if (missing.length) {
      const cf = JSON.parse(grande.render(JSON.stringify({ ...request, state: contentFree }), JSON.stringify(spec.layout)));
      const r = await rowsFor({ prefix: cf.prefix, branches: missing.map((i) => cf.branches[i]) }, mode === "shared" ? "batched" : mode);
      missing.forEach((i, j) => baselineCache.set(keys[i], r.rows[j]));
      forwards = r.forwards;
      tokens = r.tokens;
    }
    return { rows: keys.map((k) => baselineCache.get(k)), forwards, tokens };
  }

  // `calibrate`: false, true ("N/A" as the content-free state) or a string to
  // use as the content-free state instead.
  async function answerNow(request, { temperature = 1.0, mode = "shared", calibrate = false } = {}) {
    const reqJson = JSON.stringify(request);
    const rendered = JSON.parse(grande.render(reqJson, JSON.stringify(spec.layout)));
    const t0 = performance.now();
    // Baseline first: on a cold state its forwards would otherwise sit
    // between the state and its questions.
    const base = calibrate ? await baselineRows(request, rendered, mode, typeof calibrate === "string" ? calibrate : CONTENT_FREE) : null;
    const r = await rowsFor(rendered, mode);
    const ms = performance.now() - t0;
    const { rows } = r;
    const tokens = r.tokens + (base?.tokens ?? 0);
    const resp = JSON.parse(grande.answer(reqJson, JSON.stringify(rows), temperature, spec.id, tokens, base ? JSON.stringify(base.rows) : undefined));
    const stateTokens = r.state_tokens ?? tok.encode(segmentsToText(rendered.prefix, bos), { add_special_tokens: false }).length;
    return { ...resp, usage: { ...resp.usage, state_tokens: stateTokens, questions: rows.length, mode: r.mode, ms, forwards: r.forwards + (base?.forwards ?? 0),
        ...(r.warm === undefined ? {} : { state_resident: r.warm }), ...(base ? { calibrated: "contextual", baseline_forwards: base.forwards } : {}) },
      diagnostics: { candidate_mass: Object.fromEntries(rendered.branches.map((b, i) => [b.id, rows[i].candidate_mass])), rows, ...(base ? { baseline: base.rows } : {}) } };
  }

  return {
    model, spec, tokenizer: tok, net, device,
    labels: LABELS.slice(0, labelIds.length),
    answer: (request, opts) => enqueue(() => answerNow(request, opts)),
    render: (request) => JSON.parse(grande.render(JSON.stringify(request), JSON.stringify(spec.layout))),
  };
}
