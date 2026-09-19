// grande in the browser: transformers.js (ONNX Runtime Web, WebGPU) does the forward
// pass, the wasm build of grande-core does everything that must match the native
// runtime (validation, layout, labels, softmax / temperature / confidence, response).
//
//   const engine = await loadEngine({ transformers, model: "gemma-3-1b", onProgress });
//   const resp = await engine.answer(request, { temperature: 1, mode: "shared" });
//
// Zero-shot label readout: the state is decoded once into a resident KV cache, every
// question continues from it as an isolated branch ("state, then this question"), and
// only the label-token logits at each branch's last position are read. No generation.
// Modes: "shared" (above), "batched" (one forward where every row re-reads the
// state) and "sequential" (one forward per question) are kept for comparison.
// The trained pointer model (a hidden-state export without a KV cache) always
// runs batched.

import init, * as grande from "./pkg/grande.js";
import { idbCache } from "./cache.js";

const GEMMA3 = { layout: "label", turn_start: "<start_of_turn>", turn_end: "<end_of_turn>", user: "user", model: "model" };
const GEMMA4 = { layout: "label", turn_start: "<|turn>", turn_end: "<turn|>", user: "user", model: "model" };

export const MODELS = {
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

  let tok, net, head = null, idMap = null;
  if (spec.local) {
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

  async function answerNow(request, { temperature = 1.0, mode = "shared" } = {}) {
    const reqJson = JSON.stringify(request);
    const rendered = JSON.parse(grande.render(reqJson, JSON.stringify(spec.layout)));
    if (spec.kind === "pointer") {
      const t0 = performance.now();
      const { rows, tokens, prefixTokens } = await pointerBatched(rendered);
      const ms = performance.now() - t0;
      const resp = JSON.parse(grande.answer(reqJson, JSON.stringify(rows), temperature, spec.id, tokens));
      return { ...resp, usage: { ...resp.usage, state_tokens: prefixTokens, questions: rows.length, mode: "batched", ms, forwards: 1 },
        diagnostics: { candidate_mass: {}, rows } };
    }
    const prefix = segmentsToText(rendered.prefix, bos);
    const branchTexts = rendered.branches.map((b) => segmentsToText(b.segments, bos));
    const texts = branchTexts.map((t) => prefix + t);
    const keys = rendered.branches.map((b) => b.keys.length);
    for (const k of keys) if (k > labelIds.length) throw new Error(`a question has ${k} options; this tokenizer supports ${labelIds.length} single-token labels`);
    const t0 = performance.now();
    let r;
    if (mode === "sequential") r = { ...(await sequential(texts, keys)), forwards: texts.length };
    else if (mode === "batched") r = { ...(await batched(texts, keys)), forwards: 1 };
    else r = await shared(prefix, branchTexts, keys);
    const ms = performance.now() - t0;
    const { rows, tokens } = r;
    const resp = JSON.parse(grande.answer(reqJson, JSON.stringify(rows), temperature, spec.id, tokens));
    const stateTokens = r.state_tokens ?? tok.encode(prefix, { add_special_tokens: false }).length;
    return { ...resp, usage: { ...resp.usage, state_tokens: stateTokens, questions: rows.length, mode, ms, forwards: r.forwards, ...(r.warm === undefined ? {} : { state_resident: r.warm }) },
      diagnostics: { candidate_mass: Object.fromEntries(rendered.branches.map((b, i) => [b.id, rows[i].candidate_mass])), rows } };
  }

  return {
    model, spec, tokenizer: tok, net, device,
    labels: LABELS.slice(0, labelIds.length),
    answer: (request, opts) => enqueue(() => answerNow(request, opts)),
    render: (request) => JSON.parse(grande.render(JSON.stringify(request), JSON.stringify(spec.layout))),
  };
}
