import { loadEngine, MODELS } from "./engine.js";
import { PRESETS } from "./presets.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const params = new URLSearchParams(location.search);
let engine = null;
let transformers = null;
let mode = "shared";

for (const [k, m] of Object.entries(MODELS)) {
  const o = document.createElement("option");
  o.value = k; o.textContent = `${k}  ·  ${m.size}  ·  ${m.note}`;
  if (k === (params.get("model") ?? "gemma-4-e2b")) o.selected = true;
  $("model").append(o);
}
for (const [k, p] of Object.entries(PRESETS)) {
  const o = document.createElement("option");
  o.value = k; o.textContent = p.label;
  $("preset").append(o);
}

// JSON syntax highlighting: a <pre> behind a transparent <textarea>. Not a
// parser, so plain-text state and half-typed JSON still render sensibly.
const TOKEN = /("(?:[^"\\]|\\.)*")(\s*:)?|(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)|\b(true|false|null)\b|([{}\[\],])/g;
export function highlightJson(text) {
  let out = "", last = 0;
  for (const m of text.matchAll(TOKEN)) {
    out += esc(text.slice(last, m.index));
    if (m[1] !== undefined) out += `<span class="${m[2] ? "tk-k" : "tk-s"}">${esc(m[1])}</span>${esc(m[2] ?? "")}`;
    else if (m[3] !== undefined) out += `<span class="tk-n">${esc(m[3])}</span>`;
    else if (m[4] !== undefined) out += `<span class="tk-b">${esc(m[4])}</span>`;
    else out += `<span class="tk-p">${esc(m[5])}</span>`;
    last = m.index + m[0].length;
  }
  // Trailing newline needs a visible line so the pre and the textarea stay the same height.
  return out + esc(text.slice(last)) + (text.endsWith("\n") ? " " : "");
}
function bindEditor(id) {
  const ta = $(id), hl = $(`${id}-hl`);
  const sync = () => { hl.innerHTML = highlightJson(ta.value); hl.scrollTop = ta.scrollTop; };
  ta.addEventListener("input", sync);
  ta.addEventListener("scroll", () => (hl.scrollTop = ta.scrollTop));
  ta.addEventListener("keydown", (e) => {
    if (e.key !== "Tab") return;
    e.preventDefault();
    ta.setRangeText("  ", ta.selectionStart, ta.selectionEnd, "end");
    sync();
  });
  return sync;
}
const syncState = bindEditor("state");
const syncQuestions = bindEditor("questions");

function setPreset(name) {
  const p = PRESETS[name];
  $("state").value = JSON.stringify(p.state, null, 2);
  $("questions").value = JSON.stringify(p.questions, null, 2);
  syncState(); syncQuestions();
}
$("preset").addEventListener("change", (e) => setPreset(e.target.value));
setPreset("contract");
$("preset").value = "contract";
$("temp").addEventListener("input", (e) => ($("tempv").textContent = Number(e.target.value).toFixed(1)));

for (const b of $("mode").querySelectorAll("button")) {
  b.addEventListener("click", () => {
    mode = b.dataset.mode;
    for (const x of $("mode").querySelectorAll("button")) x.classList.toggle("on", x === b);
  });
}

if (!navigator.gpu) { $("gpu").textContent = "WebGPU unavailable"; $("gpu").classList.add("warn"); }

function setStatus(text, pct, cls = "") {
  $("status").textContent = text;
  $("status").className = cls;
  $("barwrap").hidden = pct == null;
  $("bar").style.width = pct == null ? "0%" : `${Math.round(pct)}%`;
}

// Loaded engines stay resident, keyed by model, so switching back to a model
// that was already loaded is instant (no re-fetch, no re-init).
const engines = new Map();
let loading = false;

function selectModel() {
  const model = $("model").value;
  engine = engines.get(model) ?? null;
  window.grandeEngine = engine; // for the console
  $("run").disabled = !engine;
  if (engine) {
    setStatus(`${engine.spec.id} loaded (WebGPU, ${engine.spec.dtype})`, null, "ok");
    $("load").textContent = "Loaded";
    $("load").disabled = true;
  } else {
    setStatus(`Not loaded. ${MODELS[model].size} streams from the Hugging Face Hub and is cached by the browser.`);
    $("load").textContent = "Load model";
    $("load").disabled = loading;
  }
}
$("model").addEventListener("change", selectModel);
selectModel();

$("load").addEventListener("click", async () => {
  const model = $("model").value;
  if (engines.has(model) || loading) return;
  loading = true;
  $("load").disabled = true;
  $("model").disabled = true; // the progress line belongs to this model
  try {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser (Chrome / Edge, Safari 26+).");
    transformers ??= await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0");
    const files = new Map();
    setStatus(`Fetching ${MODELS[model].id}…`, 0);
    const loaded = await loadEngine({
      transformers, model,
      onProgress: (info) => {
        if (info.status === "progress") {
          files.set(info.file, [info.loaded ?? 0, info.total ?? 0]);
          let l = 0, t = 0;
          for (const [a, b] of files.values()) { l += a; t += b; }
          setStatus(`${info.file}  ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB`, t ? (100 * l) / t : 0);
        } else if (info.status === "ready") setStatus("Initializing…", 100);
      },
    });
    engines.set(model, loaded);
    loading = false;
    $("model").disabled = false;
    selectModel();
  } catch (e) {
    loading = false;
    $("model").disabled = false;
    setStatus(`Failed to load: ${e.message}`, null, "err");
    $("load").disabled = false;
    console.error(e);
  }
});

function parseState(text) {
  try { return JSON.parse(text); } catch { return text; }
}

async function run() {
  if (!engine || $("run").disabled) return;
  $("run").disabled = true;
  $("results").innerHTML = "";
  $("usage").innerHTML = `<span class="chip">Running…</span>`;
  try {
    const request = { state: parseState($("state").value), questions: JSON.parse($("questions").value) };
    const resp = await engine.answer(request, { temperature: Number($("temp").value), mode });
    renderResults(request, resp);
  } catch (e) {
    $("usage").innerHTML = `<span class="err">${esc(e.message)}</span>`;
    console.error(e);
  } finally {
    $("run").disabled = false;
  }
}
$("run").addEventListener("click", run);
document.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run(); });

function bar(name, p, best) {
  return `<div class="opt${best ? " best" : ""}"><span class="name" title="${esc(name)}">${esc(name)}</span><span class="pct">${(100 * p).toFixed(1)}%</span><div class="track"><div class="fill" style="width:${(100 * p).toFixed(1)}%"></div></div></div>`;
}

function renderResults(request, resp) {
  const u = resp.usage;
  const state = u.state_resident === undefined ? `state ${u.state_tokens}` : u.state_resident ? `state ${u.state_tokens} resident` : `state ${u.state_tokens} cold`;
  const chips = [`${u.ms.toFixed(0)} ms`, u.mode, `${u.forwards} forward${u.forwards === 1 ? "" : "s"}`, `${u.input_tokens} tokens (${state})`, `${u.questions} question${u.questions === 1 ? "" : "s"}`];
  $("usage").innerHTML = chips.map((c) => `<span class="chip">${esc(c)}</span>`).join("");
  const cards = [];
  for (const [id, a] of Object.entries(resp.answers)) {
    const q = request.questions[id];
    const mass = resp.diagnostics.candidate_mass[id];
    let opts = "", summary = "";
    if (a.type === "noul") {
      opts = bar("true", a.noul, a.noul >= 0.5) + bar("false", 1 - a.noul, a.noul < 0.5);
      summary = `<b>${a.noul >= 0.5 ? "true" : "false"}</b> · p ${a.noul.toFixed(3)}`;
    } else if (a.type === "choice") {
      opts = Object.entries(a.probabilities).map(([k, p]) => bar(k, p, k === a.choice)).join("");
      summary = `<b>${esc(a.choice)}</b> · conf ${a.confidence.toFixed(2)}`;
    } else {
      const best = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1])[0][0];
      opts = Object.entries(a.probabilities).map(([k, p]) => bar(`${k}  ${a.legend[k]}`, p, k === best)).join("");
      summary = `<b>${a.score.toFixed(2)}</b> · conf ${a.confidence.toFixed(2)}`;
    }
    cards.push(`<div class="card q">
      <div class="q-head"><span class="q-id">${esc(id)}</span><span class="badge">${a.type}</span></div>
      ${q.instructions ? `<div class="q-inst">${esc(q.instructions)}</div>` : ""}
      <div class="opts">${opts}</div>
      <div class="q-foot"><span>${summary}</span><span class="${mass < 0.9 ? "warn" : ""}" title="Probability mass on the candidate labels">mass ${mass.toFixed(3)}</span></div>
    </div>`);
  }
  $("results").innerHTML = cards.join("");
  $("raw").innerHTML = highlightJson(JSON.stringify(resp, null, 2));
}
