import { loadEngine, MODELS } from "./engine.js";
import { PRESETS } from "./presets.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
let engine = null;
let transformers = null;

for (const [k, m] of Object.entries(MODELS)) {
  const o = document.createElement("option");
  o.value = k; o.textContent = `${k}  (${m.size}, ${m.note})`;
  if (k === (params.get("model") ?? "gemma-3-1b")) o.selected = true;
  $("model").append(o);
}
for (const [k, p] of Object.entries(PRESETS)) {
  const o = document.createElement("option");
  o.value = k; o.textContent = p.label;
  $("preset").append(o);
}

function setPreset(name) {
  const p = PRESETS[name];
  $("state").value = JSON.stringify(p.state, null, 2);
  $("questions").value = JSON.stringify(p.questions, null, 2);
}
$("preset").addEventListener("change", (e) => setPreset(e.target.value));
setPreset("ticket");
$("temp").addEventListener("input", (e) => ($("tempv").textContent = Number(e.target.value).toFixed(1)));

function setStatus(text, pct) {
  $("status").textContent = text;
  $("bar").style.width = pct == null ? "0%" : `${Math.round(pct)}%`;
}

$("load").addEventListener("click", async () => {
  $("load").disabled = true;
  try {
    if (!navigator.gpu) throw new Error("このブラウザは WebGPU が使えません（Chrome / Edge / Safari 26+）");
    transformers ??= await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0");
    const model = $("model").value;
    const files = new Map();
    setStatus(`${MODELS[model].id} を取得中…`, 0);
    engine = await loadEngine({
      transformers, model,
      onProgress: (info) => {
        if (info.status === "progress") {
          files.set(info.file, [info.loaded ?? 0, info.total ?? 0]);
          let l = 0, t = 0;
          for (const [a, b] of files.values()) { l += a; t += b; }
          setStatus(`${info.file}  ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB`, t ? (100 * l) / t : 0);
        } else if (info.status === "ready") setStatus("初期化中…", 100);
      },
    });
    setStatus(`${engine.spec.id} 読み込み完了（WebGPU, ${engine.spec.dtype}）`, 100);
    $("run").disabled = false;
  } catch (e) {
    setStatus(`読み込み失敗: ${e.message}`, 0);
    $("load").disabled = false;
    console.error(e);
  }
});

function parseState(text) {
  try { return JSON.parse(text); } catch { return text; }
}

async function run() {
  if (!engine) return;
  $("run").disabled = true;
  $("results").innerHTML = "";
  $("usage").textContent = "実行中…";
  try {
    const request = { state: parseState($("state").value), questions: JSON.parse($("questions").value) };
    const resp = await engine.answer(request, { temperature: Number($("temp").value), mode: $("mode").value });
    renderResults(request, resp);
  } catch (e) {
    $("usage").innerHTML = `<span class="warn">${e.message}</span>`;
    console.error(e);
  } finally {
    $("run").disabled = false;
  }
}
$("run").addEventListener("click", run);
document.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run(); });

function bar(name, p, best) {
  return `<div class="opt${best ? " best" : ""}"><span class="name">${name}</span><div class="track"><div class="fill" style="width:${(100 * p).toFixed(1)}%"></div></div><span class="mono">${(100 * p).toFixed(1)}%</span></div>`;
}

function renderResults(request, resp) {
  const u = resp.usage;
  $("usage").textContent = `${u.ms.toFixed(0)} ms · ${u.mode} · ${u.forwards} forward · ${u.input_tokens} tokens（state ${u.state_tokens}）· ${u.questions} 問`;
  const rows = [];
  for (const [id, a] of Object.entries(resp.answers)) {
    const q = request.questions[id];
    const mass = resp.diagnostics.candidate_mass[id];
    let cell = "";
    if (a.type === "noul") {
      cell = bar("true", a.noul, a.noul >= 0.5) + bar("false", 1 - a.noul, a.noul < 0.5);
    } else if (a.type === "choice") {
      cell = Object.entries(a.probabilities).map(([k, p]) => bar(k, p, k === a.choice)).join("");
    } else {
      const best = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1])[0][0];
      cell = Object.entries(a.probabilities).map(([k, p]) => bar(`${k}: ${a.legend[k]}`, p, k === best)).join("");
    }
    const summary = a.type === "noul" ? `noul ${a.noul.toFixed(3)}` : a.type === "choice" ? `${a.choice} · conf ${a.confidence.toFixed(2)}` : `score ${a.score.toFixed(2)} · conf ${a.confidence.toFixed(2)}`;
    rows.push(`<tr><td><b>${id}</b><br><span class="mono">${a.type}</span><br><small>${q.instructions ?? ""}</small></td><td>${cell}</td><td class="mono">${summary}<br><small class="${mass < 0.9 ? "warn" : ""}">mass ${mass.toFixed(3)}</small></td></tr>`);
  }
  $("results").innerHTML = `<table><thead><tr><th>質問</th><th>確率</th><th>答え</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
  $("raw").textContent = JSON.stringify(resp, null, 2);
}
