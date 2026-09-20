//! Browser surface of `grande-core`. The inference engine lives in JS
//! (the wgpu engine below, or transformers.js on WebGPU); this module owns
//! everything that must be identical to the native runtime: validation,
//! the rendered layout and the branch plan (option orders, two-stage
//! Choice), label assignment, softmax / temperature / confidence, and the
//! response shape.

use grande_core::calibration::CONTENT_FREE;
use grande_core::engine::to_answer;
use grande_core::readout::{Distribution, LABELS};
use grande_core::{Plan, Renderer, Request, Response, Usage};
use indexmap::IndexMap;
use serde::Deserialize;
use wasm_bindgen::prelude::*;
#[cfg(feature = "wgpu")]
use wasm_bindgen::JsCast;

/// Render a request. `layout` is a JSON `Renderer`, e.g.
/// `{"layout":"label","turn_start":"<|turn>","turn_end":"<turn|>","user":"user","model":"model"}`
/// or `{"layout":"pointer", ...delimiters}`. Returns the `Rendered` JSON:
/// prefix segments and, per branch, segments / marks / option keys.
#[wasm_bindgen]
pub fn render(request: &str, layout: &str) -> Result<String, JsError> {
    let req: Request =
        serde_json::from_str(request).map_err(|e| JsError::new(&format!("request: {e}")))?;
    req.validate().map_err(|e| JsError::new(&e.to_string()))?;
    let renderer: Renderer =
        serde_json::from_str(layout).map_err(|e| JsError::new(&format!("layout: {e}")))?;
    serde_json::to_string(&renderer.render(&req)).map_err(|e| JsError::new(&e.to_string()))
}

/// Option labels for the label readout, in order (`A`..`Z`, `a`..`z`).
#[wasm_bindgen]
pub fn labels() -> String {
    LABELS.iter().collect()
}

#[derive(Deserialize)]
struct Row {
    logits: Vec<f32>,
    #[serde(default)]
    candidate_mass: Option<f64>,
}

/// The content-free state for contextual calibration (`"N/A"`).
#[wasm_bindgen]
pub fn content_free_state() -> String {
    CONTENT_FREE.to_string()
}

fn js<T, E: std::fmt::Display>(r: Result<T, E>, what: &str) -> Result<T, JsError> {
    r.map_err(|e| JsError::new(&format!("{what}: {e}")))
}

fn plan_of(
    request: &str,
    layout: &str,
    orders: u32,
    label_cap: u32,
) -> Result<(Request, Renderer, Plan), JsError> {
    let req: Request = js(serde_json::from_str(request), "request")?;
    let renderer: Renderer = js(serde_json::from_str(layout), "layout")?;
    let cap = (label_cap > 0).then_some(label_cap as usize);
    let plan = js(
        Plan::new(
            &renderer,
            &req,
            &IndexMap::new(),
            cap,
            orders as usize,
            None,
        ),
        "plan",
    )?;
    Ok((req, renderer, plan))
}

/// One distribution per branch from the rows a backend read for them
/// (`[{"logits": [...], "candidate_mass": m}]`, one per branch, option
/// logits in the branch's slot order), contextually calibrated against
/// `baseline` (same shape, the same branches over the content-free state)
/// when given.
fn distributions(
    branches: &[grande_core::RenderedBranch],
    rows: &str,
    baseline: Option<&str>,
    temperature: f32,
    what: &str,
) -> Result<Vec<Distribution>, JsError> {
    let rows: Vec<Row> = js(serde_json::from_str(rows), what)?;
    if rows.len() != branches.len() {
        return Err(JsError::new(&format!(
            "{} {what} for {} branches",
            rows.len(),
            branches.len()
        )));
    }
    let baseline: Option<Vec<Row>> = match baseline {
        Some(b) => {
            let b: Vec<Row> = js(serde_json::from_str(b), "baseline rows")?;
            if b.len() != branches.len() {
                return Err(JsError::new(&format!(
                    "{} baseline rows for {} branches",
                    b.len(),
                    branches.len()
                )));
            }
            Some(b)
        }
        None => None,
    };
    let mut out = Vec::with_capacity(rows.len());
    for (i, (branch, row)) in branches.iter().zip(rows).enumerate() {
        if row.logits.len() != branch.keys.len() {
            return Err(JsError::new(&format!(
                "branch {}: {} logits for {} options",
                branch.id,
                row.logits.len(),
                branch.keys.len()
            )));
        }
        let mut dist = Distribution::from_logits(row.logits, temperature, row.candidate_mass);
        if let Some(b) = &baseline {
            if b[i].logits.len() != dist.logits.len() {
                return Err(JsError::new(&format!(
                    "branch {}: {} baseline logits for {} options",
                    branch.id,
                    b[i].logits.len(),
                    dist.logits.len()
                )));
            }
            dist.calibrate(b[i].logits.clone(), temperature);
        }
        out.push(dist);
    }
    Ok(out)
}

/// The first pass of a request: `{"prefix": [segments], "branches":
/// [RenderedBranch]}`. With `orders` > 1 every Choice / Noul appears under
/// that many option orders; a Choice with more options than `label_cap`
/// (0 = no cap, pointer readout) appears as groups whose finalists
/// `plan_second` asks for. Same code as the native engine.
#[wasm_bindgen]
pub fn plan(request: &str, layout: &str, orders: u32, label_cap: u32) -> Result<String, JsError> {
    let (_, _, plan) = plan_of(request, layout, orders, label_cap)?;
    js(
        serde_json::to_string(&serde_json::json!({
            "prefix": plan.rendered.prefix,
            "branches": plan.first,
        })),
        "plan",
    )
}

/// The second pass, given the first pass's rows: `{"branches": [...],
/// "finalists": {qid: [keys]}}`; `branches` is empty when no Choice was
/// grouped. `rows` / `baseline_rows` are the first pass's, as for `answer`.
#[wasm_bindgen]
pub fn plan_second(
    request: &str,
    layout: &str,
    orders: u32,
    label_cap: u32,
    rows: &str,
    temperature: f32,
    baseline_rows: Option<String>,
) -> Result<String, JsError> {
    let (req, renderer, plan) = plan_of(request, layout, orders, label_cap)?;
    let first = distributions(
        &plan.first,
        rows,
        baseline_rows.as_deref(),
        temperature,
        "rows",
    )?;
    let (branches, finalists, _rechecked) = plan.second(&renderer, &req, &first);
    js(
        serde_json::to_string(&serde_json::json!({
            "branches": branches,
            "finalists": finalists,
        })),
        "plan",
    )
}

/// Assemble the TypeSafe-shaped response from the rows of both passes:
/// `{"response": Response, "diagnostics": {"candidate_mass": {qid: m},
/// "order_spread": {qid: max |Δp| between orders}, "two_stage": {qid:
/// [finalist keys]}}}`. `rows` has one entry per `plan` branch, `rows2` one
/// per `plan_second` branch (omit when it returned none); the
/// `baseline_rows*` are the same branches over the content-free state and
/// turn on contextual calibration.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn answer(
    request: &str,
    layout: &str,
    orders: u32,
    label_cap: u32,
    rows: &str,
    rows2: Option<String>,
    temperature: f32,
    model: &str,
    input_tokens: u32,
    baseline_rows: Option<String>,
    baseline_rows2: Option<String>,
) -> Result<String, JsError> {
    let (req, renderer, plan) = plan_of(request, layout, orders, label_cap)?;
    let first = distributions(
        &plan.first,
        rows,
        baseline_rows.as_deref(),
        temperature,
        "rows",
    )?;
    let (branches, finalists, _rechecked) = plan.second(&renderer, &req, &first);
    let second = match (&rows2, branches.is_empty()) {
        (_, true) => Vec::new(),
        (Some(r), false) => distributions(
            &branches,
            r,
            baseline_rows2.as_deref(),
            temperature,
            "rows2",
        )?,
        (None, false) => {
            return Err(JsError::new(&format!(
                "{} second-pass branches need rows2",
                branches.len()
            )))
        }
    };
    let folded = plan.fold(
        first,
        branches.into_iter().zip(second).collect(),
        temperature,
    );
    let mut answers = IndexMap::new();
    let mut candidate_mass = IndexMap::new();
    for (branch, dist) in &folded.results {
        candidate_mass.insert(branch.id.clone(), dist.candidate_mass);
        answers.insert(
            branch.id.clone(),
            to_answer(&req.questions[&branch.id], branch, dist),
        );
    }
    let resp = Response {
        model: model.to_string(),
        answers,
        usage: Usage {
            input_tokens: u64::from(input_tokens),
            output_tokens: 0,
        },
    };
    js(
        serde_json::to_string(&serde_json::json!({
            "response": resp,
            "diagnostics": {
                "candidate_mass": candidate_mass,
                "order_spread": folded.order_spread,
                "two_stage": finalists,
            },
        })),
        "response",
    )
}

/// The wgpu engine on WebGPU: the whole request (state prefix + isolated
/// branches) is one forward pass, no KV-cache continuation, no padding.
#[cfg(feature = "wgpu")]
#[wasm_bindgen]
pub struct WgpuEngine {
    inner: grande_wgpu::Engine,
}

#[cfg(feature = "wgpu")]
#[derive(Deserialize)]
struct BranchIn {
    tokens: Vec<u32>,
    want: Vec<usize>,
}

#[cfg(feature = "wgpu")]
#[wasm_bindgen]
impl WgpuEngine {
    /// `config` is the checkpoint's config.json, `weights` its
    /// model.safetensors (f16 or f32). `capacity` is the packed-token budget
    /// of the workspace, `max_rows` how many rows a request may read back.
    pub async fn load(
        config: &str,
        weights: &[u8],
        capacity: u32,
        max_rows: u32,
    ) -> Result<WgpuEngine, JsError> {
        let w = grande_wgpu::Weights::load(config.as_bytes(), weights)
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        let inner = grande_wgpu::Engine::new(&w, capacity as usize, max_rows as usize)
            .await
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        Ok(WgpuEngine { inner })
    }

    /// Hidden width, vocabulary size, BOS id, per-layer embedding width and
    /// layer count, as JSON.
    pub fn config(&self) -> String {
        let c = &self.inner.config;
        format!(
            "{{\"d\":{},\"vocab\":{},\"bos\":{},\"per_layer_dim\":{},\"layers\":{}}}",
            c.d, c.vocab, c.bos, c.per_layer_dim, c.layers
        )
    }

    /// How the last `evaluate` obtained its prefix: `"resident"` (same
    /// prefix as the previous request, only the branches ran) or `"decoded"`.
    pub fn prefix_source(&self) -> Option<String> {
        self.inner.prefix_source().map(|s| s.as_str().to_string())
    }

    /// Forget the resident prefix (the next request decodes it again).
    pub fn evict_resident(&self) {
        self.inner.evict_resident();
    }

    /// Keep the K/V of every state seen (f16, sliding layers window-only)
    /// in a RAM LRU of `bytes`, so coming back to a state is a restore
    /// (`prefix_source` "ram") instead of a decode. 0 turns it off.
    pub fn set_state_cache_bytes(&mut self, bytes: u32) {
        self.inner.set_state_cache(bytes as usize, None, "wgpu");
    }

    /// Run a two-token request so the first real one does not pay for the
    /// workspace's first touch.
    pub async fn warmup(&self) -> Result<(), JsError> {
        self.inner
            .warmup()
            .await
            .map_err(|e| JsError::new(&format!("{e:#}")))
    }

    /// Run one packed pass. `branches` is JSON `[{"tokens":[...],"want":[...]}]`
    /// (want = branch-relative positions to read); `want` is `"hidden"` or
    /// `"logits"`. Returns the rows concatenated, branch by branch, each of
    /// width `d` (hidden) or `vocab` (logits).
    pub async fn evaluate(
        &self,
        prefix: Vec<u32>,
        branches: &str,
        want: &str,
    ) -> Result<Vec<f32>, JsError> {
        self.evaluate_rows(prefix, branches, want, None).await
    }

    /// `evaluate` for models with per-layer embeddings (Gemma 4): the caller
    /// gathers the per-layer token table rows for prefix + branch tokens,
    /// f16 little-endian `[tokens][layers x P]`.
    pub async fn evaluate_rows(
        &self,
        prefix: Vec<u32>,
        branches: &str,
        want: &str,
        per_layer_rows: Option<Vec<u8>>,
    ) -> Result<Vec<f32>, JsError> {
        let branches: Vec<BranchIn> =
            serde_json::from_str(branches).map_err(|e| JsError::new(&format!("branches: {e}")))?;
        let branches: Vec<grande_core::BranchTokens> = branches
            .into_iter()
            .map(|b| grande_core::BranchTokens {
                tokens: b
                    .tokens
                    .into_iter()
                    .map(|t| grande_core::Token(t as i32))
                    .collect(),
                want: b.want,
            })
            .collect();
        let want = match want {
            "logits" => grande_core::Want::Logits,
            _ => grande_core::Want::Hidden,
        };
        let out = self
            .inner
            .evaluate_rows(&prefix, &branches, want, per_layer_rows.as_deref())
            .await
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        Ok(out
            .into_iter()
            .flat_map(|b| b.rows.into_iter().flatten())
            .collect())
    }
}

/// Streams an exported model directory (tools/export_wgpu_gguf.py) into the
/// wgpu engine one tensor at a time, so a multi-GB checkpoint never has to
/// sit in wasm memory whole.
#[cfg(feature = "wgpu")]
#[wasm_bindgen]
pub struct WgpuLoader {
    inner: Option<grande_wgpu::EngineBuilder>,
}

#[cfg(feature = "wgpu")]
#[wasm_bindgen]
impl WgpuLoader {
    /// `config` is the directory's config.json.
    pub async fn open(config: &str) -> Result<WgpuLoader, JsError> {
        let v: serde_json::Value =
            serde_json::from_str(config).map_err(|e| JsError::new(&format!("config: {e}")))?;
        let cfg =
            grande_wgpu::Config::from_json(&v).map_err(|e| JsError::new(&format!("{e:#}")))?;
        let inner = grande_wgpu::EngineBuilder::new(cfg)
            .await
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        Ok(WgpuLoader { inner: Some(inner) })
    }

    /// Upload one tensor: `dtype` is "f16" / "q8" / "q4", `shape` its
    /// dimensions, `data` the payload and `scales` the f16 block scales
    /// (empty for f16), both as the manifest lays them out.
    pub fn push(
        &mut self,
        name: &str,
        dtype: &str,
        shape: Vec<u32>,
        data: Vec<u8>,
        scales: Vec<u8>,
    ) -> Result<(), JsError> {
        let b = self
            .inner
            .as_mut()
            .ok_or_else(|| JsError::new("loader already finished"))?;
        let dtype =
            grande_wgpu::Dtype::parse(dtype).map_err(|e| JsError::new(&format!("{e:#}")))?;
        let t = grande_wgpu::QTensor::from_raw(
            dtype,
            shape.into_iter().map(|x| x as usize).collect(),
            data,
            scales,
        )
        .map_err(|e| JsError::new(&format!("{name}: {e:#}")))?;
        b.push(name, &t)
            .map_err(|e| JsError::new(&format!("{name}: {e:#}")))
    }

    /// Names still to push.
    pub fn missing(&self) -> Vec<String> {
        self.inner.as_ref().map(|b| b.missing()).unwrap_or_default()
    }

    /// Allocate the workspace and wire the engine. `capacity` is the
    /// packed-token budget, `max_rows` how many rows a request may read back.
    pub fn finish(&mut self, capacity: u32, max_rows: u32) -> Result<WgpuEngine, JsError> {
        let b = self
            .inner
            .take()
            .ok_or_else(|| JsError::new("loader already finished"))?;
        let inner = b
            .finish(capacity as usize, max_rows as usize)
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        Ok(WgpuEngine { inner })
    }
}

/// Laya (ModernBERT / mmBERT encoder + decision head) on WebGPU: every
/// question of a request is one bidirectional sequence, all packed into one
/// pass; the scorer reads each option's `[MASK]` row. Tokenization is the
/// page's (transformers.js), handed in as a JS function.
#[cfg(feature = "wgpu")]
#[wasm_bindgen]
pub struct LayaEngine {
    inner: grande_wgpu::laya::LayaEngine,
    name: String,
}

#[cfg(feature = "wgpu")]
struct JsTokenize<'a>(&'a js_sys::Function);

#[cfg(feature = "wgpu")]
impl grande_wgpu::laya::prompt::Tokenize for JsTokenize<'_> {
    fn encode(&self, text: &str) -> Vec<u32> {
        let out = self
            .0
            .call1(&JsValue::NULL, &JsValue::from_str(text))
            .unwrap_or(JsValue::NULL);
        if let Some(a) = out.dyn_ref::<js_sys::Uint32Array>() {
            return a.to_vec();
        }
        js_sys::Array::from(&out)
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as u32)
            .collect()
    }
}

#[cfg(feature = "wgpu")]
#[wasm_bindgen]
impl LayaEngine {
    /// The checkpoint's `config.json` (tools/export_laya.py) as JSON and
    /// the model name written into responses.
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Hidden width, vocabulary, prompt budgets and special ids, as JSON.
    pub fn config(&self) -> String {
        let c = &self.inner.config;
        serde_json::json!({
            "d": c.d, "vocab": c.vocab, "layers": c.layers, "head_layers": c.head_layers,
            "max_len": c.max_len, "head_max_len": c.head_max_len,
            "cls": c.cls, "sep": c.sep, "mask": c.mask, "pad": c.pad,
        })
        .to_string()
    }

    pub async fn warmup(&self) -> Result<(), JsError> {
        self.inner
            .warmup()
            .await
            .map_err(|e| JsError::new(&format!("{e:#}")))
    }

    /// Answer a request (JSON). `tokenize` maps text to token ids without
    /// special tokens (a `Uint32Array` or array). `temperature` multiplies
    /// the checkpoint's own calibration temperatures. Returns
    /// `{"response": Response, "diagnostics": {"act_probability": {qid: p},
    /// "branch_tokens": [...], "state_tokens": n}}`.
    pub async fn answer(
        &self,
        request: &str,
        tokenize: &js_sys::Function,
        temperature: f32,
    ) -> Result<String, JsError> {
        let req: Request = js(serde_json::from_str(request), "request")?;
        let tok = JsTokenize(tokenize);
        let (resp, _, diag) = self
            .inner
            .decide(&tok, &self.name, temperature, &req, &IndexMap::new())
            .await
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        js(
            serde_json::to_string(&serde_json::json!({
                "response": resp,
                "diagnostics": {
                    "act_probability": diag.act_probability,
                    "branch_tokens": diag.branch_tokens,
                    "state_tokens": diag.prefix_tokens,
                },
            })),
            "response",
        )
    }
}

/// Streams an exported Laya directory (tools/export_laya.py) into the
/// engine one tensor at a time.
#[cfg(feature = "wgpu")]
#[wasm_bindgen]
pub struct LayaLoader {
    inner: Option<grande_wgpu::laya::LayaBuilder>,
    name: String,
}

#[cfg(feature = "wgpu")]
#[wasm_bindgen]
impl LayaLoader {
    /// `config` is the directory's config.json; `specials` maps the special
    /// tokens' surface forms (`<bos>`, `<mask>`, …) to their ids in the
    /// page's tokenizer, as JSON.
    pub async fn open(config: &str, specials: &str) -> Result<LayaLoader, JsError> {
        let v: serde_json::Value = js(serde_json::from_str(config), "config")?;
        let ids: std::collections::HashMap<String, u32> =
            js(serde_json::from_str(specials), "specials")?;
        let cfg = grande_wgpu::laya::config_from_export(&v, |s| ids.get(s).copied())
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        let name = v["laya_name"].as_str().unwrap_or("laya").to_string();
        let inner = grande_wgpu::laya::LayaBuilder::new(cfg)
            .await
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        Ok(LayaLoader {
            inner: Some(inner),
            name,
        })
    }

    pub fn push(
        &mut self,
        name: &str,
        dtype: &str,
        shape: Vec<u32>,
        data: Vec<u8>,
        scales: Vec<u8>,
    ) -> Result<(), JsError> {
        let b = self
            .inner
            .as_mut()
            .ok_or_else(|| JsError::new("loader already finished"))?;
        let dtype =
            grande_wgpu::Dtype::parse(dtype).map_err(|e| JsError::new(&format!("{e:#}")))?;
        let t = grande_wgpu::QTensor::from_raw(
            dtype,
            shape.into_iter().map(|x| x as usize).collect(),
            data,
            scales,
        )
        .map_err(|e| JsError::new(&format!("{name}: {e:#}")))?;
        b.push(name, &t)
            .map_err(|e| JsError::new(&format!("{name}: {e:#}")))
    }

    pub fn missing(&self) -> Vec<String> {
        self.inner.as_ref().map(|b| b.missing()).unwrap_or_default()
    }

    /// `capacity` is the packed-token budget per pass, `max_rows` the rows
    /// read back (one per option plus one per question).
    pub fn finish(&mut self, capacity: u32, max_rows: u32) -> Result<LayaEngine, JsError> {
        let b = self
            .inner
            .take()
            .ok_or_else(|| JsError::new("loader already finished"))?;
        let inner = b
            .finish(capacity as usize, max_rows as usize)
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        Ok(LayaEngine {
            inner,
            name: self.name.clone(),
        })
    }
}
