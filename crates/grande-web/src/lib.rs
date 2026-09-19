//! Browser surface of `grande-core`. The inference engine lives in JS
//! (transformers.js on WebGPU); this module owns everything that must be
//! identical to the native runtime: validation, the rendered layout, label
//! assignment, softmax / temperature / confidence, and the response shape.

use grande_core::calibration::CONTENT_FREE;
use grande_core::engine::to_answer;
use grande_core::readout::{Distribution, LABELS};
use grande_core::{Renderer, Request, Response, Usage};
use indexmap::IndexMap;
use serde::Deserialize;
use wasm_bindgen::prelude::*;

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

fn parse_rows(rows: &str, req: &Request, what: &str) -> Result<Vec<Row>, JsError> {
    let rows: Vec<Row> =
        serde_json::from_str(rows).map_err(|e| JsError::new(&format!("{what}: {e}")))?;
    if rows.len() != req.questions.len() {
        return Err(JsError::new(&format!(
            "{} {what} for {} questions",
            rows.len(),
            req.questions.len()
        )));
    }
    Ok(rows)
}

/// Assemble the TypeSafe-shaped response. `rows` is a JSON array with one
/// entry per branch (request order): the option logits the backend read at
/// the branch's answer position, plus an optional candidate-mass diagnostic.
/// `baseline_rows`, same shape, are the logits of the same branches over the
/// content-free state; when given, each answer is contextually calibrated
/// (its logits minus the baseline's) before the softmax.
#[wasm_bindgen]
pub fn answer(
    request: &str,
    rows: &str,
    temperature: f32,
    model: &str,
    input_tokens: u32,
    baseline_rows: Option<String>,
) -> Result<String, JsError> {
    let req: Request =
        serde_json::from_str(request).map_err(|e| JsError::new(&format!("request: {e}")))?;
    let rows = parse_rows(rows, &req, "rows")?;
    let baseline = match &baseline_rows {
        Some(b) => Some(parse_rows(b, &req, "baseline rows")?),
        None => None,
    };
    // Any layout gives the same keys/order for the default (unpermuted) render.
    let rendered = Renderer::gemma_pointer().render(&req);
    let mut answers = IndexMap::new();
    for (i, (branch, row)) in rendered.branches.iter().zip(rows).enumerate() {
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
        answers.insert(
            branch.id.clone(),
            to_answer(&req.questions[&branch.id], branch, &dist),
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
    serde_json::to_string(&resp).map_err(|e| JsError::new(&e.to_string()))
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

    /// Hidden width and vocabulary size, as JSON.
    pub fn config(&self) -> String {
        let c = &self.inner.config;
        format!("{{\"d\":{},\"vocab\":{},\"bos\":{}}}", c.d, c.vocab, c.bos)
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
            .evaluate(&prefix, &branches, want)
            .await
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        Ok(out
            .into_iter()
            .flat_map(|b| b.rows.into_iter().flatten())
            .collect())
    }
}
