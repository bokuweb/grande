//! Browser surface of `grande-core`. The inference engine lives in JS
//! (transformers.js on WebGPU); this module owns everything that must be
//! identical to the native runtime: validation, the rendered layout, label
//! assignment, softmax / temperature / confidence, and the response shape.

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
    let req: Request = serde_json::from_str(request).map_err(|e| JsError::new(&format!("request: {e}")))?;
    req.validate().map_err(|e| JsError::new(&e.to_string()))?;
    let renderer: Renderer = serde_json::from_str(layout).map_err(|e| JsError::new(&format!("layout: {e}")))?;
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

/// Assemble the TypeSafe-shaped response. `rows` is a JSON array with one
/// entry per branch (request order): the option logits the backend read at
/// the branch's answer position, plus an optional candidate-mass diagnostic.
#[wasm_bindgen]
pub fn answer(request: &str, rows: &str, temperature: f32, model: &str, input_tokens: u32) -> Result<String, JsError> {
    let req: Request = serde_json::from_str(request).map_err(|e| JsError::new(&format!("request: {e}")))?;
    let rows: Vec<Row> = serde_json::from_str(rows).map_err(|e| JsError::new(&format!("rows: {e}")))?;
    if rows.len() != req.questions.len() {
        return Err(JsError::new(&format!("{} rows for {} questions", rows.len(), req.questions.len())));
    }
    // Any layout gives the same keys/order for the default (unpermuted) render.
    let rendered = Renderer::gemma_pointer().render(&req);
    let mut answers = IndexMap::new();
    for (branch, row) in rendered.branches.iter().zip(rows) {
        if row.logits.len() != branch.keys.len() {
            return Err(JsError::new(&format!("branch {}: {} logits for {} options", branch.id, row.logits.len(), branch.keys.len())));
        }
        let dist = Distribution::from_logits(row.logits, temperature, row.candidate_mass);
        answers.insert(branch.id.clone(), to_answer(&req.questions[&branch.id], branch, &dist));
    }
    let resp = Response { model: model.to_string(), answers, usage: Usage { input_tokens: u64::from(input_tokens), output_tokens: 0 } };
    serde_json::to_string(&resp).map_err(|e| JsError::new(&e.to_string()))
}
