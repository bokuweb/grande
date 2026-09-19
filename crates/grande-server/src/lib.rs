//! `POST /v1/systemone` in TypeSafe's request / response shape, `GET
//! /v1/models`, `GET /health`, plus kev-style diagnostics endpoints.
//!
//! One engine, one request at a time: the packed pass already parallelizes
//! inside a request, and mixing tenants into one batch makes failures hard to
//! attribute. Scale by running more processes; GGUF weights are mmapped.

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use grande_core::{Backend, Engine, Error, Mode, Request};
use indexmap::IndexMap;
use serde_json::{json, Value};

pub const ALIASES: [&str; 3] = ["grande-latest", "jev-latest", "jev-preview"];

pub struct AppState<B: Backend> {
    pub engine: Mutex<Engine<B>>,
    pub api_key: Option<String>,
    pub model_id: String,
}

pub fn router<B: Backend + Send + 'static>(state: Arc<AppState<B>>) -> Router {
    Router::new()
        .route("/health", get(health::<B>))
        .route("/v1/models", get(models::<B>))
        .route("/v1/systemone", post(systemone::<B>))
        .route("/v1/systemone/separate", post(systemone_separate::<B>))
        .route("/v1/systemone/permute", post(permute::<B>))
        .with_state(state)
}

struct ApiError(StatusCode, Value);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(self.1)).into_response()
    }
}

fn map_err(e: Error) -> ApiError {
    match e {
        Error::Invalid { path, message } => ApiError(
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({"detail": [{"loc": ["body", path], "msg": message, "type": "value_error"}]}),
        ),
        Error::LabelNotSingleToken(l) => ApiError(
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({"detail": [{"loc": ["body", "questions"], "msg": format!("label {l:?} is not a single token"), "type": "value_error"}]}),
        ),
        Error::Backend(m) => ApiError(StatusCode::BAD_GATEWAY, json!({"detail": m})),
    }
}

fn authorize<B: Backend>(state: &AppState<B>, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(key) = &state.api_key else {
        return Ok(());
    };
    let ok = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|k| k == key);
    if ok {
        Ok(())
    } else {
        Err(ApiError(
            StatusCode::UNAUTHORIZED,
            json!({"detail": "invalid or missing API key"}),
        ))
    }
}

fn check_model<B: Backend>(state: &AppState<B>, req: &Request) -> Result<(), ApiError> {
    if ALIASES.contains(&req.model.as_str()) || req.model == state.model_id {
        Ok(())
    } else {
        Err(ApiError(
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({"detail": [{"loc": ["body", "model"], "msg": "unknown model; see GET /v1/models", "type": "value_error"}]}),
        ))
    }
}

async fn health<B: Backend + Send + 'static>(State(s): State<Arc<AppState<B>>>) -> Json<Value> {
    Json(json!({"status": "ok", "model": s.model_id}))
}

async fn models<B: Backend + Send + 'static>(
    State(s): State<Arc<AppState<B>>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    authorize(&s, &headers)?;
    let mut list: Vec<Value> = ALIASES
        .iter()
        .map(|a| json!({"name": a, "description": format!("alias of {}", s.model_id), "release_date": "2026-09-19"}))
        .collect();
    list.push(json!({"name": s.model_id, "description": "local grande model", "release_date": "2026-09-19"}));
    Ok(Json(json!({"models": list})))
}

fn diag_headers(diag: &grande_core::Diagnostics, model: &str, ms: u128) -> HeaderMap {
    let mut h = HeaderMap::new();
    let put = |h: &mut HeaderMap, k: &'static str, v: String| {
        if let Ok(val) = v.parse() {
            h.insert(k, val);
        }
    };
    put(&mut h, "x-grande-backend", model.to_string());
    put(
        &mut h,
        "x-grande-prefix-tokens",
        diag.prefix_tokens.to_string(),
    );
    put(
        &mut h,
        "x-grande-branch-tokens",
        diag.branch_tokens.iter().sum::<usize>().to_string(),
    );
    put(&mut h, "x-grande-passes", diag.passes.to_string());
    if let Some(src) = diag.prefix_source {
        put(&mut h, "x-grande-state", src.as_str().to_string());
    }
    put(&mut h, "x-grande-latency-ms", ms.to_string());
    if let Some(m) = diag.candidate_mass.values().cloned().reduce(f64::min) {
        put(&mut h, "x-grande-candidate-mass-min", format!("{m:.4}"));
    }
    h
}

async fn run<B: Backend + Send + 'static>(
    s: Arc<AppState<B>>,
    headers: HeaderMap,
    req: Request,
    mode: Mode,
) -> Result<Response, ApiError> {
    authorize(&s, &headers)?;
    check_model(&s, &req)?;
    let t = std::time::Instant::now();
    let (mut resp, diag) = {
        let mut engine = s.engine.lock().map_err(|_| {
            ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"detail": "engine poisoned"}),
            )
        })?;
        engine.answer(&req, mode).map_err(map_err)?
    };
    resp.model = if ALIASES.contains(&req.model.as_str()) {
        s.model_id.clone()
    } else {
        req.model.clone()
    };
    let h = diag_headers(&diag, &s.model_id, t.elapsed().as_millis());
    Ok((h, Json(resp)).into_response())
}

async fn systemone<B: Backend + Send + 'static>(
    State(s): State<Arc<AppState<B>>>,
    headers: HeaderMap,
    Json(req): Json<Request>,
) -> Result<Response, ApiError> {
    run(s, headers, req, Mode::Packed).await
}

async fn systemone_separate<B: Backend + Send + 'static>(
    State(s): State<Arc<AppState<B>>>,
    headers: HeaderMap,
    Json(req): Json<Request>,
) -> Result<Response, ApiError> {
    run(s, headers, req, Mode::Separate).await
}

/// Re-ask one Choice under several option orders; returns the distribution
/// mapped back to option keys for each order, for position-bias probes.
async fn permute<B: Backend + Send + 'static>(
    State(s): State<Arc<AppState<B>>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorize(&s, &headers)?;
    let req: Request = serde_json::from_value(body["request"].clone()).map_err(|e| {
        ApiError(
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({"detail": e.to_string()}),
        )
    })?;
    check_model(&s, &req)?;
    let qid = body["question"].as_str().unwrap_or_default().to_string();
    let n = body["orders"].as_u64().unwrap_or(4) as usize;
    let k = match req.questions.get(&qid) {
        Some(grande_core::Question::Choice { criteria, .. }) => criteria.len(),
        _ => {
            return Err(ApiError(
                StatusCode::UNPROCESSABLE_ENTITY,
                json!({"detail": "question must name a choice"}),
            ))
        }
    };
    let mut results = Vec::new();
    let mut engine = s.engine.lock().map_err(|_| {
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"detail": "engine poisoned"}),
        )
    })?;
    for r in 0..n {
        // Deterministic rotations / reversals; enough to expose position bias.
        let mut order: Vec<usize> = (0..k).collect();
        order.rotate_left(r % k);
        if r % 2 == 1 {
            order.reverse();
        }
        let mut orders = IndexMap::new();
        orders.insert(qid.clone(), order.clone());
        let (dists, _) = engine
            .distributions(&req, &orders, Mode::Packed)
            .map_err(map_err)?;
        let (branch, dist) = dists
            .iter()
            .find(|(b, _)| b.id == qid)
            .expect("question present");
        let probs: IndexMap<&str, f64> = branch
            .keys
            .iter()
            .map(String::as_str)
            .zip(dist.probs.iter().copied())
            .collect();
        results.push(json!({"order": order, "probabilities": probs}));
    }
    Ok(Json(json!({"question": qid, "results": results})))
}
