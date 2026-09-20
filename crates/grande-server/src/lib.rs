//! `POST /v1/systemone` in TypeSafe's request / response shape, `GET
//! /v1/models`, `GET /health`, plus kev-style diagnostics endpoints.
//!
//! One engine on one worker thread. Requests that arrive while a pass is
//! running are queued, and when the engine is free everything queued goes
//! to it together: the backend puts every request's state and branches in
//! one pass ([`Engine::answer_many`]), so throughput under concurrency is
//! set by the GPU's prefill rate, not by the request rate. Requests stay
//! independent: each gets its own response or its own error, and the
//! `X-Grande-Batch` header says how many shared its pass. Scale further by
//! running more processes; GGUF weights are mmapped.

use std::sync::mpsc;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use grande_core::readout::Distribution;
use grande_core::{Backend, Diagnostics, Engine, Error, Mode, RenderedBranch, Request};
use indexmap::IndexMap;
use serde_json::{json, Value};
use tokio::sync::oneshot;

pub const ALIASES: [&str; 3] = ["grande-latest", "jev-latest", "jev-preview"];

type Answered = grande_core::Result<(grande_core::Response, Diagnostics)>;
type Distributed = grande_core::Result<(Vec<(RenderedBranch, Distribution)>, Diagnostics)>;

/// One queued call into the engine.
enum Work {
    Answer {
        req: Request,
        mode: Mode,
        reply: oneshot::Sender<Answered>,
    },
    Distributions {
        req: Request,
        orders: IndexMap<String, Vec<usize>>,
        reply: oneshot::Sender<Distributed>,
    },
}

/// The engine's worker thread: queue in, replies out.
#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<Work>,
}

impl EngineHandle {
    /// Move `engine` to a thread of its own and hand back its queue. Up to
    /// `max_batch` queued `/v1/systemone` requests share one
    /// [`Engine::answer_many`] call; the engine still splits them by the
    /// backend's limits.
    pub fn spawn<B: Backend + Send + 'static>(mut engine: Engine<B>, max_batch: usize) -> Self {
        let (tx, rx) = mpsc::channel::<Work>();
        std::thread::Builder::new()
            .name("grande-engine".into())
            .spawn(move || {
                let max_batch = max_batch.max(1);
                while let Ok(first) = rx.recv() {
                    // Everything that queued up while the last pass ran.
                    let mut queued = vec![first];
                    while queued.len() < max_batch {
                        match rx.try_recv() {
                            Ok(w) => queued.push(w),
                            Err(_) => break,
                        }
                    }
                    let mut packed: Vec<(Request, oneshot::Sender<Answered>)> = Vec::new();
                    for w in queued {
                        match w {
                            Work::Answer {
                                req,
                                mode: Mode::Packed,
                                reply,
                            } => packed.push((req, reply)),
                            Work::Answer { req, mode, reply } => {
                                reply.send(engine.answer(&req, mode)).ok();
                            }
                            Work::Distributions { req, orders, reply } => {
                                reply
                                    .send(engine.distributions(&req, &orders, Mode::Packed))
                                    .ok();
                            }
                        }
                    }
                    if !packed.is_empty() {
                        let reqs: Vec<&Request> = packed.iter().map(|(r, _)| r).collect();
                        let results = engine.answer_many(&reqs, Mode::Packed);
                        for ((_, reply), r) in packed.into_iter().zip(results) {
                            reply.send(r).ok();
                        }
                    }
                }
            })
            .expect("spawn engine thread");
        EngineHandle { tx }
    }

    async fn answer(&self, req: Request, mode: Mode) -> Result<Answered, ApiError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Work::Answer { req, mode, reply })
            .map_err(|_| engine_gone())?;
        rx.await.map_err(|_| engine_gone())
    }

    async fn distributions(
        &self,
        req: Request,
        orders: IndexMap<String, Vec<usize>>,
    ) -> Result<Distributed, ApiError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Work::Distributions { req, orders, reply })
            .map_err(|_| engine_gone())?;
        rx.await.map_err(|_| engine_gone())
    }
}

fn engine_gone() -> ApiError {
    ApiError(
        StatusCode::INTERNAL_SERVER_ERROR,
        json!({"detail": "engine thread gone"}),
    )
}

pub struct AppState {
    pub engine: EngineHandle,
    pub api_key: Option<String>,
    pub model_id: String,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(systemone))
        .route("/v1/systemone/separate", post(systemone_separate))
        .route("/v1/systemone/permute", post(permute))
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

fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
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

fn check_model(state: &AppState, req: &Request) -> Result<(), ApiError> {
    if ALIASES.contains(&req.model.as_str()) || req.model == state.model_id {
        Ok(())
    } else {
        Err(ApiError(
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({"detail": [{"loc": ["body", "model"], "msg": "unknown model; see GET /v1/models", "type": "value_error"}]}),
        ))
    }
}

async fn health(State(s): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"status": "ok", "model": s.model_id}))
}

async fn models(
    State(s): State<Arc<AppState>>,
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
    if !diag.baseline.is_empty() {
        put(&mut h, "x-grande-calibrated", "contextual".to_string());
    }
    if let Some(src) = diag.prefix_source {
        put(&mut h, "x-grande-state", src.as_str().to_string());
    }
    if diag.orders > 1 {
        put(&mut h, "x-grande-orders", diag.orders.to_string());
        if let Some(s) = diag.order_spread.values().cloned().reduce(f64::max) {
            put(&mut h, "x-grande-order-spread-max", format!("{s:.4}"));
        }
    }
    if !diag.two_stage.is_empty() {
        put(
            &mut h,
            "x-grande-two-stage",
            diag.two_stage.keys().cloned().collect::<Vec<_>>().join(","),
        );
    }
    put(&mut h, "x-grande-batch", diag.batch.max(1).to_string());
    if !diag.rechecked.is_empty() {
        put(&mut h, "x-grande-rechecked", diag.rechecked.join(","));
    }
    put(&mut h, "x-grande-latency-ms", ms.to_string());
    if let Some(m) = diag.candidate_mass.values().cloned().reduce(f64::min) {
        put(&mut h, "x-grande-candidate-mass-min", format!("{m:.4}"));
    }
    h
}

async fn run(
    s: Arc<AppState>,
    headers: HeaderMap,
    req: Request,
    mode: Mode,
) -> Result<Response, ApiError> {
    authorize(&s, &headers)?;
    check_model(&s, &req)?;
    let t = std::time::Instant::now();
    let alias = ALIASES.contains(&req.model.as_str());
    let model = req.model.clone();
    let (mut resp, diag) = s.engine.answer(req, mode).await?.map_err(map_err)?;
    resp.model = if alias { s.model_id.clone() } else { model };
    let h = diag_headers(&diag, &s.model_id, t.elapsed().as_millis());
    Ok((h, Json(resp)).into_response())
}

async fn systemone(
    State(s): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<Request>,
) -> Result<Response, ApiError> {
    run(s, headers, req, Mode::Packed).await
}

async fn systemone_separate(
    State(s): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<Request>,
) -> Result<Response, ApiError> {
    run(s, headers, req, Mode::Separate).await
}

/// Re-ask one Choice under several option orders; returns the distribution
/// mapped back to option keys for each order, for position-bias probes.
async fn permute(
    State(s): State<Arc<AppState>>,
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
    // Deterministic rotations, then reversals; enough to expose position bias.
    for order in grande_core::math::option_orders(k, n) {
        let mut orders = IndexMap::new();
        orders.insert(qid.clone(), order.clone());
        let (dists, _) = s
            .engine
            .distributions(req.clone(), orders)
            .await?
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
