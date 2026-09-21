//! Request → texts → vectors → distributions → response, shared by the
//! native backend and the browser (which differ only in who tokenizes).
//! The texts are exactly what `tools/e5_serve.py` renders, so the head
//! trained on `tools/e5_generic.py`'s features sees the same input here.

use anyhow::{anyhow, Result};
use grande_core::engine::to_answer;
use grande_core::readout::Distribution;
use grande_core::render::Kind;
use grande_core::{Diagnostics, Question, RenderedBranch, Request, Response, Usage};
use indexmap::IndexMap;
use serde_json::Value;

use super::{E5Engine, Tokenize};

/// Python's `json.dumps(v, ensure_ascii=False)` (what the shim and the
/// feature dump used for non-string values).
fn py_json(v: &Value) -> String {
    #[cfg(feature = "laya")]
    {
        crate::laya::prompt::py_json(v, false)
    }
    #[cfg(not(feature = "laya"))]
    {
        serde_json::to_string(v).unwrap_or_default()
    }
}

/// The state as one line: `key: value` pairs joined by spaces for an
/// object, the string itself, or JSON for anything else (without the
/// `query: ` prefix, which the engine adds to every sequence).
pub fn state_text(state: &Value) -> String {
    match state {
        Value::Object(m) => m
            .iter()
            .map(|(k, v)| match v {
                Value::String(s) => format!("{k}: {s}"),
                other => format!("{k}: {}", py_json(other)),
            })
            .collect::<Vec<_>>()
            .join(" "),
        Value::String(s) => s.clone(),
        other => py_json(other),
    }
}

fn option_text(key: &str, desc: Option<&Value>) -> String {
    match desc {
        None | Some(Value::Null) => key.to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => py_json(other),
    }
}

/// A question's option keys (grande's order) and each option's text.
fn options_of(q: &Question) -> (Kind, Vec<String>, Vec<String>) {
    match q {
        Question::Choice { criteria, .. } => (
            Kind::Choice,
            criteria.keys().cloned().collect(),
            criteria
                .iter()
                .map(|(k, v)| option_text(k, v.as_ref()))
                .collect(),
        ),
        Question::Score { criteria, .. } => (
            Kind::Score,
            (0..criteria.len()).map(|i| i.to_string()).collect(),
            criteria
                .iter()
                .enumerate()
                .map(|(i, c)| option_text(&i.to_string(), Some(c)))
                .collect(),
        ),
        Question::Noul { criteria, .. } => (
            Kind::Noul,
            vec!["true".into(), "false".into()],
            ["true", "false"]
                .iter()
                .map(|k| {
                    option_text(
                        k,
                        criteria
                            .as_ref()
                            .and_then(|m| m.get(*k))
                            .and_then(|v| v.as_ref()),
                    )
                })
                .collect(),
        ),
    }
}

/// One built request: the state text first, then every option text in
/// question order, with each question's branch (keys in `orders`' slot
/// order where given) and its span in the text list.
pub struct Built {
    pub texts: Vec<String>,
    pub branches: Vec<RenderedBranch>,
    /// Per question: first option index into `texts`, option count.
    pub spans: Vec<(usize, usize)>,
}

pub fn build(req: &Request, orders: &IndexMap<String, Vec<usize>>) -> Result<Built> {
    req.validate().map_err(|e| anyhow!("{e}"))?;
    let mut texts = vec![state_text(&req.state)];
    let mut branches = Vec::with_capacity(req.questions.len());
    let mut spans = Vec::with_capacity(req.questions.len());
    for (id, q) in &req.questions {
        let (kind, keys, opts) = options_of(q);
        let order: Vec<usize> = match orders.get(id) {
            Some(o) => {
                if o.len() != keys.len() || o.iter().any(|&i| i >= keys.len()) {
                    return Err(anyhow!(
                        "question {id:?}: order {o:?} for {} options",
                        keys.len()
                    ));
                }
                o.clone()
            }
            None => (0..keys.len()).collect(),
        };
        spans.push((texts.len(), keys.len()));
        texts.extend(order.iter().map(|&i| opts[i].clone()));
        branches.push(RenderedBranch {
            id: id.clone(),
            kind,
            keys: order.iter().map(|&i| keys[i].clone()).collect(),
            order,
            segments: Vec::new(),
            marks: Vec::new(),
        });
    }
    Ok(Built {
        texts,
        branches,
        spans,
    })
}

/// One answered request: response, per-question distributions, diagnostics.
pub type Decided = (Response, Vec<(RenderedBranch, Distribution)>, Diagnostics);

/// The TypeSafe-shaped response for a request's distributions.
pub fn response(
    model: &str,
    req: &Request,
    dists: &[(RenderedBranch, Distribution)],
    diag: &Diagnostics,
) -> Response {
    let mut answers = IndexMap::new();
    for (branch, dist) in dists {
        answers.insert(
            branch.id.clone(),
            to_answer(&req.questions[&branch.id], branch, dist),
        );
    }
    Response {
        model: model.to_string(),
        answers,
        usage: Usage {
            input_tokens: (diag.prefix_tokens + diag.branch_tokens.iter().sum::<usize>()) as u64,
            output_tokens: 0,
        },
    }
}

/// The (state, option) pairs of a built request: text 0 against every
/// option text.
fn pairs_of(built: &Built) -> Vec<(usize, usize)> {
    built
        .spans
        .iter()
        .flat_map(|&(start, k)| (start..start + k).map(|i| (0, i)))
        .collect()
}

impl E5Engine {
    /// Distributions for a built request whose pairs scored `logits` (one
    /// per option, in text order) with `tokens[i]` the token count of text
    /// i. `temperature` multiplies the head's own.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        model: &str,
        temperature: f32,
        req: &Request,
        built: Built,
        logits: &[f32],
        tokens: &[usize],
        batch: usize,
    ) -> Decided {
        let t = (self.config.temperature * temperature).max(1e-3);
        let mut dists = Vec::with_capacity(built.branches.len());
        let mut branch_tokens = Vec::with_capacity(built.branches.len());
        for (branch, &(start, k)) in built.branches.into_iter().zip(&built.spans) {
            // logits are in text order, offset by the state at index 0
            let z = logits[start - 1..start - 1 + k].to_vec();
            branch_tokens.push(tokens[start..start + k].iter().sum());
            dists.push((branch, Distribution::from_logits(z, t, None)));
        }
        let diag = Diagnostics {
            prefix_tokens: tokens[0],
            branch_tokens,
            passes: 1,
            orders: 1,
            batch,
            ..Default::default()
        };
        let resp = response(model, req, &dists, &diag);
        (resp, dists, diag)
    }

    /// Answer a whole request: the state and every option not cached go
    /// through the encoder, the head scores each pair, all in one pass.
    pub async fn decide(
        &self,
        tok: &dyn Tokenize,
        model: &str,
        temperature: f32,
        req: &Request,
        orders: &IndexMap<String, Vec<usize>>,
    ) -> Result<Decided> {
        let built = build(req, orders)?;
        let cacheable: Vec<bool> = (0..built.texts.len()).map(|i| i > 0).collect();
        let pairs = pairs_of(&built);
        let (logits, tokens) = self.evaluate(tok, &built.texts, &cacheable, &pairs).await?;
        Ok(self.finish(model, temperature, req, built, &logits, &tokens, 1))
    }

    /// Several requests in one pass (every text is its own sequence, so
    /// packing them changes nothing but the throughput). One result per
    /// request, in order; a request that fails to build stays a failure
    /// without taking the others down.
    pub async fn decide_many(
        &self,
        tok: &dyn Tokenize,
        model: &str,
        temperature: f32,
        reqs: &[&Request],
    ) -> Vec<Result<Decided>> {
        let none = IndexMap::new();
        let built: Vec<Result<Built>> = reqs.iter().map(|r| build(r, &none)).collect();
        let mut texts: Vec<String> = Vec::new();
        let mut cacheable: Vec<bool> = Vec::new();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        // Per request: first text index, first pair index.
        let mut starts: Vec<(usize, usize)> = Vec::new();
        for b in &built {
            starts.push((texts.len(), pairs.len()));
            if let Ok(b) = b {
                let base = texts.len();
                pairs.extend(pairs_of(b).into_iter().map(|(a, o)| (base + a, base + o)));
                cacheable.extend((0..b.texts.len()).map(|i| i > 0));
                texts.extend(b.texts.iter().cloned());
            }
        }
        let ok = built.iter().filter(|b| b.is_ok()).count();
        let (logits, tokens) = match self.evaluate(tok, &texts, &cacheable, &pairs).await {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("{e:#}");
                return built
                    .into_iter()
                    .map(|b| b.and_then(|_| Err(anyhow!("{msg}"))))
                    .collect();
            }
        };
        built
            .into_iter()
            .zip(starts)
            .zip(reqs)
            .map(|((b, (ts, ps)), req)| {
                b.map(|b| {
                    let nt = b.texts.len();
                    let np = nt - 1;
                    self.finish(
                        model,
                        temperature,
                        req,
                        b,
                        &logits[ps..ps + np],
                        &tokens[ts..ts + nt],
                        ok,
                    )
                })
            })
            .collect()
    }
}
