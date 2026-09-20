//! Request → sequences → distributions → response, shared by the native
//! backend and the browser (which differ only in who tokenizes).

use anyhow::{anyhow, Result};
use grande_core::engine::to_answer;
use grande_core::readout::Distribution;
use grande_core::render::Kind;
use grande_core::{Diagnostics, Question, RenderedBranch, Request, Response, Usage};
use indexmap::IndexMap;

use super::prompt::{self, Tokenize};
use super::{LayaConfig, LayaEngine, Output, Sequence};

/// A question's option keys in grande's original order, with the Laya
/// option index of each: identical for Choice / Score, swapped for Noul
/// (grande lists `true` first, Laya `false`).
fn keys_of(q: &Question) -> (Kind, Vec<String>, Vec<usize>) {
    match q {
        Question::Choice { criteria, .. } => (
            Kind::Choice,
            criteria.keys().cloned().collect(),
            (0..criteria.len()).collect(),
        ),
        Question::Score { criteria, .. } => (
            Kind::Score,
            (0..criteria.len()).map(|i| i.to_string()).collect(),
            (0..criteria.len()).collect(),
        ),
        Question::Noul { .. } => (Kind::Noul, vec!["true".into(), "false".into()], vec![1, 0]),
    }
}

/// One sequence and one branch per question, options in `orders`' slot
/// order (grande original indices) where given; plus the state's token
/// count.
pub fn sequences(
    cfg: &LayaConfig,
    tok: &dyn Tokenize,
    req: &Request,
    orders: &IndexMap<String, Vec<usize>>,
) -> Result<Built> {
    req.validate().map_err(|e| anyhow!("{e}"))?;
    let state = prompt::state_ids(tok, cfg, &req.state);
    let mut seqs = Vec::with_capacity(req.questions.len());
    let mut branches = Vec::with_capacity(req.questions.len());
    for (id, q) in &req.questions {
        let (kind, keys, laya_of) = keys_of(q);
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
        let laya_order: Vec<usize> = order.iter().map(|&i| laya_of[i]).collect();
        let seq = prompt::build_sequence(tok, cfg, &state, q, Some(&laya_order))
            .map_err(|e| anyhow!("question {id:?}: {e}"))?;
        seqs.push(seq);
        branches.push(RenderedBranch {
            id: id.clone(),
            kind,
            keys: order.iter().map(|&i| keys[i].clone()).collect(),
            order,
            segments: Vec::new(),
            marks: Vec::new(),
        });
    }
    Ok((seqs, branches, state.len()))
}

/// Distributions from the engine's outputs under the checkpoint's
/// temperatures times `temperature`.
pub fn distributions(
    cfg: &LayaConfig,
    temperature: f32,
    seqs: &[Sequence],
    branches: Vec<RenderedBranch>,
    outs: Vec<Output>,
    state_tokens: usize,
) -> (Vec<(RenderedBranch, Distribution)>, Diagnostics) {
    let mut diag = Diagnostics {
        prefix_tokens: state_tokens,
        branch_tokens: seqs.iter().map(|s| s.ids.len()).collect(),
        passes: 1,
        orders: 1,
        batch: 1,
        ..Default::default()
    };
    let mut dists = Vec::with_capacity(outs.len());
    for ((seq, out), branch) in seqs.iter().zip(outs).zip(branches) {
        let t = cfg.temperature_for(seq.qtype, out.logits.len()) * temperature;
        let dist = Distribution::from_logits(out.logits, t.max(1e-3), None);
        diag.act_probability
            .insert(branch.id.clone(), f64::from(out.act_probability));
        dists.push((branch, dist));
    }
    (dists, diag)
}

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
            input_tokens: diag.branch_tokens.iter().sum::<usize>() as u64,
            output_tokens: 0,
        },
    }
}

/// One answered request: response, per-question distributions, diagnostics.
pub type Decided = (Response, Vec<(RenderedBranch, Distribution)>, Diagnostics);
/// A request's sequences, branches and state token count.
pub type Built = (Vec<Sequence>, Vec<RenderedBranch>, usize);

impl LayaEngine {
    /// Several requests in as few passes as the capacity allows: their
    /// sequences are packed together (each is its own sequence, so nothing
    /// changes for the mask) and split back afterwards. One result per
    /// request, in order; a request that fails on its own stays a failure
    /// without taking the others down.
    pub async fn decide_many(
        &self,
        tok: &dyn Tokenize,
        model: &str,
        temperature: f32,
        reqs: &[&Request],
    ) -> Vec<Result<Decided>> {
        let none = IndexMap::new();
        // Build every request's sequences first; keep the failures.
        let built: Vec<Result<Built>> = reqs
            .iter()
            .map(|r| sequences(&self.config, tok, r, &none))
            .collect();
        let mut results: Vec<Option<Result<Decided>>> = (0..reqs.len()).map(|_| None).collect();
        // Greedy passes over the requests that built, in order, within the
        // engine's token and row capacity.
        let mut passes: Vec<Vec<usize>> = Vec::new();
        let mut batch: Vec<usize> = Vec::new();
        let (mut tokens, mut rows) = (0usize, 0usize);
        for (i, b) in built.iter().enumerate() {
            let Ok((seqs, _, _)) = b else { continue };
            let t: usize = seqs.iter().map(|s| s.ids.len()).sum();
            let r: usize = seqs.iter().map(|s| s.markers.len() + 1).sum();
            if !batch.is_empty() && (tokens + t > self.capacity || rows + r > self.max_rows) {
                passes.push(std::mem::take(&mut batch));
                tokens = 0;
                rows = 0;
            }
            batch.push(i);
            tokens += t;
            rows += r;
        }
        if !batch.is_empty() {
            passes.push(batch);
        }
        for pass in passes {
            let mut all: Vec<Sequence> = Vec::new();
            for &i in &pass {
                if let Ok((seqs, _, _)) = &built[i] {
                    all.extend(seqs.iter().cloned());
                }
            }
            match self.evaluate(&all).await {
                Err(e) => {
                    let msg = format!("{e:#}");
                    for &i in &pass {
                        results[i] = Some(Err(anyhow!("{msg}")));
                    }
                }
                Ok(mut outs) => {
                    for &i in &pass {
                        let Ok((seqs, branches, state_tokens)) = &built[i] else {
                            continue;
                        };
                        let mine: Vec<Output> = outs.drain(..seqs.len()).collect();
                        let (dists, mut diag) = distributions(
                            &self.config,
                            temperature,
                            seqs,
                            branches.clone(),
                            mine,
                            *state_tokens,
                        );
                        diag.batch = pass.len();
                        let resp = response(model, reqs[i], &dists, &diag);
                        results[i] = Some(Ok((resp, dists, diag)));
                    }
                }
            }
        }
        results
            .into_iter()
            .zip(built)
            .map(|(r, b)| match (r, b) {
                (Some(r), _) => r,
                (None, Err(e)) => Err(e),
                (None, Ok(_)) => Err(anyhow!("request not evaluated")),
            })
            .collect()
    }

    /// Answer a whole request: tokenize with `tok`, one pass, distributions
    /// and the response.
    pub async fn decide(
        &self,
        tok: &dyn Tokenize,
        model: &str,
        temperature: f32,
        req: &Request,
        orders: &IndexMap<String, Vec<usize>>,
    ) -> Result<(Response, Vec<(RenderedBranch, Distribution)>, Diagnostics)> {
        let (seqs, branches, state_tokens) = sequences(&self.config, tok, req, orders)?;
        let outs = self.evaluate(&seqs).await?;
        let (dists, diag) = distributions(
            &self.config,
            temperature,
            &seqs,
            branches,
            outs,
            state_tokens,
        );
        let resp = response(model, req, &dists, &diag);
        Ok((resp, dists, diag))
    }
}
