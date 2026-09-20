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
) -> Result<(Vec<Sequence>, Vec<RenderedBranch>, usize)> {
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

impl LayaEngine {
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
