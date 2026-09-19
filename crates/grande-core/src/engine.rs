//! Glue: render → tokenize/pack → evaluate → read out → typed answers.

use std::collections::HashMap;

use indexmap::IndexMap;
use serde::Serialize;

use crate::api::{Answer, Question, Request, Response, Usage};
use crate::backend::{Backend, BranchTokens, PrefixSource, Token, Want};
use crate::math::{argmax, confidence, expected_index};
use crate::readout::{Distribution, Readout};
use crate::render::{Kind, Mark, Rendered, RenderedBranch, Renderer, Segment};
use crate::Result;

/// Per-request diagnostics, surfaced as headers by a server or printed by the CLI.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Diagnostics {
    pub prefix_tokens: usize,
    pub branch_tokens: Vec<usize>,
    /// Label readout only.
    pub candidate_mass: IndexMap<String, f64>,
    /// Number of `evaluate` calls (1 when packed; +1 for a baseline pass
    /// that was not served from the cache).
    pub passes: usize,
    /// Where the backend got the state from (resident / ram / disk / decoded).
    pub prefix_source: Option<PrefixSource>,
    /// Contextual calibration only: per question, the option logits over
    /// the content-free state that were subtracted.
    pub baseline: IndexMap<String, Vec<f32>>,
}

/// How branches are evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Every branch in one `evaluate` call: state once, one pass.
    Packed,
    /// One `evaluate` call per branch, re-evaluating the prefix each time.
    /// Only for checking that packing does not change the numbers.
    Separate,
}

pub struct Engine<B: Backend> {
    pub backend: B,
    pub renderer: Renderer,
    pub readout: Readout,
    pub temperature: f32,
    pub model: String,
    /// Contextual calibration: when set, every question is also asked over
    /// this content-free state and the option logits it yields are
    /// subtracted from the live ones (see [`crate::calibration::contextual`]).
    /// The baseline depends on the question alone, so it is cached per
    /// rendered branch: a fixed question set over changing states pays for
    /// it once.
    pub baseline: Option<String>,
    baseline_cache: HashMap<String, Vec<f32>>,
}

/// A tokenized request ready for the backend.
#[derive(Debug, Clone)]
pub struct Packed {
    pub prefix: Vec<Token>,
    pub branches: Vec<BranchTokens>,
}

impl<B: Backend> Engine<B> {
    pub fn new(backend: B, renderer: Renderer, readout: Readout, model: impl Into<String>) -> Self {
        Engine {
            backend,
            renderer,
            readout,
            temperature: 1.0,
            model: model.into(),
            baseline: None,
            baseline_cache: HashMap::new(),
        }
    }

    /// Option logits for every branch over the content-free state `cf`,
    /// from the cache where possible. Returns whether the backend was called.
    fn baseline_logits(
        &mut self,
        req: &Request,
        rendered: &Rendered,
        packed: &Packed,
        label_ids: &[Token],
        cf: &str,
    ) -> Result<(Vec<Vec<f32>>, bool)> {
        let keys: Vec<String> = rendered
            .branches
            .iter()
            .map(|b| serde_json::to_string(&(cf, &b.segments, &b.keys)).unwrap_or_default())
            .collect();
        let missing: Vec<usize> = (0..keys.len())
            .filter(|&i| !self.baseline_cache.contains_key(&keys[i]))
            .collect();
        let evaluated = !missing.is_empty();
        if evaluated {
            let mut cf_req = req.clone();
            cf_req.state = serde_json::Value::String(cf.to_string());
            let cf_prefix = self.renderer.render(&cf_req).prefix;
            let (cf_prefix, _) = self.tokenize_segments(&cf_prefix)?;
            let branches: Vec<BranchTokens> = missing
                .iter()
                .map(|&i| packed.branches[i].clone())
                .collect();
            let outs = self
                .backend
                .evaluate(&cf_prefix, &branches, self.readout.want())?;
            for (&i, out) in missing.iter().zip(outs) {
                let dist =
                    self.readout
                        .distribution(&rendered.branches[i], &out, label_ids, 1.0)?;
                self.baseline_cache.insert(keys[i].clone(), dist.logits);
            }
        }
        Ok((
            keys.iter()
                .map(|k| self.baseline_cache[k].clone())
                .collect(),
            evaluated,
        ))
    }

    fn tokenize_segments(&self, segments: &[Segment]) -> Result<(Vec<Token>, Vec<usize>)> {
        // Returns tokens and, per segment, the index of its last token.
        let mut tokens = Vec::new();
        let mut ends = Vec::with_capacity(segments.len());
        for s in segments {
            match s {
                Segment::Bos => tokens.push(self.backend.bos()),
                Segment::Special(name) => tokens.push(self.backend.special(name)?),
                Segment::Text(t) => tokens.extend(self.backend.tokenize(t)?),
            }
            ends.push(tokens.len().saturating_sub(1));
        }
        Ok((tokens, ends))
    }

    pub fn pack(&self, rendered: &Rendered) -> Result<Packed> {
        let (prefix, _) = self.tokenize_segments(&rendered.prefix)?;
        let mut branches = Vec::with_capacity(rendered.branches.len());
        for b in &rendered.branches {
            let (tokens, ends) = self.tokenize_segments(&b.segments)?;
            let want = b
                .marks
                .iter()
                .map(|(seg, mark)| match mark {
                    Mark::Last => tokens.len() - 1,
                    _ => ends[*seg],
                })
                .collect();
            branches.push(BranchTokens { tokens, want });
        }
        Ok(Packed { prefix, branches })
    }

    /// Distributions for every question, in request order.
    pub fn distributions(
        &mut self,
        req: &Request,
        orders: &IndexMap<String, Vec<usize>>,
        mode: Mode,
    ) -> Result<(Vec<(RenderedBranch, Distribution)>, Diagnostics)> {
        req.validate()?;
        if matches!(self.readout, Readout::Label) {
            let max_k = req
                .questions
                .values()
                .map(|q| match q {
                    crate::api::Question::Choice { criteria, .. } => criteria.len(),
                    crate::api::Question::Score { criteria, .. } => criteria.len(),
                    crate::api::Question::Noul { .. } => 2,
                })
                .max()
                .unwrap_or(0);
            if max_k > crate::readout::LABELS.len() {
                return Err(crate::Error::invalid(
                    "questions",
                    format!(
                        "label readout supports at most {} options",
                        crate::readout::LABELS.len()
                    ),
                ));
            }
        }
        let rendered = self.renderer.render_with(req, orders);
        let packed = self.pack(&rendered)?;
        let max_k = rendered
            .branches
            .iter()
            .map(|b| b.keys.len())
            .max()
            .unwrap_or(0);
        let label_ids = match self.readout {
            Readout::Label => Readout::label_ids(&self.backend, max_k)?,
            Readout::Pointer(_) => Vec::new(),
        };
        let want: Want = self.readout.want();
        let mut diag = Diagnostics {
            prefix_tokens: packed.prefix.len(),
            branch_tokens: packed.branches.iter().map(|b| b.tokens.len()).collect(),
            ..Default::default()
        };
        // The baseline pass goes first so the live state, not "N/A", is what
        // stays resident in the backend.
        let baseline = match self.baseline.clone() {
            Some(cf) => {
                let (b, evaluated) =
                    self.baseline_logits(req, &rendered, &packed, &label_ids, &cf)?;
                diag.passes += usize::from(evaluated);
                Some(b)
            }
            None => None,
        };
        let outputs = match mode {
            Mode::Packed => {
                diag.passes += 1;
                self.backend
                    .evaluate(&packed.prefix, &packed.branches, want)?
            }
            Mode::Separate => {
                let mut outs = Vec::with_capacity(packed.branches.len());
                for b in &packed.branches {
                    outs.extend(self.backend.evaluate(
                        &packed.prefix,
                        std::slice::from_ref(b),
                        want,
                    )?);
                }
                diag.passes += packed.branches.len();
                outs
            }
        };
        diag.prefix_source = self.backend.prefix_source();
        let mut result = Vec::with_capacity(outputs.len());
        for (i, (branch, out)) in rendered.branches.into_iter().zip(outputs).enumerate() {
            let mut dist =
                self.readout
                    .distribution(&branch, &out, &label_ids, self.temperature)?;
            if let Some(m) = dist.candidate_mass {
                diag.candidate_mass.insert(branch.id.clone(), m);
            }
            if let Some(b) = &baseline {
                dist.calibrate(b[i].clone(), self.temperature);
                diag.baseline.insert(branch.id.clone(), b[i].clone());
            }
            result.push((branch, dist));
        }
        Ok((result, diag))
    }

    /// Full TypeSafe-shaped response.
    pub fn answer(&mut self, req: &Request, mode: Mode) -> Result<(Response, Diagnostics)> {
        let (dists, diag) = self.distributions(req, &IndexMap::new(), mode)?;
        let mut answers = IndexMap::with_capacity(dists.len());
        for (branch, dist) in &dists {
            let q = &req.questions[&branch.id];
            answers.insert(branch.id.clone(), to_answer(q, branch, dist));
        }
        let input_tokens = diag.prefix_tokens + diag.branch_tokens.iter().sum::<usize>();
        let response = Response {
            model: self.model.clone(),
            answers,
            usage: Usage {
                input_tokens: input_tokens as u64,
                output_tokens: 0,
            },
        };
        Ok((response, diag))
    }
}

/// Map a distribution over rendered options back onto the question's own keys.
pub fn to_answer(q: &Question, branch: &RenderedBranch, dist: &Distribution) -> Answer {
    // p_orig[i] = probability of the option that was at original index i.
    let k = branch.keys.len();
    let mut p_orig = vec![0.0; k];
    for (slot, &orig) in branch.order.iter().enumerate() {
        p_orig[orig] = dist.probs[slot];
    }
    match (q, branch.kind) {
        (Question::Noul { .. }, Kind::Noul) => Answer::Noul { noul: p_orig[0] },
        (Question::Choice { criteria, .. }, Kind::Choice) => {
            let probabilities: IndexMap<String, f64> = criteria
                .keys()
                .cloned()
                .zip(p_orig.iter().copied())
                .collect();
            let best = argmax(&p_orig);
            Answer::Choice {
                choice: criteria.keys().nth(best).cloned().unwrap_or_default(),
                probabilities,
                confidence: confidence(&p_orig),
            }
        }
        (Question::Score { criteria, .. }, Kind::Score) => {
            let legend = criteria
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), v.clone()))
                .collect();
            let probabilities = p_orig
                .iter()
                .enumerate()
                .map(|(i, &p)| (i.to_string(), p))
                .collect();
            Answer::Score {
                score: expected_index(&p_orig),
                legend,
                probabilities,
                confidence: confidence(&p_orig),
            }
        }
        _ => unreachable!("renderer kind must match the question type"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BranchOutput;
    use std::cell::Cell;

    /// A backend with a fixed "yes" lean: every label-readout row puts label
    /// A 3 logits above B, plus a bonus for A when the prefix contains the
    /// evidence token. Counts `evaluate` calls.
    struct Leaning {
        calls: Cell<usize>,
    }

    const EVIDENCE: i32 = 999;

    impl Backend for Leaning {
        fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
            Ok(text
                .split_whitespace()
                .map(|w| {
                    Token(match w {
                        "A" => 1,
                        "B" => 2,
                        "evidence" => EVIDENCE,
                        _ => 3,
                    })
                })
                .collect())
        }
        fn special(&self, _: &str) -> Result<Token> {
            Ok(Token(2))
        }
        fn bos(&self) -> Token {
            Token(0)
        }
        fn n_embd(&self) -> usize {
            1
        }
        fn n_vocab(&self) -> usize {
            4
        }
        fn evaluate(
            &mut self,
            prefix: &[Token],
            branches: &[BranchTokens],
            _: Want,
        ) -> Result<Vec<BranchOutput>> {
            self.calls.set(self.calls.get() + 1);
            let seen = prefix.iter().any(|t| t.0 == EVIDENCE);
            Ok(branches
                .iter()
                .map(|_| BranchOutput {
                    // row[1] = label A, row[2] = label B (see `tokenize`).
                    rows: vec![vec![0.0, 3.0 + if seen { 6.0 } else { 0.0 }, 0.0, 0.0]],
                })
                .collect())
        }
    }

    fn request(state: &str) -> Request {
        serde_json::from_value(serde_json::json!({
            "state": state,
            "questions": {"q": {"type": "noul", "instructions": "is it so"}}
        }))
        .unwrap()
    }

    #[test]
    fn baseline_removes_the_lean_and_is_cached() {
        let backend = Leaning {
            calls: Cell::new(0),
        };
        let mut engine = Engine::new(backend, Renderer::gemma_label(), Readout::Label, "t");
        // Uncalibrated: the lean reads as 95% yes with nothing in the state.
        let (d, _) = engine
            .distributions(&request("nothing here"), &IndexMap::new(), Mode::Packed)
            .unwrap();
        assert!(d[0].1.probs[0] > 0.9);

        engine.baseline = Some("N/A".into());
        let (d, diag) = engine
            .distributions(&request("nothing here"), &IndexMap::new(), Mode::Packed)
            .unwrap();
        assert!((d[0].1.probs[0] - 0.5).abs() < 1e-9, "{:?}", d[0].1.probs);
        assert_eq!(diag.passes, 2);
        assert_eq!(diag.baseline["q"], vec![3.0, 0.0]);

        // Same question over a state with evidence: the baseline is served
        // from the cache (one pass) and the evidence survives calibration.
        let calls = engine.backend.calls.get();
        let (d, diag) = engine
            .distributions(&request("the evidence"), &IndexMap::new(), Mode::Packed)
            .unwrap();
        assert_eq!(engine.backend.calls.get(), calls + 1);
        assert_eq!(diag.passes, 1);
        assert!(d[0].1.probs[0] > 0.99);
    }
}
