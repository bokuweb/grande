//! Glue: render → tokenize/pack → evaluate → read out → typed answers.

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
    /// Number of `evaluate` calls (1 when packed).
    pub passes: usize,
    /// Where the backend got the state from (resident / ram / disk / decoded).
    pub prefix_source: Option<PrefixSource>,
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
        }
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
        let outputs = match mode {
            Mode::Packed => {
                diag.passes = 1;
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
                diag.passes = packed.branches.len();
                outs
            }
        };
        diag.prefix_source = self.backend.prefix_source();
        let mut result = Vec::with_capacity(outputs.len());
        for (branch, out) in rendered.branches.into_iter().zip(outputs) {
            let dist = self
                .readout
                .distribution(&branch, &out, &label_ids, self.temperature)?;
            if let Some(m) = dist.candidate_mass {
                diag.candidate_mass.insert(branch.id.clone(), m);
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
