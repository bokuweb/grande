//! From backend rows to a distribution over the rendered options.

use crate::backend::{Backend, BranchOutput, Token, Want};
use crate::math::{dot, log_sum_exp, softmax};
use crate::render::{Mark, RenderedBranch};
use crate::{Error, Result};

/// Single-character option labels for the label readout. 52 is the cap for a
/// tokenizer that splits digits (Gemma does).
pub const LABELS: [char; 52] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
];

/// A trained pointer head: two affine maps `d -> dp`, scaled dot product.
#[derive(Debug, Clone)]
pub struct PointerHead {
    pub d: usize,
    pub dp: usize,
    /// Row-major `[dp, d]`.
    pub w_q: Vec<f32>,
    pub b_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub b_k: Vec<f32>,
}

impl PointerHead {
    fn project(&self, w: &[f32], b: &[f32], h: &[f32]) -> Vec<f32> {
        (0..self.dp)
            .map(|r| dot(&w[r * self.d..(r + 1) * self.d], h) + b[r])
            .collect()
    }

    /// Logits over options from the decide row and one row per option.
    pub fn logits(&self, decide: &[f32], opts: &[&[f32]]) -> Vec<f32> {
        let q = self.project(&self.w_q, &self.b_q, decide);
        let scale = 1.0 / (self.dp as f32).sqrt();
        opts.iter()
            .map(|h| dot(&self.project(&self.w_k, &self.b_k, h), &q) * scale)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub enum Readout {
    /// Read next-token logits at the branch's last position, keep the label
    /// tokens, normalize among them.
    Label,
    /// Read hidden states at every `</opt>` and at `<decide>`, score with a
    /// pointer head.
    Pointer(PointerHead),
}

/// Probabilities over the branch's rendered options plus diagnostics.
#[derive(Debug, Clone)]
pub struct Distribution {
    /// Raw option logits before temperature (kept so callers can refit).
    pub logits: Vec<f32>,
    pub probs: Vec<f64>,
    /// Label readout only: share of full-vocabulary probability mass that
    /// landed on the label tokens. Low values mean the model wanted to say
    /// something else first and the normalized answer should not be trusted.
    pub candidate_mass: Option<f64>,
}

impl Readout {
    pub fn want(&self) -> Want {
        match self {
            Readout::Label => Want::Logits,
            Readout::Pointer(_) => Want::Hidden,
        }
    }

    /// Resolve label token ids for `k` options; every label must be one token.
    pub fn label_ids(backend: &dyn Backend, k: usize) -> Result<Vec<Token>> {
        if k > LABELS.len() {
            return Err(Error::invalid(
                "questions",
                format!("label readout supports at most {} options", LABELS.len()),
            ));
        }
        LABELS[..k]
            .iter()
            .map(|c| {
                let s = c.to_string();
                let t = backend.tokenize(&s)?;
                if t.len() != 1 {
                    return Err(Error::LabelNotSingleToken(s));
                }
                Ok(t[0])
            })
            .collect()
    }

    pub fn distribution(
        &self,
        branch: &RenderedBranch,
        out: &BranchOutput,
        label_ids: &[Token],
        temperature: f32,
    ) -> Result<Distribution> {
        match self {
            Readout::Label => {
                let row = out
                    .rows
                    .first()
                    .ok_or_else(|| Error::Backend("no logits row".into()))?;
                let k = branch.keys.len();
                let logits: Vec<f32> = label_ids[..k].iter().map(|t| row[t.0 as usize]).collect();
                let lse_all = log_sum_exp(row);
                let lse_labels = log_sum_exp(&logits);
                let probs = softmax(&logits, temperature);
                Ok(Distribution {
                    logits,
                    probs,
                    candidate_mass: Some((lse_labels - lse_all).exp()),
                })
            }
            Readout::Pointer(head) => {
                let k = branch.keys.len();
                let mut opts: Vec<&[f32]> = Vec::with_capacity(k);
                let mut decide: Option<&[f32]> = None;
                for ((_, mark), row) in branch.marks.iter().zip(&out.rows) {
                    match mark {
                        Mark::OptEnd(_) => opts.push(row),
                        Mark::Decide => decide = Some(row),
                        Mark::Last => {}
                    }
                }
                let decide = decide.ok_or_else(|| Error::Backend("no decide row".into()))?;
                if opts.len() != k {
                    return Err(Error::Backend(format!(
                        "expected {k} option rows, got {}",
                        opts.len()
                    )));
                }
                let logits = head.logits(decide, &opts);
                let probs = softmax(&logits, temperature);
                Ok(Distribution {
                    logits,
                    probs,
                    candidate_mass: None,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_head_identity_projection_is_a_dot_product() {
        let d = 4;
        let mut w = vec![0.0; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0;
        }
        let head = PointerHead {
            d,
            dp: d,
            w_q: w.clone(),
            b_q: vec![0.0; d],
            w_k: w,
            b_k: vec![0.0; d],
        };
        let decide = [1.0, 0.0, 0.0, 0.0];
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        let z = head.logits(&decide, &[&a, &b]);
        assert!(z[0] > z[1]);
        assert!((z[0] - 0.5).abs() < 1e-6); // 1 / sqrt(4)
    }
}
