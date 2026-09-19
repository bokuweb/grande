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

/// Minimal safetensors reader for the pointer head. Expects tensors
/// `q.weight [dp, d]`, `q.bias [dp]`, `k.weight [dp, d]`, `k.bias [dp]` in
/// F32, F16 or BF16 (kev's `PointerHead` naming, exported with
/// `safetensors.torch.save_file(head.state_dict(), "head.safetensors")`).
pub mod safetensors {
    use super::PointerHead;
    use crate::{Error, Result};
    use serde_json::Value;

    fn bad(m: impl Into<String>) -> Error {
        Error::Backend(format!("head.safetensors: {}", m.into()))
    }

    fn tensor(
        bytes: &[u8],
        header: &Value,
        base: usize,
        name: &str,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let t = header
            .get(name)
            .ok_or_else(|| bad(format!("missing tensor {name}")))?;
        let dtype = t["dtype"].as_str().unwrap_or_default();
        let shape: Vec<usize> = t["shape"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(Value::as_u64)
            .map(|v| v as usize)
            .collect();
        let off = t["data_offsets"]
            .as_array()
            .ok_or_else(|| bad("data_offsets"))?;
        let (a, b) = (
            off[0].as_u64().unwrap_or(0) as usize + base,
            off[1].as_u64().unwrap_or(0) as usize + base,
        );
        let raw = bytes.get(a..b).ok_or_else(|| bad("offsets out of range"))?;
        let data: Vec<f32> = match dtype {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect(),
            other => return Err(bad(format!("unsupported dtype {other}"))),
        };
        if data.len() != shape.iter().product::<usize>() {
            return Err(bad(format!(
                "{name}: {} values for shape {shape:?}",
                data.len()
            )));
        }
        Ok((shape, data))
    }

    fn f16_to_f32(h: u16) -> f32 {
        let sign = u32::from(h >> 15) << 31;
        let exp = u32::from((h >> 10) & 0x1f);
        let frac = u32::from(h & 0x3ff);
        let bits = if exp == 0 {
            if frac == 0 {
                sign
            } else {
                // subnormal
                let mut e = 127 - 15 + 1;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                sign | ((e as u32) << 23) | ((f & 0x3ff) << 13)
            }
        } else if exp == 0x1f {
            sign | 0x7f80_0000 | (frac << 13)
        } else {
            sign | ((exp + 127 - 15) << 23) | (frac << 13)
        };
        f32::from_bits(bits)
    }

    pub fn load(bytes: &[u8]) -> Result<PointerHead> {
        if bytes.len() < 8 {
            return Err(bad("too short"));
        }
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: Value =
            serde_json::from_slice(bytes.get(8..8 + n).ok_or_else(|| bad("header"))?)
                .map_err(|e| bad(e.to_string()))?;
        let base = 8 + n;
        let (sq, w_q) = tensor(bytes, &header, base, "q.weight")?;
        let (_, b_q) = tensor(bytes, &header, base, "q.bias")?;
        let (sk, w_k) = tensor(bytes, &header, base, "k.weight")?;
        let (_, b_k) = tensor(bytes, &header, base, "k.bias")?;
        if sq.len() != 2 || sq != sk {
            return Err(bad(format!("weight shapes {sq:?} / {sk:?}")));
        }
        let (dp, d) = (sq[0], sq[1]);
        if b_q.len() != dp || b_k.len() != dp {
            return Err(bad("bias shape"));
        }
        Ok(PointerHead {
            d,
            dp,
            w_q,
            b_q,
            w_k,
            b_k,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn st(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
            let mut data = Vec::new();
            let mut header = serde_json::Map::new();
            for (name, shape, vals) in tensors {
                let start = data.len();
                for v in *vals {
                    data.extend_from_slice(&v.to_le_bytes());
                }
                header.insert(
                    name.to_string(),
                    serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
                );
            }
            let h = serde_json::to_vec(&Value::Object(header)).unwrap();
            let mut out = (h.len() as u64).to_le_bytes().to_vec();
            out.extend(h);
            out.extend(data);
            out
        }

        #[test]
        fn roundtrip_small_head() {
            let bytes = st(&[
                ("q.weight", &[2, 3], &[1., 0., 0., 0., 1., 0.]),
                ("q.bias", &[2], &[0., 0.]),
                ("k.weight", &[2, 3], &[1., 0., 0., 0., 1., 0.]),
                ("k.bias", &[2], &[0.5, 0.]),
            ]);
            let head = load(&bytes).unwrap();
            assert_eq!((head.d, head.dp), (3, 2));
            let z = head.logits(&[1., 0., 0.], &[&[1., 0., 0.], &[0., 1., 0.]]);
            assert!(z[0] > z[1]);
        }

        #[test]
        fn f16_conversion() {
            assert_eq!(f16_to_f32(0x3c00), 1.0);
            assert_eq!(f16_to_f32(0xc000), -2.0);
            assert_eq!(f16_to_f32(0x0000), 0.0);
        }
    }
}
