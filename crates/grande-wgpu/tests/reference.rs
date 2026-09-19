//! The shaders against a plain-Rust Gemma 3 forward pass on a tiny random
//! model: same weights, same packed request, rows must agree to f16 weight
//! precision. Also checks isolation the direct way — a branch's rows do not
//! change when a sibling branch changes.
//!
//! Skips (passes) when no GPU adapter is available, e.g. on CI.

use grande_core::{BranchTokens, Token, Want};
use grande_wgpu::model::{Config, Layer, Tensor16, Weights};
use grande_wgpu::Engine;
use half::f16;

const D: usize = 64;
const HD: usize = 256;
const HEADS: usize = 2;
const FF: usize = 96;
const VOCAB: usize = 40;
const LAYERS: usize = 3;
const WINDOW: usize = 6;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        // xorshift, uniform in [-1, 1)
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn tensor(&mut self, shape: &[usize], scale: f32) -> Tensor16 {
        let n: usize = shape.iter().product();
        Tensor16 {
            shape: shape.to_vec(),
            data: (0..n).map(|_| f16::from_f32(self.next() * scale)).collect(),
        }
    }
}

fn config() -> Config {
    Config {
        vocab: VOCAB,
        d: D,
        layers: LAYERS,
        heads: HEADS,
        kv_heads: 1,
        head_dim: HD,
        ff: FF,
        eps: 1e-6,
        window: WINDOW,
        sliding: vec![true, false, true],
        theta_global: 1e6,
        theta_local: 1e4,
        query_scale: (HD as f32).powf(-0.5),
        bos: 2,
    }
}

fn weights(seed: u64) -> Weights {
    let mut r = Rng(seed | 1);
    let layers = (0..LAYERS)
        .map(|_| Layer {
            input_norm: r.tensor(&[D], 0.5),
            qkv: r.tensor(&[(HEADS + 2) * HD, D], 0.15),
            q_norm: r.tensor(&[HD], 0.5),
            k_norm: r.tensor(&[HD], 0.5),
            o: r.tensor(&[D, HEADS * HD], 0.08),
            post_attn_norm: r.tensor(&[D], 0.5),
            pre_ff_norm: r.tensor(&[D], 0.5),
            gate_up: r.tensor(&[2 * FF, D], 0.2),
            down: r.tensor(&[D, FF], 0.15),
            post_ff_norm: r.tensor(&[D], 0.5),
        })
        .collect();
    Weights {
        config: config(),
        embed: r.tensor(&[VOCAB, D], 1.0),
        layers,
        final_norm: r.tensor(&[D], 0.5),
    }
}

fn f(t: &Tensor16, i: usize) -> f32 {
    t.data[i].to_f32()
}

fn rmsnorm(x: &[f32], w: &Tensor16, eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| v * inv * (1.0 + f(w, i)))
        .collect()
}

/// y[n] = sum_k x[k] * W[n, k]
fn linear(x: &[f32], w: &Tensor16) -> Vec<f32> {
    let (n, k) = (w.shape[0], w.shape[1]);
    (0..n)
        .map(|r| (0..k).map(|c| x[c] * f(w, r * k + c)).sum())
        .collect()
}

fn rope(v: &mut [f32], pos: i32, theta: f32) {
    let half = HD / 2;
    for i in 0..half {
        let ang = pos as f32 * theta.powf(-(i as f32) / half as f32);
        let (s, c) = ang.sin_cos();
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * c - b * s;
        v[i + half] = b * c + a * s;
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}

/// Plain forward over the packed sequence; returns the final hidden state per token.
fn reference(w: &Weights, ids: &[u32], meta: &[(i32, i32)]) -> Vec<Vec<f32>> {
    let cfg = &w.config;
    let t = ids.len();
    let mut x: Vec<Vec<f32>> = ids
        .iter()
        .map(|&id| {
            (0..D)
                .map(|i| f(&w.embed, id as usize * D + i) * (D as f32).sqrt())
                .collect()
        })
        .collect();
    for (li, l) in w.layers.iter().enumerate() {
        let sliding = cfg.sliding[li];
        let theta = if sliding {
            cfg.theta_local
        } else {
            cfg.theta_global
        };
        let mut q = vec![vec![0f32; HEADS * HD]; t];
        let mut k = vec![vec![0f32; HD]; t];
        let mut v = vec![vec![0f32; HD]; t];
        for i in 0..t {
            let h = rmsnorm(&x[i], &l.input_norm, cfg.eps);
            let qkv = linear(&h, &l.qkv);
            for hd in 0..HEADS {
                let mut qh = rmsnorm(&qkv[hd * HD..(hd + 1) * HD], &l.q_norm, cfg.eps);
                rope(&mut qh, meta[i].0, theta);
                for d in 0..HD {
                    q[i][hd * HD + d] = qh[d] * cfg.query_scale;
                }
            }
            let mut kk = rmsnorm(&qkv[HEADS * HD..(HEADS + 1) * HD], &l.k_norm, cfg.eps);
            rope(&mut kk, meta[i].0, theta);
            k[i] = kk;
            v[i] = qkv[(HEADS + 1) * HD..].to_vec();
        }
        for i in 0..t {
            let (qp, qs) = meta[i];
            let mut attn = vec![0f32; HEADS * HD];
            for hd in 0..HEADS {
                let qh = &q[i][hd * HD..(hd + 1) * HD];
                let mut scores = Vec::new();
                for j in 0..t {
                    let (kp, ks) = meta[j];
                    let visible = (ks == 0 || ks == qs)
                        && kp <= qp
                        && (!sliding || (qp - kp) < WINDOW as i32);
                    if visible {
                        scores.push((j, qh.iter().zip(&k[j]).map(|(a, b)| a * b).sum::<f32>()));
                    }
                }
                let m = scores.iter().map(|s| s.1).fold(f32::MIN, f32::max);
                let l_sum: f32 = scores.iter().map(|s| (s.1 - m).exp()).sum();
                for (j, sc) in &scores {
                    let p = (sc - m).exp() / l_sum;
                    for d in 0..HD {
                        attn[hd * HD + d] += p * v[*j][d];
                    }
                }
            }
            let a = rmsnorm(&linear(&attn, &l.o), &l.post_attn_norm, cfg.eps);
            for d in 0..D {
                x[i][d] += a[d];
            }
            let h = rmsnorm(&x[i], &l.pre_ff_norm, cfg.eps);
            let gu = linear(&h, &l.gate_up);
            let act: Vec<f32> = (0..FF).map(|j| gelu(gu[j]) * gu[FF + j]).collect();
            let m = rmsnorm(&linear(&act, &l.down), &l.post_ff_norm, cfg.eps);
            for d in 0..D {
                x[i][d] += m[d];
            }
        }
    }
    x.iter()
        .map(|row| rmsnorm(row, &w.final_norm, cfg.eps))
        .collect()
}

fn engine(w: &Weights) -> Option<Engine> {
    match pollster::block_on(Engine::new(w, 64, 16)) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("no GPU, skipping: {e:#}");
            None
        }
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn matches_cpu_reference() {
    let w = weights(7);
    let Some(eng) = engine(&w) else { return };
    let prefix: Vec<u32> = vec![2, 5, 9, 14, 20, 31, 7, 8, 3, 22];
    let branches = vec![
        BranchTokens {
            tokens: [11, 12, 13, 14, 15].map(Token).to_vec(),
            want: vec![1, 4],
        },
        BranchTokens {
            tokens: [30, 31, 32, 33, 34, 35, 36, 37].map(Token).to_vec(),
            want: vec![0, 7],
        },
    ];
    let out = pollster::block_on(eng.evaluate(&prefix, &branches, Want::Hidden)).unwrap();

    let mut ids = prefix.clone();
    let mut meta: Vec<(i32, i32)> = (0..prefix.len()).map(|i| (i as i32, 0)).collect();
    for (bi, b) in branches.iter().enumerate() {
        for (j, t) in b.tokens.iter().enumerate() {
            ids.push(t.0 as u32);
            meta.push(((prefix.len() + j) as i32, bi as i32 + 1));
        }
    }
    let hidden = reference(&w, &ids, &meta);
    let mut idx = prefix.len();
    for (b, o) in branches.iter().zip(&out) {
        for (slot, &want) in b.want.iter().enumerate() {
            let got = &o.rows[slot];
            let exp = &hidden[idx + want];
            let scale = exp.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1.0);
            let d = max_abs_diff(got, exp) / scale;
            assert!(d < 2e-2, "row {slot} of branch: relative diff {d}");
        }
        idx += b.tokens.len();
    }
}

#[test]
fn branches_do_not_see_each_other() {
    let w = weights(11);
    let Some(eng) = engine(&w) else { return };
    let prefix: Vec<u32> = vec![2, 4, 6, 8];
    let a = BranchTokens {
        tokens: [10, 11, 12].map(Token).to_vec(),
        want: vec![2],
    };
    let b1 = BranchTokens {
        tokens: [20, 21, 22, 23].map(Token).to_vec(),
        want: vec![3],
    };
    let b2 = BranchTokens {
        tokens: [30, 31].map(Token).to_vec(),
        want: vec![1],
    };
    let with_b1 =
        pollster::block_on(eng.evaluate(&prefix, &[a.clone(), b1], Want::Hidden)).unwrap();
    let with_b2 =
        pollster::block_on(eng.evaluate(&prefix, &[a.clone(), b2], Want::Hidden)).unwrap();
    let alone = pollster::block_on(eng.evaluate(&prefix, &[a], Want::Hidden)).unwrap();
    assert_eq!(with_b1[0].rows[0], with_b2[0].rows[0]);
    assert_eq!(with_b1[0].rows[0], alone[0].rows[0]);
}

#[test]
fn logits_are_hidden_times_embedding() {
    let w = weights(3);
    let Some(eng) = engine(&w) else { return };
    let prefix: Vec<u32> = vec![2, 1, 3];
    let branches = vec![BranchTokens {
        tokens: [4, 5].map(Token).to_vec(),
        want: vec![1],
    }];
    let hidden = pollster::block_on(eng.evaluate(&prefix, &branches, Want::Hidden)).unwrap();
    let logits = pollster::block_on(eng.evaluate(&prefix, &branches, Want::Logits)).unwrap();
    let h = &hidden[0].rows[0];
    let expected: Vec<f32> = (0..VOCAB)
        .map(|v| (0..D).map(|i| h[i] * f(&w.embed, v * D + i)).sum())
        .collect();
    let scale = expected
        .iter()
        .map(|v| v.abs())
        .fold(0.0, f32::max)
        .max(1.0);
    assert!(max_abs_diff(&logits[0].rows[0], &expected) / scale < 1e-3);
}
