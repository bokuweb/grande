//! The shaders against a plain-Rust forward pass on tiny random models: same
//! weights, same packed request, rows must agree to weight precision. Two
//! configurations: a Gemma 3 shape (f16, one head_dim, own K/V everywhere)
//! and a Gemma 4 shape (Q8 / Q4 weights, head_dim 256 sliding + 512 global
//! with partial RoPE, shared K/V, V norm, per-layer embeddings, output
//! scalars, softcapped logits). Also checks isolation the direct way — a
//! branch's rows do not change when a sibling branch changes.
//!
//! Skips (passes) when no GPU adapter is available, e.g. on CI.

use std::collections::HashMap;

use grande_core::{BranchTokens, PrefixSource, Token, Want};
use grande_wgpu::model::{Arch, Config, Dtype, QTensor, Weights};
use grande_wgpu::Engine;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        // xorshift, uniform in [-1, 1)
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn values(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
    fn tensor(&mut self, shape: &[usize], scale: f32, dtype: Dtype) -> QTensor {
        let n: usize = shape.iter().product();
        let v = self.values(n, scale);
        match dtype {
            Dtype::F16 => QTensor::from_f32(shape.to_vec(), &v),
            Dtype::Q8 => QTensor::quantize_q8(shape.to_vec(), &v).unwrap(),
            Dtype::Q4 => QTensor::quantize_q4(shape.to_vec(), &v).unwrap(),
        }
    }
}

fn gemma3_config() -> Config {
    let layers = 3;
    Config {
        arch: Arch::Gemma3,
        vocab: 40,
        d: 64,
        layers,
        heads: 2,
        kv_heads: 1,
        head_dim: vec![256; layers],
        ff: vec![96; layers],
        sliding: vec![true, false, true],
        kv_source: (0..layers).collect(),
        rope_dims: vec![256; layers],
        eps: 1e-6,
        window: 6,
        theta_global: 1e6,
        theta_local: 1e4,
        query_scale: (256f32).powf(-0.5),
        norm_offset: 1.0,
        v_norm: false,
        per_layer_dim: 0,
        softcap: 0.0,
        bos: 2,
    }
}

/// Gemma 4 shape: 4 layers, the last two share K/V with the first two of
/// their type (layer 2 sliding -> 0, layer 3 global -> 1).
fn gemma4_config() -> Config {
    let layers = 4;
    Config {
        arch: Arch::Gemma4,
        vocab: 64,
        d: 64,
        layers,
        heads: 4,
        kv_heads: 1,
        head_dim: vec![256, 512, 256, 512],
        ff: vec![96, 96, 192, 192],
        sliding: vec![true, false, true, false],
        kv_source: vec![0, 1, 0, 1],
        rope_dims: vec![256, 128, 256, 128],
        eps: 1e-6,
        window: 5,
        theta_global: 1e6,
        theta_local: 1e4,
        query_scale: 1.0,
        norm_offset: 0.0,
        v_norm: true,
        per_layer_dim: 32,
        softcap: 30.0,
        bos: 2,
    }
}

/// Random weights for a config. Linears use `linear`, norms are f16 around 1
/// (Gemma 4) or 0 (Gemma 3), the embedding uses `embed`.
fn weights(cfg: &Config, seed: u64, linear: Dtype, embed: Dtype) -> Weights {
    let mut r = Rng(seed | 1);
    let mut tensors = HashMap::new();
    let d = cfg.d;
    let norm = |r: &mut Rng, n: usize| -> QTensor {
        let base = if cfg.norm_offset == 0.0 { 1.0 } else { 0.0 };
        let v: Vec<f32> = r.values(n, 0.5).into_iter().map(|x| x + base).collect();
        QTensor::from_f32(vec![n], &v)
    };
    for spec in cfg.tensors() {
        let name = spec.name.as_str();
        let shape = spec.shape.clone();
        let t = if name == "embed" {
            r.tensor(&shape, 1.0, embed)
        } else if name.ends_with("out_scale") {
            QTensor::from_f32(vec![1], &[0.6 + 0.2 * r.next()])
        } else if shape.len() == 1 {
            norm(&mut r, shape[0])
        } else {
            let scale = if shape[1] == d { 0.15 } else { 0.08 };
            r.tensor(&shape, scale, linear)
        };
        tensors.insert(spec.name, t);
    }
    let per_layer_table = (cfg.per_layer_dim > 0)
        .then(|| r.tensor(&[cfg.vocab, cfg.per_layer_dim * cfg.layers], 1.0, linear));
    Weights {
        config: cfg.clone(),
        tensors,
        per_layer_table,
    }
}

fn rmsnorm(x: &[f32], w: Option<&QTensor>, eps: f32, offset: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| v * inv * w.map_or(1.0, |w| offset + w.get(i)))
        .collect()
}

/// y[n] = sum_k x[k] * W[n, k]
fn linear(x: &[f32], w: &QTensor) -> Vec<f32> {
    let (n, k) = (w.shape[0], w.shape[1]);
    let wf = w.to_f32();
    (0..n)
        .map(|r| (0..k).map(|c| x[c] * wf[r * k + c]).sum())
        .collect()
}

fn rope(v: &mut [f32], pos: i32, theta: f32, rope_dims: usize) {
    let half = v.len() / 2;
    for i in 0..rope_dims / 2 {
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
    let d = cfg.d;
    let heads = cfg.heads;
    let t = ids.len();
    let g = |name: &str| w.get(name).unwrap();
    let embed = g("embed");
    let mut x: Vec<Vec<f32>> = ids
        .iter()
        .map(|&id| {
            (0..d)
                .map(|i| embed.get(id as usize * d + i) * (d as f32).sqrt())
                .collect()
        })
        .collect();
    // Per-layer inputs [t][layers x P].
    let pl = cfg.per_layer_dim;
    let pli: Vec<Vec<f32>> = if pl > 0 {
        let table = w.per_layer_table.as_ref().unwrap();
        let proj = g("pl_model_proj");
        let pn = g("pl_proj_norm");
        (0..t)
            .map(|i| {
                let p = linear(&x[i], proj);
                let mut out = Vec::with_capacity(pl * cfg.layers);
                for l in 0..cfg.layers {
                    let slice: Vec<f32> = p[l * pl..(l + 1) * pl]
                        .iter()
                        .map(|v| v / (d as f32).sqrt())
                        .collect();
                    let n = rmsnorm(&slice, Some(pn), cfg.eps, cfg.norm_offset);
                    for (j, nv) in n.iter().enumerate() {
                        let e = table.get(ids[i] as usize * pl * cfg.layers + l * pl + j)
                            * (pl as f32).sqrt();
                        out.push((nv + e) / 2f32.sqrt());
                    }
                }
                out
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut kv_store: Vec<Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>> = vec![None; cfg.layers];
    for li in 0..cfg.layers {
        let n = |s: &str| format!("blk.{li}.{s}");
        let hd = cfg.head_dim[li];
        let sliding = cfg.sliding[li];
        let theta = if sliding {
            cfg.theta_local
        } else {
            cfg.theta_global
        };
        let has_kv = cfg.has_kv(li);
        let mut q = vec![vec![0f32; heads * hd]; t];
        let mut k = vec![vec![0f32; hd]; t];
        let mut v = vec![vec![0f32; hd]; t];
        for i in 0..t {
            let h = rmsnorm(&x[i], Some(g(&n("attn_norm"))), cfg.eps, cfg.norm_offset);
            let qkv = linear(&h, g(&n("qkv")));
            for hh in 0..heads {
                let mut qh = rmsnorm(
                    &qkv[hh * hd..(hh + 1) * hd],
                    Some(g(&n("q_norm"))),
                    cfg.eps,
                    cfg.norm_offset,
                );
                rope(&mut qh, meta[i].0, theta, cfg.rope_dims[li]);
                for dd in 0..hd {
                    q[i][hh * hd + dd] = qh[dd] * cfg.query_scale;
                }
            }
            if has_kv {
                let mut kk = rmsnorm(
                    &qkv[heads * hd..(heads + 1) * hd],
                    Some(g(&n("k_norm"))),
                    cfg.eps,
                    cfg.norm_offset,
                );
                rope(&mut kk, meta[i].0, theta, cfg.rope_dims[li]);
                k[i] = kk;
                let vv = &qkv[(heads + 1) * hd..];
                v[i] = if cfg.v_norm {
                    rmsnorm(vv, None, cfg.eps, 0.0)
                } else {
                    vv.to_vec()
                };
            }
        }
        if has_kv {
            kv_store[li] = Some((k, v));
        }
        let (k, v) = kv_store[cfg.kv_source[li]].as_ref().unwrap();
        for i in 0..t {
            let (qp, qs) = meta[i];
            let mut attn = vec![0f32; heads * hd];
            for hh in 0..heads {
                let qh = &q[i][hh * hd..(hh + 1) * hd];
                let mut scores = Vec::new();
                for j in 0..t {
                    let (kp, ks) = meta[j];
                    let visible = (ks == 0 || ks == qs)
                        && kp <= qp
                        && (!sliding || (qp - kp) < cfg.window as i32);
                    if visible {
                        scores.push((j, qh.iter().zip(&k[j]).map(|(a, b)| a * b).sum::<f32>()));
                    }
                }
                let m = scores.iter().map(|s| s.1).fold(f32::MIN, f32::max);
                let l_sum: f32 = scores.iter().map(|s| (s.1 - m).exp()).sum();
                for (j, sc) in &scores {
                    let p = (sc - m).exp() / l_sum;
                    for dd in 0..hd {
                        attn[hh * hd + dd] += p * v[*j][dd];
                    }
                }
            }
            let a = rmsnorm(
                &linear(&attn, g(&n("o"))),
                Some(g(&n("post_attn_norm"))),
                cfg.eps,
                cfg.norm_offset,
            );
            for dd in 0..d {
                x[i][dd] += a[dd];
            }
            let h = rmsnorm(&x[i], Some(g(&n("ffn_norm"))), cfg.eps, cfg.norm_offset);
            let gt = linear(&h, g(&n("gate")));
            let u = linear(&h, g(&n("up")));
            let act: Vec<f32> = (0..cfg.ff[li]).map(|j| gelu(gt[j]) * u[j]).collect();
            let m = rmsnorm(
                &linear(&act, g(&n("down"))),
                Some(g(&n("post_ffn_norm"))),
                cfg.eps,
                cfg.norm_offset,
            );
            for dd in 0..d {
                x[i][dd] += m[dd];
            }
            let mut scale = 1.0;
            if pl > 0 {
                let gate = linear(&x[i], g(&n("pl_gate")));
                let gated: Vec<f32> = (0..pl)
                    .map(|j| gelu(gate[j]) * pli[i][li * pl + j])
                    .collect();
                let o = rmsnorm(
                    &linear(&gated, g(&n("pl_proj"))),
                    Some(g(&n("pl_norm"))),
                    cfg.eps,
                    cfg.norm_offset,
                );
                for dd in 0..d {
                    x[i][dd] += o[dd];
                }
                scale = g(&n("out_scale")).get(0);
            }
            for dd in 0..d {
                x[i][dd] *= scale;
            }
        }
    }
    x.iter()
        .map(|row| rmsnorm(row, Some(g("final_norm")), cfg.eps, cfg.norm_offset))
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

fn pack(prefix: &[u32], branches: &[BranchTokens]) -> (Vec<u32>, Vec<(i32, i32)>) {
    let mut ids = prefix.to_vec();
    let mut meta: Vec<(i32, i32)> = (0..prefix.len()).map(|i| (i as i32, 0)).collect();
    for (bi, b) in branches.iter().enumerate() {
        for (j, t) in b.tokens.iter().enumerate() {
            ids.push(t.0 as u32);
            meta.push(((prefix.len() + j) as i32, bi as i32 + 1));
        }
    }
    (ids, meta)
}

fn check_hidden(w: &Weights, tol: f32) {
    let Some(eng) = engine(w) else { return };
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
    let (ids, meta) = pack(&prefix, &branches);
    let hidden = reference(w, &ids, &meta);
    let mut idx = prefix.len();
    for (b, o) in branches.iter().zip(&out) {
        for (slot, &want) in b.want.iter().enumerate() {
            let got = &o.rows[slot];
            let exp = &hidden[idx + want];
            let scale = exp.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1.0);
            let d = max_abs_diff(got, exp) / scale;
            assert!(d < tol, "row {slot} of branch: relative diff {d}");
        }
        idx += b.tokens.len();
    }
}

#[test]
fn gemma3_matches_cpu_reference() {
    check_hidden(&weights(&gemma3_config(), 7, Dtype::F16, Dtype::F16), 2e-2);
}

#[test]
fn gemma4_q8_matches_cpu_reference() {
    check_hidden(&weights(&gemma4_config(), 7, Dtype::Q8, Dtype::Q8), 2e-2);
}

#[test]
fn gemma4_q4_matches_cpu_reference() {
    // The reference dequantizes the same codes, so Q4 is no looser.
    check_hidden(&weights(&gemma4_config(), 9, Dtype::Q4, Dtype::Q8), 2e-2);
}

#[test]
fn gemma4_f16_matches_cpu_reference() {
    check_hidden(&weights(&gemma4_config(), 5, Dtype::F16, Dtype::F16), 2e-2);
}

/// A request over the prefix of the previous one runs only its branches on
/// top of the resident K/V and must produce the same rows as a full pass.
#[test]
fn resident_prefix_matches_full_pass() {
    for w in [
        weights(&gemma3_config(), 13, Dtype::F16, Dtype::F16),
        weights(&gemma4_config(), 13, Dtype::Q4, Dtype::Q8),
    ] {
        let Some(eng) = engine(&w) else { return };
        let prefix: Vec<u32> = vec![2, 5, 9, 14, 20, 31, 7];
        let branches = vec![
            BranchTokens {
                tokens: [11, 12, 13, 14, 15].map(Token).to_vec(),
                want: vec![1, 4],
            },
            BranchTokens {
                tokens: [30, 31, 32].map(Token).to_vec(),
                want: vec![0, 2],
            },
        ];
        let other = vec![BranchTokens {
            tokens: [33, 34, 35, 36, 37, 38].map(Token).to_vec(),
            want: vec![5],
        }];
        let full = pollster::block_on(eng.evaluate(&prefix, &branches, Want::Hidden)).unwrap();
        assert_eq!(eng.prefix_source(), Some(PrefixSource::Decoded));
        // Same prefix, different branches: resident, appended past the prefix.
        let cont = pollster::block_on(eng.evaluate(&prefix, &other, Want::Hidden)).unwrap();
        assert_eq!(eng.prefix_source(), Some(PrefixSource::Resident));
        eng.evict_resident();
        let cold = pollster::block_on(eng.evaluate(&prefix, &other, Want::Hidden)).unwrap();
        assert_eq!(eng.prefix_source(), Some(PrefixSource::Decoded));
        assert_eq!(cont[0].rows, cold[0].rows);
        // Back to the first branches, resident again.
        let again = pollster::block_on(eng.evaluate(&prefix, &branches, Want::Hidden)).unwrap();
        assert_eq!(eng.prefix_source(), Some(PrefixSource::Resident));
        for (a, b) in full.iter().zip(&again) {
            assert_eq!(a.rows, b.rows);
        }
        // A different prefix decodes and becomes resident in turn.
        let prefix2: Vec<u32> = vec![2, 3, 4];
        let p2 = pollster::block_on(eng.evaluate(&prefix2, &branches, Want::Logits)).unwrap();
        assert_eq!(eng.prefix_source(), Some(PrefixSource::Decoded));
        let p2r = pollster::block_on(eng.evaluate(&prefix2, &branches, Want::Logits)).unwrap();
        assert_eq!(eng.prefix_source(), Some(PrefixSource::Resident));
        for (a, b) in p2.iter().zip(&p2r) {
            assert_eq!(a.rows, b.rows);
        }
    }
}

#[test]
fn branches_do_not_see_each_other() {
    for w in [
        weights(&gemma3_config(), 11, Dtype::F16, Dtype::F16),
        weights(&gemma4_config(), 11, Dtype::Q8, Dtype::Q8),
    ] {
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
}

#[test]
fn logits_are_hidden_times_embedding() {
    for (w, softcap) in [
        (weights(&gemma3_config(), 3, Dtype::F16, Dtype::F16), 0.0),
        (weights(&gemma4_config(), 3, Dtype::Q8, Dtype::Q8), 30.0),
    ] {
        let Some(eng) = engine(&w) else { return };
        let cfg = &w.config;
        let prefix: Vec<u32> = vec![2, 1, 3];
        let branches = vec![BranchTokens {
            tokens: [4, 5].map(Token).to_vec(),
            want: vec![1],
        }];
        let hidden = pollster::block_on(eng.evaluate(&prefix, &branches, Want::Hidden)).unwrap();
        let logits = pollster::block_on(eng.evaluate(&prefix, &branches, Want::Logits)).unwrap();
        let h = &hidden[0].rows[0];
        let embed = w.get("embed").unwrap();
        let expected: Vec<f32> = (0..cfg.vocab)
            .map(|v| {
                let z: f32 = (0..cfg.d).map(|i| h[i] * embed.get(v * cfg.d + i)).sum();
                if softcap > 0.0 {
                    softcap * (z / softcap).tanh()
                } else {
                    z
                }
            })
            .collect();
        let scale = expected
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f32::max)
            .max(1.0);
        assert!(max_abs_diff(&logits[0].rows[0], &expected) / scale < 1e-3);
    }
}

#[test]
fn quantization_roundtrips() {
    let mut r = Rng(5);
    let v = r.values(256, 0.3);
    let q8 = QTensor::quantize_q8(vec![8, 32], &v).unwrap();
    let d8 = max_abs_diff(&q8.to_f32(), &v);
    assert!(d8 < 0.3 / 127.0 * 1.01, "q8 error {d8}");
    let q4 = QTensor::quantize_q4(vec![8, 32], &v).unwrap();
    let d4 = max_abs_diff(&q4.to_f32(), &v);
    assert!(d4 < 0.3 / 8.0 * 1.01, "q4 error {d4}");
}
