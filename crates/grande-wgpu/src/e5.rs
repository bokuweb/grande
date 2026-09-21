//! Sentence embedders on wgpu — multilingual-e5-small (BERT, MIT) and
//! Ruri v3 (ModernBERT-Ja, Apache-2.0) — plus the (state, option) head
//! `tools/e5_generic.py` trains: the cheapest System One measured in
//! docs/e5.md / docs/ruri.md. The encoder maps a text to a unit vector; a
//! request is the rendered state and every option's description, embedded
//! in one pass, and the head scores each (state, option) pair from
//! `[s, o, |s−o|, s∗o]`; a softmax over a question's options is its answer.
//!
//! Every text is one sequence `<s> prefix + text </s>`; all of a request's
//! texts are packed into one token stream and one command buffer, isolated
//! by the bidirectional attention mask (same sequence, and the local window
//! on ModernBERT's sliding layers). BERT is post-LayerNorm with absolute
//! positions and biases: the residual stream alternates between two buffers
//! so each LayerNorm writes the other one. ModernBERT is pre-LayerNorm with
//! RoPE, GeGLU and no biases: the Laya encoder's kernels (`laya.rs`), the
//! stream stays in one buffer. Mean pooling (L2-normalised) writes each text's
//! unit vector into a slot; the head reads (state slot, option slot) pairs
//! from there — its feature rows, two matmuls and the GELU are three more
//! dispatches of the same pass, so a request is one submit and one
//! read-back (the new vectors, for the cache, and the logits). Option
//! vectors are cached across requests (a question's criteria are fixed
//! text) and uploaded into their slots, so a typical request embeds only
//! its state.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use serde_json::Value;
use wgpu::util::DeviceExt;

use crate::engine::{
    attn_tile, attn_workgroup_bytes, div_ceil, open_device, report_profile, Dispatch, GpuTensor,
    Kernels, Params, Profiler, Step, GELU_ERF, K, PARAM_SLOT,
};
use crate::model::{Dtype, QTensor, TensorSpec};

pub mod decide;

#[cfg(feature = "native")]
pub mod backend;
#[cfg(feature = "native")]
pub use backend::E5Backend;

/// The tokenizer the engine needs: text → ids, no special tokens.
pub trait Tokenize {
    fn encode(&self, text: &str) -> Vec<u32>;
}

impl<F: Fn(&str) -> Vec<u32>> Tokenize for F {
    fn encode(&self, text: &str) -> Vec<u32> {
        self(text)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// Post-LN, absolute positions, token-type row, biases everywhere (e5).
    Bert,
    /// Pre-LN, RoPE with local / global layers, GeGLU, no biases (Ruri v3).
    ModernBert,
}

#[derive(Debug, Clone)]
pub struct E5Config {
    pub arch: Arch,
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub ff: usize,
    pub max_pos: usize,
    pub eps: f32,
    /// LayerNorms carry a bias (BERT yes, ModernBERT `norm_bias`).
    pub norm_bias: bool,
    /// ModernBERT: per layer, true = local (sliding) attention; the
    /// inclusive local distance (`local_attention / 2`); RoPE thetas.
    pub sliding: Vec<bool>,
    pub window: usize,
    pub theta_global: f32,
    pub theta_local: f32,
    /// Longest sequence, special tokens included (the model's 512).
    pub max_len: usize,
    /// Head: input width (4 d), hidden width, and the calibration
    /// temperature its logits are divided by.
    pub head_in: usize,
    pub head_hidden: usize,
    pub temperature: f32,
    /// Text put in front of every state and option (`query: `).
    pub prefix: String,
    pub cls: u32,
    pub sep: u32,
    pub pad: u32,
    pub name: String,
}

impl E5Config {
    /// An exported directory's `config.json` (tools/export_e5.py): the HF
    /// BERT config carrying the head and token ids under `grande_e5`.
    pub fn from_json(config: &Value) -> Result<Self> {
        let arch = match config["model_type"].as_str() {
            Some("bert") => Arch::Bert,
            Some("modernbert") => Arch::ModernBert,
            other => bail!("model_type {other:?} is not bert or modernbert"),
        };
        for key in ["hidden_act", "hidden_activation"] {
            if let Some(a) = config[key].as_str() {
                if a != "gelu" {
                    bail!("{key} {a} is not gelu");
                }
            }
        }
        if arch == Arch::Bert {
            if let Some(p) = config["position_embedding_type"].as_str() {
                if p != "absolute" {
                    bail!("position_embedding_type {p} is not absolute");
                }
            }
        }
        let n = |k: &str| -> Result<usize> {
            config[k]
                .as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("config.json: missing {k}"))
        };
        let g = config
            .get("grande_e5")
            .ok_or_else(|| anyhow!("config.json: missing grande_e5"))?;
        let gn = |k: &str| -> Result<u64> {
            g[k].as_u64()
                .ok_or_else(|| anyhow!("config.json: missing grande_e5.{k}"))
        };
        let d = n("hidden_size")?;
        let max_pos = n("max_position_embeddings")?;
        let layers = n("num_hidden_layers")?;
        let (sliding, window, theta_global, theta_local, eps, norm_bias) = match arch {
            Arch::Bert => (
                vec![false; layers],
                0,
                0.0,
                0.0,
                config["layer_norm_eps"].as_f64().unwrap_or(1e-12) as f32,
                true,
            ),
            Arch::ModernBert => {
                let every = config["global_attn_every_n_layers"].as_u64().unwrap_or(3) as usize;
                let sliding: Vec<bool> = match config["layer_types"].as_array() {
                    Some(a) => a
                        .iter()
                        .map(|t| t.as_str() == Some("sliding_attention"))
                        .collect(),
                    None => (0..layers).map(|i| i % every != 0).collect(),
                };
                if sliding.len() != layers {
                    bail!(
                        "layer_types has {} entries for {layers} layers",
                        sliding.len()
                    );
                }
                let rope = &config["rope_parameters"];
                let theta = |kind: &str, key: &str, fallback: f64| -> f32 {
                    rope[kind]["rope_theta"]
                        .as_f64()
                        .or_else(|| config[key].as_f64())
                        .unwrap_or(fallback) as f32
                };
                (
                    sliding,
                    config["local_attention"].as_u64().unwrap_or(128) as usize / 2,
                    theta("full_attention", "global_rope_theta", 160000.0),
                    theta("sliding_attention", "local_rope_theta", 10000.0),
                    config["norm_eps"]
                        .as_f64()
                        .or_else(|| config["layer_norm_eps"].as_f64())
                        .unwrap_or(1e-5) as f32,
                    config["norm_bias"].as_bool().unwrap_or(false),
                )
            }
        };
        let cfg = E5Config {
            arch,
            vocab: n("vocab_size")?,
            d,
            layers,
            heads: n("num_attention_heads")?,
            ff: n("intermediate_size")?,
            max_pos,
            eps,
            norm_bias,
            sliding,
            window,
            theta_global,
            theta_local,
            max_len: g["max_len"].as_u64().unwrap_or(max_pos.min(512) as u64) as usize,
            head_in: g["head_in"].as_u64().unwrap_or(4 * d as u64) as usize,
            head_hidden: gn("head_hidden")? as usize,
            temperature: g["temperature"].as_f64().unwrap_or(1.0) as f32,
            prefix: g["prefix"].as_str().unwrap_or("query: ").to_string(),
            cls: gn("cls")? as u32,
            sep: gn("sep")? as u32,
            pad: gn("pad")? as u32,
            name: g["name"].as_str().unwrap_or("e5").to_string(),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        self.d / self.heads
    }

    fn validate(&self) -> Result<()> {
        if self.heads == 0 || !self.d.is_multiple_of(self.heads) {
            bail!("hidden_size {} / heads {}", self.d, self.heads);
        }
        let hd = self.head_dim();
        let (_, kb) = attn_tile(hd);
        if !hd.is_multiple_of(4 * kb) || hd < 32 {
            bail!("head_dim {hd} must be a multiple of {}", 4 * kb);
        }
        if !self.d.is_multiple_of(32) || !self.ff.is_multiple_of(32) {
            bail!("hidden_size and intermediate_size must be multiples of 32");
        }
        if self.head_in != 4 * self.d {
            bail!("head_in {} is not 4 x hidden_size", self.head_in);
        }
        if self.max_len < 3 || self.max_len > self.max_pos {
            bail!("max_len {} outside 3..={}", self.max_len, self.max_pos);
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            bail!("temperature must be finite and positive");
        }
        Ok(())
    }

    /// Every tensor the engine needs. Linears are `[out, in]`; the head's
    /// feature z-scoring is folded into its first layer and 1/sqrt(head_dim)
    /// into the Q projection (tools/export_e5.py).
    pub fn tensors(&self) -> Vec<TensorSpec> {
        let d = self.d;
        let ff = self.ff;
        let mut v = vec![TensorSpec::new("embed", vec![self.vocab, d])];
        let norm = |v: &mut Vec<TensorSpec>, name: &str| {
            v.push(TensorSpec::new(&format!("{name}_w"), vec![d]));
            if self.norm_bias {
                v.push(TensorSpec::new(&format!("{name}_b"), vec![d]));
            }
        };
        match self.arch {
            Arch::Bert => {
                v.push(TensorSpec::new("pos_embed", vec![self.max_pos, d]));
                v.push(TensorSpec::new("type_embed", vec![d]));
                norm(&mut v, "embed_norm");
                for l in 0..self.layers {
                    let n = |s: &str| format!("enc.{l}.{s}");
                    v.push(TensorSpec::new(&n("qkv"), vec![3 * d, d]));
                    v.push(TensorSpec::new(&n("qkv_b"), vec![3 * d]));
                    v.push(TensorSpec::new(&n("o"), vec![d, d]));
                    v.push(TensorSpec::new(&n("o_b"), vec![d]));
                    norm(&mut v, &n("attn_norm"));
                    v.push(TensorSpec::new(&n("wi"), vec![ff, d]));
                    v.push(TensorSpec::new(&n("wi_b"), vec![ff]));
                    v.push(TensorSpec::new(&n("wo"), vec![d, ff]));
                    v.push(TensorSpec::new(&n("wo_b"), vec![d]));
                    norm(&mut v, &n("mlp_norm"));
                }
            }
            Arch::ModernBert => {
                norm(&mut v, "embed_norm");
                norm(&mut v, "final_norm");
                for l in 0..self.layers {
                    let n = |s: &str| format!("enc.{l}.{s}");
                    if l > 0 {
                        norm(&mut v, &n("attn_norm"));
                    }
                    v.push(TensorSpec::new(&n("qkv"), vec![3 * d, d]));
                    v.push(TensorSpec::new(&n("o"), vec![d, d]));
                    norm(&mut v, &n("mlp_norm"));
                    v.push(TensorSpec::new(&n("wi_val"), vec![ff, d]));
                    v.push(TensorSpec::new(&n("wi_gate"), vec![ff, d]));
                    v.push(TensorSpec::new(&n("wo"), vec![d, ff]));
                }
            }
        }
        v.push(TensorSpec::new(
            "head.l1",
            vec![self.head_hidden, self.head_in],
        ));
        v.push(TensorSpec::new("head.l1_b", vec![self.head_hidden]));
        v.push(TensorSpec::new("head.l2", vec![1, self.head_hidden]));
        v.push(TensorSpec::new("head.l2_b", vec![1]));
        v
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmbedParams {
    t: u32,
    d: u32,
    scale: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LnParams {
    t: u32,
    d: u32,
    eps: f32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatmulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BiasParams {
    t: u32,
    n: u32,
    mode: u32,
    has_bias: u32,
    in_place: u32,
    residual: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttnParams {
    t: u32,
    heads: u32,
    window: u32,
    stride: u32,
    k_off: u32,
    v_off: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RopeParams {
    t: u32,
    heads: u32,
    stride: u32,
    k_off: u32,
    theta: f32,
    scale: f32,
    _p0: u32,
    _p1: u32,
}

/// Pool (sequences, d) and pair-feature (pairs, d) kernels.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CountParams {
    n: u32,
    d: u32,
    _p0: u32,
    _p1: u32,
}

/// One encoder layer's dispatches.
enum EncLayer {
    /// The stream enters in `x`, the attention LayerNorm writes `h`, the
    /// MLP LayerNorm writes `x` again.
    Bert {
        mm_qkv: Step,
        bias_qkv: Step,
        attn: Step,
        mm_o: Step,
        add_o: Step,
        attn_norm: Step,
        mm_wi: Step,
        gelu_wi: Step,
        mm_wo: Step,
        add_wo: Step,
        mlp_norm: Step,
    },
    /// The stream stays in `x`; norms write `h` for the projections to read.
    /// `attn_norm` is None on layer 0 (the projection reads `x`).
    Modern {
        attn_norm: Option<Step>,
        mm_qkv: Step,
        rope: Step,
        attn: Step,
        mm_o: Step,
        add_o: Step,
        mlp_norm: Step,
        mm_wi: Step,
        mm_wo: Step,
        add_wo: Step,
        sliding: bool,
    },
}

/// The (state, option) head: feature rows, Linear, GELU, Linear.
struct HeadSteps {
    feats: Step,
    mm_l1: Step,
    gelu_l1: Step,
    mm_l2: Step,
    bias_l2: Step,
}

/// Receives the weights one tensor at a time, then wires the engine.
pub struct E5Builder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: E5Config,
    expected: HashMap<String, Vec<usize>>,
    weights: HashMap<String, GpuTensor>,
    profile: bool,
}

impl E5Builder {
    pub async fn new(config: E5Config) -> Result<Self> {
        config.validate()?;
        let need = (config.vocab * config.d * 2) as u64;
        let (device, queue, profile) =
            open_device(need, attn_workgroup_bytes(config.head_dim())).await?;
        let expected = config
            .tensors()
            .into_iter()
            .map(|s| (s.name, s.shape))
            .collect();
        Ok(E5Builder {
            device,
            queue,
            config,
            expected,
            weights: HashMap::new(),
            profile,
        })
    }

    pub fn config(&self) -> &E5Config {
        &self.config
    }

    pub fn missing(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .expected
            .keys()
            .filter(|n| !self.weights.contains_key(*n))
            .cloned()
            .collect();
        v.sort();
        v
    }

    pub fn push(&mut self, name: &str, t: &QTensor) -> Result<()> {
        let shape = self
            .expected
            .get(name)
            .ok_or_else(|| anyhow!("unexpected tensor {name}"))?;
        if &t.shape != shape {
            bail!("{name}: shape {:?}, expected {:?}", t.shape, shape);
        }
        if t.shape.len() == 1 && t.dtype != Dtype::F16 {
            bail!("{name}: 1-D tensors must be f16");
        }
        if t.shape.len() == 2 && !t.shape[1].is_multiple_of(32) {
            bail!(
                "{name}: inner dimension {} is not a multiple of 32",
                t.shape[1]
            );
        }
        let upload = |label: &str, bytes: &[u8]| {
            let mut padded = bytes.to_vec();
            while !padded.len().is_multiple_of(4) {
                padded.push(0);
            }
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &padded,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let data = upload(name, &t.data);
        let scales = (t.dtype != Dtype::F16).then(|| upload(&format!("{name}.scales"), &t.scales));
        self.weights.insert(
            name.to_string(),
            GpuTensor {
                dtype: t.dtype,
                data,
                scales,
            },
        );
        Ok(())
    }

    /// Allocate the workspace for `capacity` packed tokens and `max_seqs`
    /// texts (vector slots, and at most as many (state, option) pairs) per
    /// pass, and wire every dispatch.
    pub fn finish(self, capacity: usize, max_seqs: usize) -> Result<E5Engine> {
        let missing = self.missing();
        if !missing.is_empty() {
            bail!(
                "{} tensors missing, e.g. {}",
                missing.len(),
                missing
                    .iter()
                    .take(4)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let E5Builder {
            device,
            queue,
            config: cfg,
            weights,
            profile,
            ..
        } = self;
        let d = cfg.d;
        let hd = cfg.head_dim();
        let mut kernels = Kernels {
            device: device.clone(),
            map: HashMap::new(),
        };
        let f32_buf = |name: &str, n: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: (n.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let dummy = f32_buf("dummy", 4);
        let u32_buf = |name: &str, n: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: (n.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let ids = u32_buf("ids", capacity);
        let positions = u32_buf("positions", capacity);
        let meta = u32_buf("meta", capacity * 4);
        let spans = u32_buf("spans", max_seqs * 3);
        let pairs = u32_buf("pairs", max_seqs * 2);
        let x = f32_buf("x", capacity * d);
        let h = f32_buf("h", capacity * d);
        let qkv = f32_buf("qkv", capacity * 3 * d);
        let attn = f32_buf("attn", capacity * d);
        let a = f32_buf("a", capacity * d);
        let act = f32_buf("act", capacity * cfg.ff);
        let vectors = f32_buf("vectors", max_seqs * d);
        let feats = f32_buf("feats", max_seqs * cfg.head_in);
        let hact = f32_buf("hact", max_seqs * cfg.head_hidden);
        let logits = f32_buf("logits", max_seqs);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (max_seqs * (d + 1) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let param_slots = 12 + cfg.layers * 11;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: PARAM_SLOT * param_slots as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let w = |name: &str| -> Result<&GpuTensor> {
            weights
                .get(name)
                .ok_or_else(|| anyhow!("missing tensor {name}"))
        };
        fn sc<'b>(t: &'b GpuTensor, dummy: &'b wgpu::Buffer) -> &'b wgpu::Buffer {
            t.scales.as_ref().unwrap_or(dummy)
        }
        let mut step = |k: K, hd: usize, quant: u32, name: &str, bufs: &[&wgpu::Buffer]| -> Step {
            let kernel = kernels.get(k, hd, quant);
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &params,
                    offset: 0,
                    size: wgpu::BufferSize::new(PARAM_SLOT),
                }),
            }];
            for (i, b) in bufs.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: (i + 1) as u32,
                    resource: b.as_entire_binding(),
                });
            }
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(name),
                layout: &kernel.layout,
                entries: &entries,
            });
            Step {
                kernel: (
                    k.name(),
                    if k.uses_head_dim() { hd } else { 0 },
                    if k.reads_weights() { quant } else { 0 },
                ),
                bind,
            }
        };
        let embed_t = w("embed")?;
        let nb = |name: &str| -> Result<&wgpu::Buffer> {
            Ok(if cfg.norm_bias {
                &w(&format!("{name}_b"))?.data
            } else {
                &dummy
            })
        };
        // BERT: h = word[ids]; a = pos[positions]; h += a + type_embed[0]; x = LN(h).
        // ModernBERT: h = word[ids]; x = LN(h).
        let embed = step(
            K::Embed,
            0,
            embed_t.dtype.code(),
            "embed",
            &[&ids, &embed_t.data, sc(embed_t, &dummy), &h],
        );
        let (pos_embed, embed_add) = match cfg.arch {
            Arch::Bert => {
                let pos_t = w("pos_embed")?;
                (
                    Some(step(
                        K::Embed,
                        0,
                        pos_t.dtype.code(),
                        "pos_embed",
                        &[&positions, &pos_t.data, sc(pos_t, &dummy), &a],
                    )),
                    Some(step(
                        K::BiasAct,
                        0,
                        0,
                        "embed_add",
                        &[&a, &w("type_embed")?.data, &h],
                    )),
                )
            }
            Arch::ModernBert => (None, None),
        };
        let embed_norm = step(
            K::LayerNorm,
            0,
            0,
            "embed_norm",
            &[&h, &w("embed_norm_w")?.data, nb("embed_norm")?, &x],
        );
        let mut enc = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let n = |s: &str| format!("enc.{l}.{s}");
            let t_qkv = w(&n("qkv"))?;
            let t_o = w(&n("o"))?;
            let t_wo = w(&n("wo"))?;
            match cfg.arch {
                Arch::Bert => {
                    let t_wi = w(&n("wi"))?;
                    enc.push(EncLayer::Bert {
                        mm_qkv: step(
                            K::Matmul,
                            0,
                            t_qkv.dtype.code(),
                            "mm_qkv",
                            &[&x, &t_qkv.data, sc(t_qkv, &dummy), &qkv],
                        ),
                        bias_qkv: step(
                            K::BiasAct,
                            0,
                            0,
                            "bias_qkv",
                            &[&dummy, &w(&n("qkv_b"))?.data, &qkv],
                        ),
                        attn: step(K::AttentionBi, hd, 0, "attention", &[&qkv, &meta, &attn]),
                        mm_o: step(
                            K::Matmul,
                            0,
                            t_o.dtype.code(),
                            "mm_o",
                            &[&attn, &t_o.data, sc(t_o, &dummy), &a],
                        ),
                        add_o: step(K::BiasAct, 0, 0, "add_o", &[&a, &w(&n("o_b"))?.data, &x]),
                        attn_norm: step(
                            K::LayerNorm,
                            0,
                            0,
                            "attn_norm",
                            &[&x, &w(&n("attn_norm_w"))?.data, nb(&n("attn_norm"))?, &h],
                        ),
                        mm_wi: step(
                            K::Matmul,
                            0,
                            t_wi.dtype.code(),
                            "mm_wi",
                            &[&h, &t_wi.data, sc(t_wi, &dummy), &act],
                        ),
                        gelu_wi: step(
                            K::BiasAct,
                            0,
                            0,
                            "gelu_wi",
                            &[&dummy, &w(&n("wi_b"))?.data, &act],
                        ),
                        mm_wo: step(
                            K::Matmul,
                            0,
                            t_wo.dtype.code(),
                            "mm_wo",
                            &[&act, &t_wo.data, sc(t_wo, &dummy), &a],
                        ),
                        add_wo: step(K::BiasAct, 0, 0, "add_wo", &[&a, &w(&n("wo_b"))?.data, &h]),
                        mlp_norm: step(
                            K::LayerNorm,
                            0,
                            0,
                            "mlp_norm",
                            &[&h, &w(&n("mlp_norm_w"))?.data, nb(&n("mlp_norm"))?, &x],
                        ),
                    });
                }
                Arch::ModernBert => {
                    let t_val = w(&n("wi_val"))?;
                    let t_gate = w(&n("wi_gate"))?;
                    if t_val.dtype != t_gate.dtype {
                        bail!("layer {l}: Wi halves must share a storage type");
                    }
                    let attn_norm = if l > 0 {
                        Some(step(
                            K::LayerNorm,
                            0,
                            0,
                            "attn_norm",
                            &[&x, &w(&n("attn_norm_w"))?.data, nb(&n("attn_norm"))?, &h],
                        ))
                    } else {
                        None
                    };
                    let qkv_src = if l > 0 { &h } else { &x };
                    enc.push(EncLayer::Modern {
                        attn_norm,
                        mm_qkv: step(
                            K::Matmul,
                            0,
                            t_qkv.dtype.code(),
                            "mm_qkv",
                            &[qkv_src, &t_qkv.data, sc(t_qkv, &dummy), &qkv],
                        ),
                        rope: step(K::RopeBi, hd, 0, "rope", &[&meta, &qkv]),
                        attn: step(K::AttentionBi, hd, 0, "attention", &[&qkv, &meta, &attn]),
                        mm_o: step(
                            K::Matmul,
                            0,
                            t_o.dtype.code(),
                            "mm_o",
                            &[&attn, &t_o.data, sc(t_o, &dummy), &a],
                        ),
                        add_o: step(K::BiasAct, 0, 0, "add_o", &[&a, &dummy, &x]),
                        mlp_norm: step(
                            K::LayerNorm,
                            0,
                            0,
                            "mlp_norm",
                            &[&x, &w(&n("mlp_norm_w"))?.data, nb(&n("mlp_norm"))?, &h],
                        ),
                        mm_wi: step(
                            K::MatmulGated,
                            0,
                            t_val.dtype.code() | GELU_ERF,
                            "mm_wi",
                            &[
                                &h,
                                &t_val.data,
                                sc(t_val, &dummy),
                                &t_gate.data,
                                sc(t_gate, &dummy),
                                &act,
                            ],
                        ),
                        mm_wo: step(
                            K::Matmul,
                            0,
                            t_wo.dtype.code(),
                            "mm_wo",
                            &[&act, &t_wo.data, sc(t_wo, &dummy), &a],
                        ),
                        add_wo: step(K::BiasAct, 0, 0, "add_wo", &[&a, &dummy, &x]),
                        sliding: cfg.sliding[l],
                    });
                }
            }
        }
        // ModernBERT ends with a LayerNorm (into h); the pool reads it. BERT's
        // last MLP LayerNorm already wrote x.
        let final_norm = match cfg.arch {
            Arch::Bert => None,
            Arch::ModernBert => Some(step(
                K::LayerNorm,
                0,
                0,
                "final_norm",
                &[&x, &w("final_norm_w")?.data, nb("final_norm")?, &h],
            )),
        };
        let pool_src = if final_norm.is_some() { &h } else { &x };
        let pool = step(K::Pool, 0, 0, "pool", &[&spans, pool_src, &vectors]);
        let t_l1 = w("head.l1")?;
        let t_l2 = w("head.l2")?;
        let head = HeadSteps {
            feats: step(
                K::PairFeats,
                0,
                0,
                "pair_feats",
                &[&pairs, &vectors, &feats],
            ),
            mm_l1: step(
                K::Matmul,
                0,
                t_l1.dtype.code(),
                "head_l1",
                &[&feats, &t_l1.data, sc(t_l1, &dummy), &hact],
            ),
            gelu_l1: step(
                K::BiasAct,
                0,
                0,
                "head_gelu",
                &[&dummy, &w("head.l1_b")?.data, &hact],
            ),
            mm_l2: step(
                K::Matmul,
                0,
                t_l2.dtype.code(),
                "head_l2",
                &[&hact, &t_l2.data, sc(t_l2, &dummy), &logits],
            ),
            bias_l2: step(
                K::BiasAct,
                0,
                0,
                "head_bias",
                &[&dummy, &w("head.l2_b")?.data, &logits],
            ),
        };
        let profile = profile.then(|| Profiler::new(&device, &queue, 2 * param_slots as u32));
        Ok(E5Engine {
            device,
            queue,
            config: cfg,
            capacity,
            max_seqs,
            kernels,
            params,
            param_slots,
            ids,
            positions,
            meta,
            spans,
            pairs,
            vectors,
            logits,
            staging,
            embed,
            pos_embed,
            embed_add,
            embed_norm,
            enc,
            final_norm,
            pool,
            head,
            cache: Mutex::new(HashMap::new()),
            profile,
            _keep: vec![x, h, qkv, attn, a, act, feats, hact, dummy],
            _weights: weights.into_values().collect(),
        })
    }
}

/// Most option vectors kept across requests before the cache is cleared.
const CACHE_MAX: usize = 8192;

pub struct E5Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub config: E5Config,
    pub capacity: usize,
    pub max_seqs: usize,
    kernels: Kernels,
    params: wgpu::Buffer,
    param_slots: usize,
    ids: wgpu::Buffer,
    positions: wgpu::Buffer,
    meta: wgpu::Buffer,
    spans: wgpu::Buffer,
    pairs: wgpu::Buffer,
    vectors: wgpu::Buffer,
    logits: wgpu::Buffer,
    staging: wgpu::Buffer,
    embed: Step,
    pos_embed: Option<Step>,
    embed_add: Option<Step>,
    embed_norm: Step,
    enc: Vec<EncLayer>,
    final_norm: Option<Step>,
    pool: Step,
    head: HeadSteps,
    /// Option text → (unit vector, token count), kept across requests.
    cache: Mutex<HashMap<String, (Vec<f32>, usize)>>,
    /// GPU time per dispatch (GRANDE_WGPU_PROFILE, native only).
    profile: Option<Profiler>,
    _keep: Vec<wgpu::Buffer>,
    _weights: Vec<GpuTensor>,
}

impl E5Engine {
    /// `<s> prefix + text </s>`, truncated to `max_len` like the HF
    /// tokenizer (`truncation=True`).
    pub fn sequence(&self, tok: &dyn Tokenize, text: &str) -> Vec<u32> {
        let cfg = &self.config;
        let mut body = tok.encode(&format!("{}{text}", cfg.prefix));
        body.truncate(cfg.max_len - 2);
        let mut ids = Vec::with_capacity(body.len() + 2);
        ids.push(cfg.cls);
        ids.extend(body);
        ids.push(cfg.sep);
        ids
    }

    /// Run a short sequence so the first real request does not pay for the
    /// workspace's first touch.
    pub async fn warmup(&self) -> Result<()> {
        let seq = vec![self.config.cls, self.config.sep];
        self.run(&[seq.clone(), seq], &[0, 1], &[], &[(0, 1)])
            .await?;
        Ok(())
    }

    /// Unit vectors (masked mean over every token, L2-normalised) for a
    /// batch of token sequences, one pass.
    pub async fn embed(&self, seqs: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        let slots: Vec<u32> = (0..seqs.len() as u32).collect();
        Ok(self.run(seqs, &slots, &[], &[]).await?.0)
    }

    /// One pass: encode `seqs` into vector slots `slots[i]`, after writing
    /// `uploaded` (slot, unit vector) pairs from the host, then score every
    /// (state slot, option slot) of `pairs`. Returns the new sequences'
    /// unit vectors (in order) and one logit per pair.
    pub async fn run<'s>(
        &'s self,
        seqs: &[Vec<u32>],
        slots: &[u32],
        uploaded: &[(u32, &[f32])],
        pairs: &[(u32, u32)],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let cfg = &self.config;
        let d = cfg.d;
        let hd = cfg.head_dim();
        let total: usize = seqs.iter().map(|s| s.len()).sum();
        if total > self.capacity {
            bail!(
                "{total} tokens exceed the engine capacity {}",
                self.capacity
            );
        }
        if seqs.len() != slots.len() {
            bail!("{} sequences for {} slots", seqs.len(), slots.len());
        }
        let used = slots
            .iter()
            .chain(uploaded.iter().map(|(s, _)| s))
            .chain(pairs.iter().flat_map(|(a, b)| [a, b]))
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1);
        if used > self.max_seqs || pairs.len() > self.max_seqs {
            bail!(
                "{used} vector slots / {} pairs exceed the engine's {} per pass",
                pairs.len(),
                self.max_seqs
            );
        }
        let mut ids: Vec<u32> = Vec::with_capacity(total);
        let mut positions: Vec<u32> = Vec::with_capacity(total);
        let mut meta: Vec<i32> = Vec::with_capacity(total * 4);
        let mut spans: Vec<u32> = Vec::with_capacity(seqs.len() * 3);
        for (si, s) in seqs.iter().enumerate() {
            if s.is_empty() {
                bail!("sequence {si} is empty");
            }
            if s.len() > cfg.max_len {
                bail!(
                    "sequence {si}: {} tokens > max_len {}",
                    s.len(),
                    cfg.max_len
                );
            }
            let start = ids.len();
            for (i, &t) in s.iter().enumerate() {
                if t as usize >= cfg.vocab {
                    bail!("token id {t} outside the vocabulary of {}", cfg.vocab);
                }
                ids.push(t);
                positions.push(i as u32);
                meta.push(i as i32);
                meta.push(si as i32);
                meta.push(start as i32);
                meta.push((start + s.len()) as i32);
            }
            spans.push(start as u32);
            spans.push((start + s.len()) as u32);
            spans.push(slots[si]);
        }
        let t = total;
        if t > 0 {
            self.queue
                .write_buffer(&self.ids, 0, bytemuck::cast_slice(&ids));
            self.queue
                .write_buffer(&self.positions, 0, bytemuck::cast_slice(&positions));
            self.queue
                .write_buffer(&self.meta, 0, bytemuck::cast_slice(&meta));
            self.queue
                .write_buffer(&self.spans, 0, bytemuck::cast_slice(&spans));
        }
        for (slot, v) in uploaded {
            if v.len() != d {
                bail!(
                    "uploaded vector for slot {slot} has {} values, not {d}",
                    v.len()
                );
            }
            self.queue.write_buffer(
                &self.vectors,
                (*slot as usize * d * 4) as u64,
                bytemuck::cast_slice(v),
            );
        }
        if !pairs.is_empty() {
            let flat: Vec<u32> = pairs.iter().flat_map(|(a, b)| [*a, *b]).collect();
            self.queue
                .write_buffer(&self.pairs, 0, bytemuck::cast_slice(&flat));
        }

        let mut params = Params {
            bytes: Vec::with_capacity(self.param_slots * PARAM_SLOT as usize),
        };
        let mut plan: Vec<Dispatch<'s>> = Vec::with_capacity(self.param_slots);
        {
            let mut run = |name: &'static str, step: &'s Step, offset: u32, wg: (u32, u32)| {
                plan.push(Dispatch {
                    name,
                    step,
                    offset,
                    wg,
                });
            };
            // (mode, has_bias, in_place, residual) over a [rows, n] activation
            let bias = |rows: usize,
                        n: usize,
                        mode: u32,
                        has_bias: bool,
                        in_place: bool,
                        residual: bool| BiasParams {
                t: rows as u32,
                n: n as u32,
                mode,
                has_bias: has_bias as u32,
                in_place: in_place as u32,
                residual: residual as u32,
                _p0: 0,
                _p1: 0,
            };
            let mm = |rows: usize, n: usize, k: usize| MatmulParams {
                m: rows as u32,
                n: n as u32,
                k: k as u32,
                _pad: 0,
            };
            let mm_wg = |rows: usize, n: usize| (div_ceil(n, 128), div_ceil(rows, 32));
            let ew = |rows: usize, n: usize| (div_ceil(rows * n, 256), 1);
            if t > 0 {
                let ln = LnParams {
                    t: t as u32,
                    d: d as u32,
                    eps: cfg.eps,
                    has_bias: cfg.norm_bias as u32,
                };
                let (rows_per_wg, _) = attn_tile(hd);
                let attn_wg = (div_ceil(t, rows_per_wg), cfg.heads as u32);
                let attn_p = |window: usize| AttnParams {
                    t: t as u32,
                    heads: cfg.heads as u32,
                    window: window as u32,
                    stride: 3 * d as u32,
                    k_off: d as u32,
                    v_off: 2 * d as u32,
                    _p1: 0,
                    _p2: 0,
                };
                let embed_p = EmbedParams {
                    t: t as u32,
                    d: d as u32,
                    scale: 1.0,
                    _pad: 0,
                };
                let off = params.push(embed_p);
                run("embed", &self.embed, off, (div_ceil(t * d / 4, 256), 1));
                if let (Some(pe), Some(ea)) = (&self.pos_embed, &self.embed_add) {
                    let off = params.push(embed_p);
                    run("pos_embed", pe, off, (div_ceil(t * d / 4, 256), 1));
                    let off = params.push(bias(t, d, 0, true, false, true));
                    run("embed_add", ea, off, ew(t, d));
                }
                let off = params.push(ln);
                run("embed_norm", &self.embed_norm, off, (t as u32, 1));
                for l in &self.enc {
                    match l {
                        EncLayer::Bert {
                            mm_qkv,
                            bias_qkv,
                            attn,
                            mm_o,
                            add_o,
                            attn_norm,
                            mm_wi,
                            gelu_wi,
                            mm_wo,
                            add_wo,
                            mlp_norm,
                        } => {
                            let off = params.push(mm(t, 3 * d, d));
                            run("mm_qkv", mm_qkv, off, mm_wg(t, 3 * d));
                            let off = params.push(bias(t, 3 * d, 0, true, true, false));
                            run("bias_qkv", bias_qkv, off, ew(t, 3 * d));
                            let off = params.push(attn_p(0));
                            run("attention", attn, off, attn_wg);
                            let off = params.push(mm(t, d, d));
                            run("mm_o", mm_o, off, mm_wg(t, d));
                            let off = params.push(bias(t, d, 0, true, false, true));
                            run("add_o", add_o, off, ew(t, d));
                            let off = params.push(ln);
                            run("attn_norm", attn_norm, off, (t as u32, 1));
                            let off = params.push(mm(t, cfg.ff, d));
                            run("mm_wi", mm_wi, off, mm_wg(t, cfg.ff));
                            let off = params.push(bias(t, cfg.ff, 2, true, true, false));
                            run("gelu_wi", gelu_wi, off, ew(t, cfg.ff));
                            let off = params.push(mm(t, d, cfg.ff));
                            run("mm_wo", mm_wo, off, mm_wg(t, d));
                            let off = params.push(bias(t, d, 0, true, false, true));
                            run("add_wo", add_wo, off, ew(t, d));
                            let off = params.push(ln);
                            run("mlp_norm", mlp_norm, off, (t as u32, 1));
                        }
                        EncLayer::Modern {
                            attn_norm,
                            mm_qkv,
                            rope,
                            attn,
                            mm_o,
                            add_o,
                            mlp_norm,
                            mm_wi,
                            mm_wo,
                            add_wo,
                            sliding,
                        } => {
                            if let Some(s) = attn_norm {
                                let off = params.push(ln);
                                run("attn_norm", s, off, (t as u32, 1));
                            }
                            let off = params.push(mm(t, 3 * d, d));
                            run("mm_qkv", mm_qkv, off, mm_wg(t, 3 * d));
                            let off = params.push(RopeParams {
                                t: t as u32,
                                heads: cfg.heads as u32,
                                stride: 3 * d as u32,
                                k_off: d as u32,
                                theta: if *sliding {
                                    cfg.theta_local
                                } else {
                                    cfg.theta_global
                                },
                                scale: (hd as f32).powf(-0.5),
                                _p0: 0,
                                _p1: 0,
                            });
                            run(
                                "rope",
                                rope,
                                off,
                                (div_ceil(t * cfg.heads * hd / 2, 256), 1),
                            );
                            let off = params.push(attn_p(if *sliding { cfg.window } else { 0 }));
                            run("attention", attn, off, attn_wg);
                            let off = params.push(mm(t, d, d));
                            run("mm_o", mm_o, off, mm_wg(t, d));
                            let off = params.push(bias(t, d, 0, false, false, true));
                            run("add_o", add_o, off, ew(t, d));
                            let off = params.push(ln);
                            run("mlp_norm", mlp_norm, off, (t as u32, 1));
                            let off = params.push(mm(t, cfg.ff, d));
                            run("mm_wi", mm_wi, off, mm_wg(t, cfg.ff));
                            let off = params.push(mm(t, d, cfg.ff));
                            run("mm_wo", mm_wo, off, mm_wg(t, d));
                            let off = params.push(bias(t, d, 0, false, false, true));
                            run("add_wo", add_wo, off, ew(t, d));
                        }
                    }
                }
                if let Some(s) = &self.final_norm {
                    let off = params.push(ln);
                    run("final_norm", s, off, (t as u32, 1));
                }
                let off = params.push(CountParams {
                    n: seqs.len() as u32,
                    d: d as u32,
                    _p0: 0,
                    _p1: 0,
                });
                run("pool", &self.pool, off, (seqs.len() as u32, 1));
            }
            let np = pairs.len();
            if np > 0 {
                let hh = cfg.head_hidden;
                let off = params.push(CountParams {
                    n: np as u32,
                    d: d as u32,
                    _p0: 0,
                    _p1: 0,
                });
                run("pair_feats", &self.head.feats, off, ew(np, d));
                let off = params.push(mm(np, hh, cfg.head_in));
                run("head_l1", &self.head.mm_l1, off, mm_wg(np, hh));
                let off = params.push(bias(np, hh, 2, true, true, false));
                run("head_gelu", &self.head.gelu_l1, off, ew(np, hh));
                let off = params.push(mm(np, 1, hh));
                run("head_l2", &self.head.mm_l2, off, mm_wg(np, 1));
                let off = params.push(bias(np, 1, 0, true, true, false));
                run("head_bias", &self.head.bias_l2, off, ew(np, 1));
            }
        }
        if params.bytes.len() > self.param_slots * PARAM_SLOT as usize {
            bail!("parameter slots exhausted");
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("e5") });
        match &self.profile {
            None => {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("forward"),
                    timestamp_writes: None,
                });
                for dsp in &plan {
                    pass.set_pipeline(&self.kernels.map[&dsp.step.kernel].pipeline);
                    pass.set_bind_group(0, &dsp.step.bind, &[dsp.offset]);
                    pass.dispatch_workgroups(dsp.wg.0, dsp.wg.1, 1);
                }
            }
            Some(prof) => {
                if plan.len() as u32 * 2 > prof.capacity {
                    bail!("too many dispatches to profile");
                }
                for (i, dsp) in plan.iter().enumerate() {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(dsp.name),
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &prof.queries,
                            beginning_of_pass_write_index: Some(2 * i as u32),
                            end_of_pass_write_index: Some(2 * i as u32 + 1),
                        }),
                    });
                    pass.set_pipeline(&self.kernels.map[&dsp.step.kernel].pipeline);
                    pass.set_bind_group(0, &dsp.step.bind, &[dsp.offset]);
                    pass.dispatch_workgroups(dsp.wg.0, dsp.wg.1, 1);
                }
                let n = 2 * plan.len() as u32;
                enc.resolve_query_set(&prof.queries, 0..n, &prof.resolve, 0);
                enc.copy_buffer_to_buffer(&prof.resolve, 0, &prof.readback, 0, n as u64 * 8);
            }
        }
        // Read back: the new sequences' vectors (from their slots), then the logits.
        let row_bytes = (d * 4) as u64;
        for (i, &slot) in slots.iter().enumerate() {
            enc.copy_buffer_to_buffer(
                &self.vectors,
                slot as u64 * row_bytes,
                &self.staging,
                i as u64 * row_bytes,
                row_bytes,
            );
        }
        let vec_bytes = seqs.len() as u64 * row_bytes;
        let logit_bytes = (pairs.len() * 4) as u64;
        if logit_bytes > 0 {
            enc.copy_buffer_to_buffer(&self.logits, 0, &self.staging, vec_bytes, logit_bytes);
        }
        let out_bytes = vec_bytes + logit_bytes;
        if !params.bytes.is_empty() {
            self.queue.write_buffer(&self.params, 0, &params.bytes);
        }
        self.queue.submit(Some(enc.finish()));
        if out_bytes == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let slice = self.staging.slice(..out_bytes);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow!("device poll: {e:?}"))?;
        rx.await
            .context("map callback dropped")?
            .map_err(|e| anyhow!("map_async: {e:?}"))?;
        let data: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        self.staging.unmap();
        if let Some(prof) = &self.profile {
            report_profile(&self.device, prof, &plan, plan.len()).await?;
        }
        let nv = seqs.len() * d;
        let vecs = data[..nv].chunks(d).map(|v| v.to_vec()).collect();
        Ok((vecs, data[nv..].to_vec()))
    }

    /// Score every (state text, option text) pair of `pairs` (indices into
    /// `texts`): texts not in the option cache go through the encoder, the
    /// rest are uploaded into their slots, and the head runs on the GPU in
    /// the same pass. `cacheable[i]` says whether text i may be kept.
    /// Returns one raw logit per pair and each text's token count.
    pub async fn evaluate(
        &self,
        tok: &dyn Tokenize,
        texts: &[String],
        cacheable: &[bool],
        pairs: &[(usize, usize)],
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        if texts.len() > self.max_seqs || pairs.len() > self.max_seqs {
            bail!(
                "{} texts / {} pairs exceed the engine's {} per pass",
                texts.len(),
                pairs.len(),
                self.max_seqs
            );
        }
        // Slot = text index. Cached vectors are uploaded; identical texts
        // within the request are encoded once and copied.
        let mut tokens = vec![0usize; texts.len()];
        let mut cached: Vec<(u32, Vec<f32>)> = Vec::new();
        let mut first: HashMap<&str, usize> = HashMap::new();
        let mut unique: Vec<usize> = Vec::new();
        let mut alias: Vec<(usize, usize)> = Vec::new();
        {
            let cache = self.cache.lock().map_err(|_| anyhow!("cache poisoned"))?;
            for (i, t) in texts.iter().enumerate() {
                if let Some((v, n)) = cache.get(t) {
                    cached.push((i as u32, v.clone()));
                    tokens[i] = *n;
                } else if let Some(&j) = first.get(t.as_str()) {
                    alias.push((i, j));
                } else {
                    first.insert(t.as_str(), i);
                    unique.push(i);
                }
            }
        }
        let seqs: Vec<Vec<u32>> = unique
            .iter()
            .map(|&i| self.sequence(tok, &texts[i]))
            .collect();
        let total: usize = seqs.iter().map(|s| s.len()).sum();
        let mut fresh: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut logits = Vec::new();
        let slots: Vec<u32> = unique.iter().map(|&i| i as u32).collect();
        let up: Vec<(u32, &[f32])> = cached.iter().map(|(s, v)| (*s, v.as_slice())).collect();
        let pr: Vec<(u32, u32)> = pairs.iter().map(|&(a, b)| (a as u32, b as u32)).collect();
        if total <= self.capacity && alias.is_empty() {
            // The common case: everything in one pass.
            let (vecs, z) = self.run(&seqs, &slots, &up, &pr).await?;
            for (k, v) in vecs.into_iter().enumerate() {
                fresh.push((unique[k], v));
            }
            logits = z;
        } else {
            // Encode in as many passes as the capacity needs, then one pass
            // with every vector uploaded for the head.
            let mut start = 0;
            while start < seqs.len() {
                let mut end = start;
                let mut n = 0;
                while end < seqs.len() && (end == start || n + seqs[end].len() <= self.capacity) {
                    n += seqs[end].len();
                    end += 1;
                }
                let vecs = self.embed(&seqs[start..end]).await?;
                for (k, v) in vecs.into_iter().enumerate() {
                    fresh.push((unique[start + k], v));
                }
                start = end;
            }
            let by_index: HashMap<usize, &Vec<f32>> = fresh.iter().map(|(i, v)| (*i, v)).collect();
            let mut all: Vec<(u32, &[f32])> = up.clone();
            for (i, v) in &fresh {
                all.push((*i as u32, v.as_slice()));
            }
            for &(i, j) in &alias {
                all.push((i as u32, by_index[&j].as_slice()));
            }
            if !pr.is_empty() {
                logits = self.run(&[], &[], &all, &pr).await?.1;
            }
        }
        for (k, &i) in unique.iter().enumerate() {
            tokens[i] = seqs[k].len();
        }
        for &(i, j) in &alias {
            tokens[i] = tokens[j];
        }
        {
            let mut cache = self.cache.lock().map_err(|_| anyhow!("cache poisoned"))?;
            if cache.len() + fresh.len() > CACHE_MAX {
                cache.clear();
            }
            for (i, v) in fresh {
                if cacheable[i] {
                    cache.insert(texts[i].clone(), (v, tokens[i]));
                }
            }
        }
        if logits.len() != pairs.len() {
            bail!("{} logits for {} pairs", logits.len(), pairs.len());
        }
        Ok((logits, tokens))
    }
}

/// Manifest of an exported directory (`tools/export_e5.py`): the same
/// layout as [`crate::model::Manifest`].
pub type Manifest = crate::model::Manifest;
