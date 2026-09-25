//! Laya on wgpu: a ModernBERT / mmBERT encoder with Convai's decision head
//! (<https://github.com/NandhaKishorM/laya>, Apache-2.0), the other way of
//! building a System One model — a bidirectional encoder reads one sequence
//! per question and a scorer reads the hidden row at each option's `[MASK]`.
//!
//! ```text
//! [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
//! ```
//!
//! Every question of a request is one sequence; all of them are packed into
//! one token stream and one command buffer, isolated by the attention mask
//! (same sequence only, plus the 64-token local window on two layers in
//! three). The encoder and the two head layers run on the GPU; the scorer
//! (LayerNorm → Linear → GELU → Linear(1) on each marker row) and the
//! `act_head` (an act / escalate probability from the `[CLS]` row and the
//! answer's confidence features) are a few hundred KFLOPs and run on the
//! host in f32. Numbers match laya-mlx (fp16) to ~1e-2 in the logits.
//!
//! `prompt` mirrors laya's `build_sequence` token for token, including its
//! truncation budgets, so a checkpoint answers the same here as upstream.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use serde_json::Value;
use wgpu::util::DeviceExt;

use crate::engine::{
    attn_tile_bi, attn_workgroup_bytes, div_ceil, open_device, report_profile, Dispatch, GpuTensor,
    Kernels, Params, Profiler, Step, GELU_ERF, K, PARAM_SLOT,
};
use crate::model::{Dtype, QTensor, SafeTensors, TensorSpec};

pub mod decide;
pub mod prompt;

#[cfg(feature = "native")]
pub mod backend;
#[cfg(feature = "native")]
pub use backend::LayaBackend;

/// Question types as the checkpoint numbers them (`type_emb` rows,
/// `temperature` entries).
pub const QTYPES: [&str; 3] = ["choice", "score", "noul"];

#[derive(Debug, Clone)]
pub struct LayaConfig {
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub ff: usize,
    /// Per encoder layer: true = local (sliding) attention.
    pub sliding: Vec<bool>,
    /// Inclusive local distance: `local_attention / 2`.
    pub window: usize,
    pub theta_global: f32,
    pub theta_local: f32,
    pub eps: f32,
    pub head_layers: usize,
    /// A question-type embedding is added after the encoder (Laya's
    /// `type_emb`). Off for the Ruri cross-encoder (docs/cross.md), which
    /// has neither it nor decision-head layers nor an act head.
    pub type_emb: bool,
    /// Decision-head MLP width (4 × d) and the act head's hidden / output
    /// widths; `act_out` 0 = no act head.
    pub act_hidden: usize,
    pub act_out: usize,
    /// Prompt budgets (`rl_agent_config.json`).
    pub max_len: usize,
    pub head_max_len: usize,
    /// Calibration: one temperature per question type, overridden per
    /// `(type, option-count bucket)` when present.
    pub temperature: [f32; 3],
    pub temperature_by_options: HashMap<String, f32>,
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
    pub mask_text: String,
}

impl LayaConfig {
    /// `encoder` is `encoder/config.json`, `agent` is `rl_agent_config.json`,
    /// `tokenizer_config` the tokenizer's config (for the special tokens'
    /// surface forms) and `special` resolves a surface form to its id.
    pub fn from_json(
        encoder: &Value,
        agent: &Value,
        tokenizer_config: &Value,
        special: impl Fn(&str) -> Option<u32>,
    ) -> Result<Self> {
        if encoder["model_type"].as_str() != Some("modernbert") {
            bail!(
                "encoder model_type {:?} is not modernbert",
                encoder["model_type"]
            );
        }
        if let Some(a) = encoder["hidden_activation"].as_str() {
            if a != "gelu" {
                bail!("encoder hidden_activation {a} is not gelu");
            }
        }
        let n = |k: &str| -> Result<usize> {
            encoder[k]
                .as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("encoder/config.json: missing {k}"))
        };
        let layers = n("num_hidden_layers")?;
        let every = encoder["global_attn_every_n_layers"].as_u64().unwrap_or(3) as usize;
        let sliding: Vec<bool> = match encoder["layer_types"].as_array() {
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
        let rope = &encoder["rope_parameters"];
        let theta = |kind: &str, key: &str, fallback: f64| -> f32 {
            rope[kind]["rope_theta"]
                .as_f64()
                .or_else(|| encoder[key].as_f64())
                .unwrap_or(fallback) as f32
        };
        let tok = |name: &str| -> Result<(String, u32)> {
            let v = &tokenizer_config[name];
            let text = v
                .as_str()
                .or_else(|| v["content"].as_str())
                .ok_or_else(|| anyhow!("tokenizer_config.json: missing {name}"))?;
            let id = special(text).ok_or_else(|| anyhow!("tokenizer has no {text}"))?;
            Ok((text.to_string(), id))
        };
        let (mask_text, mask) = tok("mask_token")?;
        let mut temperature = [1.0f32; 3];
        if let Some(t) = agent["temperature"].as_array() {
            if t.len() != 3 {
                bail!("rl_agent_config.json: temperature has {} entries", t.len());
            }
            for (i, v) in t.iter().enumerate() {
                temperature[i] = v.as_f64().unwrap_or(1.0) as f32;
            }
        }
        let temperature_by_options: HashMap<String, f32> = agent["temperature_by_options"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(1.0) as f32))
                    .collect()
            })
            .unwrap_or_default();
        for t in temperature.iter().chain(temperature_by_options.values()) {
            if !t.is_finite() || *t <= 0.0 {
                bail!("calibration temperatures must be finite and positive");
            }
        }
        let max_len = agent["max_len"].as_u64().unwrap_or(512) as usize;
        let head_max_len = agent["head_max_len"].as_u64().unwrap_or(192) as usize;
        if !(4 < head_max_len && head_max_len < max_len) {
            bail!("expected 4 < head_max_len {head_max_len} < max_len {max_len}");
        }
        let d = n("hidden_size")?;
        let heads = n("num_attention_heads")?;
        let cfg = LayaConfig {
            vocab: n("vocab_size")?,
            d,
            layers,
            heads,
            ff: n("intermediate_size")?,
            sliding,
            window: encoder["local_attention"].as_u64().unwrap_or(128) as usize / 2,
            theta_global: theta("full_attention", "global_rope_theta", 160000.0),
            theta_local: theta("sliding_attention", "local_rope_theta", 10000.0),
            eps: encoder["norm_eps"]
                .as_f64()
                .or_else(|| encoder["layer_norm_eps"].as_f64())
                .unwrap_or(1e-5) as f32,
            head_layers: agent["head_layers"].as_u64().unwrap_or(2) as usize,
            type_emb: agent["type_emb"].as_bool().unwrap_or(true),
            act_hidden: 256,
            act_out: if agent["act_head"].as_bool() == Some(false) {
                0
            } else {
                agent["act_costs"].as_object().map_or(0, |m| m.len()) + 1
            },
            max_len,
            head_max_len,
            temperature,
            temperature_by_options,
            cls: tok("cls_token")?.1,
            sep: tok("sep_token")?.1,
            mask,
            pad: tok("pad_token")?.1,
            mask_text,
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
        let (_, kb) = attn_tile_bi(hd);
        // attention_bi: every invocation owns HD / KB output dims as vec4s.
        if !hd.is_multiple_of(4 * kb) || hd < 32 {
            bail!("head_dim {hd} must be a multiple of {}", 4 * kb);
        }
        if !self.d.is_multiple_of(32) || !self.ff.is_multiple_of(32) {
            bail!("hidden_size and intermediate_size must be multiples of 32");
        }
        // layernorm.wgsl keeps a row in registers, 8 elements per invocation.
        if self.d > 2048 {
            bail!("hidden_size {} > 2048", self.d);
        }
        Ok(())
    }

    /// Temperature for a question of type `qtype` with `k` options: the
    /// `(type, bucket)` override when the config has one, else the type's.
    pub fn temperature_for(&self, qtype: usize, k: usize) -> f32 {
        let size = if k <= 2 {
            "2"
        } else if k <= 5 {
            "3-5"
        } else if k <= 10 {
            "6-10"
        } else {
            "11+"
        };
        self.temperature_by_options
            .get(&format!("{}:{size}", QTYPES[qtype]))
            .copied()
            .unwrap_or(self.temperature[qtype])
    }

    /// Every tensor the engine needs. Linears are `[out, in]`; the scorer
    /// and act-head tensors (names `scorer.*`, `act.*`) stay on the host.
    pub fn tensors(&self) -> Vec<TensorSpec> {
        let d = self.d;
        let ff = self.ff;
        let mut v = vec![
            TensorSpec::new("embed", vec![self.vocab, d]),
            TensorSpec::new("embed_norm", vec![d]),
            TensorSpec::new("final_norm", vec![d]),
        ];
        if self.type_emb {
            v.push(TensorSpec::new("type_emb", vec![3, d]));
        }
        for l in 0..self.layers {
            let n = |s: &str| format!("enc.{l}.{s}");
            if l > 0 {
                v.push(TensorSpec::new(&n("attn_norm"), vec![d]));
            }
            v.push(TensorSpec::new(&n("qkv"), vec![3 * d, d]));
            v.push(TensorSpec::new(&n("o"), vec![d, d]));
            v.push(TensorSpec::new(&n("mlp_norm"), vec![d]));
            v.push(TensorSpec::new(&n("wi_val"), vec![ff, d]));
            v.push(TensorSpec::new(&n("wi_gate"), vec![ff, d]));
            v.push(TensorSpec::new(&n("wo"), vec![d, ff]));
        }
        for h in 0..self.head_layers {
            let n = |s: &str| format!("head.{h}.{s}");
            v.push(TensorSpec::new(&n("norm1_w"), vec![d]));
            v.push(TensorSpec::new(&n("norm1_b"), vec![d]));
            v.push(TensorSpec::new(&n("in_proj"), vec![3 * d, d]));
            v.push(TensorSpec::new(&n("in_proj_b"), vec![3 * d]));
            v.push(TensorSpec::new(&n("out_proj"), vec![d, d]));
            v.push(TensorSpec::new(&n("out_proj_b"), vec![d]));
            v.push(TensorSpec::new(&n("norm2_w"), vec![d]));
            v.push(TensorSpec::new(&n("norm2_b"), vec![d]));
            v.push(TensorSpec::new(&n("l1"), vec![4 * d, d]));
            v.push(TensorSpec::new(&n("l1_b"), vec![4 * d]));
            v.push(TensorSpec::new(&n("l2"), vec![d, 4 * d]));
            v.push(TensorSpec::new(&n("l2_b"), vec![d]));
        }
        v.push(TensorSpec::new("scorer.norm_w", vec![d]));
        v.push(TensorSpec::new("scorer.norm_b", vec![d]));
        v.push(TensorSpec::new("scorer.l1", vec![d, d]));
        v.push(TensorSpec::new("scorer.l1_b", vec![d]));
        v.push(TensorSpec::new("scorer.l2", vec![1, d]));
        v.push(TensorSpec::new("scorer.l2_b", vec![1]));
        if self.act_out > 0 {
            v.push(TensorSpec::new("act.l1", vec![self.act_hidden, d + 4]));
            v.push(TensorSpec::new("act.l1_b", vec![self.act_hidden]));
            v.push(TensorSpec::new(
                "act.l2",
                vec![self.act_out, self.act_hidden],
            ));
            v.push(TensorSpec::new("act.l2_b", vec![self.act_out]));
        }
        v
    }
}

/// Load an exported directory's `config.json` (tools/export_laya.py): the
/// encoder config carrying the agent config under `laya_agent` and the
/// special tokens under `laya_tokens`.
pub fn config_from_export(
    config: &Value,
    special: impl Fn(&str) -> Option<u32>,
) -> Result<LayaConfig> {
    let agent = config
        .get("laya_agent")
        .ok_or_else(|| anyhow!("config.json: missing laya_agent"))?;
    let tokens = config
        .get("laya_tokens")
        .ok_or_else(|| anyhow!("config.json: missing laya_tokens"))?;
    LayaConfig::from_json(config, agent, tokens, special)
}

/// Whether a catalogued tensor is evaluated on the host.
pub fn is_host_tensor(name: &str) -> bool {
    name.starts_with("scorer.") || name.starts_with("act.")
}

/// Host copy of a checkpoint: every tensor of [`LayaConfig::tensors`].
pub struct LayaWeights {
    pub config: LayaConfig,
    pub tensors: HashMap<String, QTensor>,
}

impl LayaWeights {
    /// A laya checkpoint's `model.safetensors` (upstream PyTorch names, or
    /// laya-mlx's, which only differ in the scorer / act head prefixes).
    pub fn load(config: LayaConfig, safetensors: &[u8]) -> Result<Self> {
        let st = SafeTensors::parse(safetensors)?;
        let get = |a: &str, b: &str| st.tensor(a).or_else(|_| st.tensor(b));
        let mut tensors = HashMap::new();
        let mut put = |name: &str, t: crate::model::Tensor16| {
            tensors.insert(name.to_string(), QTensor::from_f16(t.shape, &t.data));
        };
        let d = config.d;
        let ff = config.ff;
        let e = |s: &str| format!("encoder.{s}");
        let embed = st.tensor(&e("embeddings.tok_embeddings.weight"))?;
        if embed.shape != [config.vocab, d] {
            bail!(
                "tok_embeddings is {:?}, config says [{}, {}]",
                embed.shape,
                config.vocab,
                d
            );
        }
        put("embed", embed);
        put("embed_norm", st.tensor(&e("embeddings.norm.weight"))?);
        put("final_norm", st.tensor(&e("final_norm.weight"))?);
        put("type_emb", st.tensor("type_emb.weight")?);
        for l in 0..config.layers {
            let src = |s: &str| st.tensor(&e(&format!("layers.{l}.{s}")));
            let n = |s: &str| format!("enc.{l}.{s}");
            if l > 0 {
                put(&n("attn_norm"), src("attn_norm.weight")?);
            }
            put(&n("qkv"), src("attn.Wqkv.weight")?);
            put(&n("o"), src("attn.Wo.weight")?);
            put(&n("mlp_norm"), src("mlp_norm.weight")?);
            // Wi = [value; gate]: gelu(value) * gate.
            let wi = src("mlp.Wi.weight")?;
            if wi.shape != [2 * ff, d] {
                bail!(
                    "layer {l}: mlp.Wi is {:?}, expected [{}, {d}]",
                    wi.shape,
                    2 * ff
                );
            }
            let (val, gate) = wi.data.split_at(ff * d);
            put(
                &n("wi_val"),
                crate::model::Tensor16 {
                    shape: vec![ff, d],
                    data: val.to_vec(),
                },
            );
            put(
                &n("wi_gate"),
                crate::model::Tensor16 {
                    shape: vec![ff, d],
                    data: gate.to_vec(),
                },
            );
            put(&n("wo"), src("mlp.Wo.weight")?);
        }
        for h in 0..config.head_layers {
            let src = |s: &str| st.tensor(&format!("head.layers.{h}.{s}"));
            let n = |s: &str| format!("head.{h}.{s}");
            put(&n("norm1_w"), src("norm1.weight")?);
            put(&n("norm1_b"), src("norm1.bias")?);
            put(
                &n("in_proj"),
                get(
                    &format!("head.layers.{h}.self_attn.in_proj.weight"),
                    &format!("head.layers.{h}.self_attn.in_proj_weight"),
                )?,
            );
            put(
                &n("in_proj_b"),
                get(
                    &format!("head.layers.{h}.self_attn.in_proj.bias"),
                    &format!("head.layers.{h}.self_attn.in_proj_bias"),
                )?,
            );
            put(&n("out_proj"), src("self_attn.out_proj.weight")?);
            put(&n("out_proj_b"), src("self_attn.out_proj.bias")?);
            put(&n("norm2_w"), src("norm2.weight")?);
            put(&n("norm2_b"), src("norm2.bias")?);
            put(&n("l1"), src("linear1.weight")?);
            put(&n("l1_b"), src("linear1.bias")?);
            put(&n("l2"), src("linear2.weight")?);
            put(&n("l2_b"), src("linear2.bias")?);
        }
        // nn.Sequential indices: scorer = LayerNorm(0), Linear(1), GELU, Linear(3);
        // act_head = Linear(0), GELU, Linear(2).
        let seq = |p: &str, i: usize, s: &str| {
            get(&format!("{p}.layers.{i}.{s}"), &format!("{p}.{i}.{s}"))
        };
        put("scorer.norm_w", seq("scorer", 0, "weight")?);
        put("scorer.norm_b", seq("scorer", 0, "bias")?);
        put("scorer.l1", seq("scorer", 1, "weight")?);
        put("scorer.l1_b", seq("scorer", 1, "bias")?);
        put("scorer.l2", seq("scorer", 3, "weight")?);
        put("scorer.l2_b", seq("scorer", 3, "bias")?);
        put("act.l1", seq("act_head", 0, "weight")?);
        put("act.l1_b", seq("act_head", 0, "bias")?);
        put("act.l2", seq("act_head", 2, "weight")?);
        put("act.l2_b", seq("act_head", 2, "bias")?);
        let w = LayaWeights { config, tensors };
        w.check()?;
        Ok(w)
    }

    pub fn get(&self, name: &str) -> Result<&QTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    pub fn check(&self) -> Result<()> {
        for spec in self.config.tensors() {
            let t = self.get(&spec.name)?;
            if t.shape != spec.shape {
                bail!(
                    "{}: shape {:?}, expected {:?}",
                    spec.name,
                    t.shape,
                    spec.shape
                );
            }
        }
        Ok(())
    }
}

/// One question, tokenized by [`prompt::build_sequence`].
#[derive(Debug, Clone)]
pub struct Sequence {
    pub ids: Vec<u32>,
    /// Token index of each option's `[MASK]`, in option order.
    pub markers: Vec<usize>,
    /// 0 = choice, 1 = score, 2 = noul.
    pub qtype: usize,
}

/// What the engine returns per sequence: the scorer's raw logit per option
/// (before temperature) and the act head's probability for acting on the
/// answer (output 0; upstream's `act_probability`, the rest being the
/// `act_costs` actions), None without an act head.
#[derive(Debug, Clone)]
pub struct Output {
    pub logits: Vec<f32>,
    pub act_probability: Option<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmbedParams {
    t: u32,
    d: u32,
    scale: f32,
    residual: u32,
}

/// layernorm.wgsl: LayerNorm of a row, optionally after adding a linear
/// layer's output (and its bias) to it, optionally written over the row.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LnParams {
    t: u32,
    d: u32,
    eps: f32,
    has_bias: u32,
    residual: u32,
    has_rbias: u32,
    in_place: u32,
    _pad: u32,
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

/// One encoder layer: 8 dispatches. The residual adds ride in the
/// LayerNorm that follows them (`add_o_norm` = x += o, h = mlp_norm(x);
/// `add_wo_norm` = x += wo, h = the next layer's attn_norm(x) — or, on the
/// last layer, x = final_norm(x) in place). Layer 0 has no attention norm
/// (the embedding norm is it) and projects x directly.
struct EncLayer {
    mm_qkv: Step,
    rope: Step,
    attn: Step,
    mm_o: Step,
    add_o_norm: Step,
    mm_wi: Step,
    mm_wo: Step,
    add_wo_norm: Step,
    sliding: bool,
    /// The layer whose attention norm `add_wo_norm` applies: the last
    /// layer normalizes x in place with the final norm instead.
    last: bool,
}

/// One decision-head layer. `norm1` is only dispatched on the first layer:
/// the others' is fused into the previous layer's `add_l2` (the last
/// layer's `add_l2` is a plain residual add; the scorer reads x on the
/// host).
struct HeadLayer {
    norm1: Step,
    mm_in: Step,
    bias_in: Step,
    attn: Step,
    mm_out: Step,
    add_out_norm2: Step,
    mm_l1: Step,
    relu_l1: Step,
    mm_l2: Step,
    add_l2: Step,
    last: bool,
}

/// Host-side scorer and act head (f32).
struct Host {
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    l1: Vec<f32>,
    l1_b: Vec<f32>,
    l2: Vec<f32>,
    l2_b: f32,
    act_l1: Vec<f32>,
    act_l1_b: Vec<f32>,
    act_l2: Vec<f32>,
    act_l2_b: Vec<f32>,
}

/// Receives the weights one tensor at a time, then wires the engine.
pub struct LayaBuilder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: LayaConfig,
    expected: HashMap<String, Vec<usize>>,
    weights: HashMap<String, GpuTensor>,
    host: HashMap<String, Vec<f32>>,
    profile: bool,
}

impl LayaBuilder {
    pub async fn new(config: LayaConfig) -> Result<Self> {
        config.validate()?;
        let need = (config.vocab * config.d * 2) as u64;
        let (device, queue, profile) = open_device(
            need,
            attn_workgroup_bytes(config.head_dim(), attn_tile_bi(config.head_dim())),
        )
        .await?;
        let expected = config
            .tensors()
            .into_iter()
            .map(|s| (s.name, s.shape))
            .collect();
        Ok(LayaBuilder {
            device,
            queue,
            config,
            expected,
            weights: HashMap::new(),
            host: HashMap::new(),
            profile,
        })
    }

    pub fn config(&self) -> &LayaConfig {
        &self.config
    }

    pub fn missing(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .expected
            .keys()
            .filter(|n| !self.weights.contains_key(*n) && !self.host.contains_key(*n))
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
        if is_host_tensor(name) {
            self.host.insert(name.to_string(), t.to_f32());
            return Ok(());
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

    /// Allocate the workspace for `capacity` packed tokens and `max_rows`
    /// read-back rows (one per option marker plus one per sequence) and wire
    /// every dispatch.
    pub fn finish(self, capacity: usize, max_rows: usize) -> Result<LayaEngine> {
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
        let LayaBuilder {
            device,
            queue,
            config: cfg,
            weights,
            host,
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
        // A dispatch may not bind one buffer both read-only and read-write.
        let dummy_rw = f32_buf("dummy_rw", 4);
        let u32_buf = |name: &str, n: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: (n.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let ids = u32_buf("ids", capacity);
        let qtypes = u32_buf("qtypes", capacity);
        let meta = u32_buf("meta", capacity * 4);
        let x = f32_buf("x", capacity * d);
        let h = f32_buf("h", capacity * d);
        let qkv = f32_buf("qkv", capacity * 3 * d);
        let attn = f32_buf("attn", capacity * d);
        let a = f32_buf("a", capacity * d);
        let act = f32_buf("act", capacity * cfg.ff.max(4 * d));
        let rows = f32_buf("rows", max_rows * d);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (max_rows * d * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let param_slots = 10 + cfg.layers * 10 + cfg.head_layers * 11;
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
        // layernorm.wgsl bindings: (added row, w, b, added row's bias, x, out).
        let embed_t = w("embed")?;
        // Embedding rows land in h; their LayerNorm writes the residual stream x.
        let embed = step(
            K::Embed,
            0,
            embed_t.dtype.code(),
            "embed",
            &[&ids, &embed_t.data, sc(embed_t, &dummy), &h],
        );
        let embed_norm = step(
            K::LayerNorm,
            0,
            0,
            "embed_norm",
            &[&dummy, &w("embed_norm")?.data, &dummy, &dummy, &h, &x],
        );
        // x = final_norm(x) (in place, by the last encoder layer) + type_emb[qtype].
        let type_embed = if cfg.type_emb {
            let type_t = w("type_emb")?;
            Some(step(
                K::Embed,
                0,
                type_t.dtype.code(),
                "type_emb",
                &[&qtypes, &type_t.data, sc(type_t, &dummy), &x],
            ))
        } else {
            None
        };

        let mut enc = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let n = |s: &str| format!("enc.{l}.{s}");
            let t_qkv = w(&n("qkv"))?;
            let t_o = w(&n("o"))?;
            let t_val = w(&n("wi_val"))?;
            let t_gate = w(&n("wi_gate"))?;
            let t_wo = w(&n("wo"))?;
            if t_val.dtype != t_gate.dtype {
                bail!("layer {l}: Wi halves must share a storage type");
            }
            let last = l + 1 == cfg.layers;
            // The norm after this layer's MLP: the next layer's attention
            // norm into h, or the final norm over x itself.
            let (next_norm, next_out) = if last {
                (w("final_norm")?, &dummy_rw)
            } else {
                (w(&format!("enc.{}.attn_norm", l + 1))?, &h)
            };
            let qkv_src = if l > 0 { &h } else { &x };
            enc.push(EncLayer {
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
                add_o_norm: step(
                    K::LayerNorm,
                    0,
                    0,
                    "add_o_norm",
                    &[&a, &w(&n("mlp_norm"))?.data, &dummy, &dummy, &x, &h],
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
                add_wo_norm: step(
                    K::LayerNorm,
                    0,
                    0,
                    "add_wo_norm",
                    &[&a, &next_norm.data, &dummy, &dummy, &x, next_out],
                ),
                sliding: cfg.sliding[l],
                last,
            });
        }
        let mut head = Vec::with_capacity(cfg.head_layers);
        for hl in 0..cfg.head_layers {
            let n = |s: &str| format!("head.{hl}.{s}");
            let t_in = w(&n("in_proj"))?;
            let t_out = w(&n("out_proj"))?;
            let t_l1 = w(&n("l1"))?;
            let t_l2 = w(&n("l2"))?;
            let last = hl + 1 == cfg.head_layers;
            head.push(HeadLayer {
                norm1: step(
                    K::LayerNorm,
                    0,
                    0,
                    "norm1",
                    &[
                        &dummy,
                        &w(&n("norm1_w"))?.data,
                        &w(&n("norm1_b"))?.data,
                        &dummy,
                        &x,
                        &h,
                    ],
                ),
                mm_in: step(
                    K::Matmul,
                    0,
                    t_in.dtype.code(),
                    "mm_in",
                    &[&h, &t_in.data, sc(t_in, &dummy), &qkv],
                ),
                bias_in: step(
                    K::BiasAct,
                    0,
                    0,
                    "bias_in",
                    &[&dummy, &w(&n("in_proj_b"))?.data, &qkv],
                ),
                attn: step(
                    K::AttentionBi,
                    hd,
                    0,
                    "head_attention",
                    &[&qkv, &meta, &attn],
                ),
                mm_out: step(
                    K::Matmul,
                    0,
                    t_out.dtype.code(),
                    "mm_out",
                    &[&attn, &t_out.data, sc(t_out, &dummy), &a],
                ),
                add_out_norm2: step(
                    K::LayerNorm,
                    0,
                    0,
                    "add_out_norm2",
                    &[
                        &a,
                        &w(&n("norm2_w"))?.data,
                        &w(&n("norm2_b"))?.data,
                        &w(&n("out_proj_b"))?.data,
                        &x,
                        &h,
                    ],
                ),
                mm_l1: step(
                    K::Matmul,
                    0,
                    t_l1.dtype.code(),
                    "mm_l1",
                    &[&h, &t_l1.data, sc(t_l1, &dummy), &act],
                ),
                relu_l1: step(
                    K::BiasAct,
                    0,
                    0,
                    "relu_l1",
                    &[&dummy, &w(&n("l1_b"))?.data, &act],
                ),
                mm_l2: step(
                    K::Matmul,
                    0,
                    t_l2.dtype.code(),
                    "mm_l2",
                    &[&act, &t_l2.data, sc(t_l2, &dummy), &a],
                ),
                add_l2: if last {
                    step(K::BiasAct, 0, 0, "add_l2", &[&a, &w(&n("l2_b"))?.data, &x])
                } else {
                    let m = |s: &str| format!("head.{}.{s}", hl + 1);
                    step(
                        K::LayerNorm,
                        0,
                        0,
                        "add_l2_norm1",
                        &[
                            &a,
                            &w(&m("norm1_w"))?.data,
                            &w(&m("norm1_b"))?.data,
                            &w(&n("l2_b"))?.data,
                            &x,
                            &h,
                        ],
                    )
                },
                last,
            });
        }
        let hv = |name: &str| -> Result<Vec<f32>> {
            if name.starts_with("act.") && cfg.act_out == 0 {
                return Ok(Vec::new());
            }
            host.get(name)
                .cloned()
                .ok_or_else(|| anyhow!("missing host tensor {name}"))
        };
        let host = Host {
            norm_w: hv("scorer.norm_w")?,
            norm_b: hv("scorer.norm_b")?,
            l1: hv("scorer.l1")?,
            l1_b: hv("scorer.l1_b")?,
            l2: hv("scorer.l2")?,
            l2_b: hv("scorer.l2_b")?[0],
            act_l1: hv("act.l1")?,
            act_l1_b: hv("act.l1_b")?,
            act_l2: hv("act.l2")?,
            act_l2_b: hv("act.l2_b")?,
        };
        let profile = profile.then(|| Profiler::new(&device, &queue, 2 * param_slots as u32));
        Ok(LayaEngine {
            device,
            queue,
            config: cfg,
            capacity,
            max_rows,
            kernels,
            params,
            param_slots,
            ids,
            qtypes,
            meta,
            x,
            rows,
            staging,
            embed,
            embed_norm,
            type_embed,
            enc,
            head,
            host,
            profile,
            _keep: vec![h, qkv, attn, a, act, dummy],
            _weights: weights.into_values().collect(),
        })
    }
}

pub struct LayaEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub config: LayaConfig,
    pub capacity: usize,
    pub max_rows: usize,
    kernels: Kernels,
    params: wgpu::Buffer,
    param_slots: usize,
    ids: wgpu::Buffer,
    qtypes: wgpu::Buffer,
    meta: wgpu::Buffer,
    x: wgpu::Buffer,
    rows: wgpu::Buffer,
    staging: wgpu::Buffer,
    embed: Step,
    embed_norm: Step,
    type_embed: Option<Step>,
    enc: Vec<EncLayer>,
    head: Vec<HeadLayer>,
    host: Host,
    /// GPU time per dispatch (OMG_WGPU_PROFILE, native only).
    profile: Option<Profiler>,
    _keep: Vec<wgpu::Buffer>,
    _weights: Vec<GpuTensor>,
}

impl LayaEngine {
    /// Open the default adapter and upload a whole host checkpoint.
    pub async fn new(weights: &LayaWeights, capacity: usize, max_rows: usize) -> Result<Self> {
        weights.check()?;
        let mut b = LayaBuilder::new(weights.config.clone()).await?;
        for spec in weights.config.tensors() {
            b.push(&spec.name, weights.get(&spec.name)?)?;
        }
        b.finish(capacity, max_rows)
    }

    /// Run a one-token sequence so the first real request does not pay for
    /// the workspace's first touch.
    pub async fn warmup(&self) -> Result<()> {
        self.evaluate(&[Sequence {
            ids: vec![self.config.cls, self.config.mask, self.config.sep],
            markers: vec![1],
            qtype: 2,
        }])
        .await?;
        Ok(())
    }

    /// The hidden rows the scorer reads, per sequence: `[CLS]` first, then
    /// each marker, each `d` wide.
    pub async fn hidden<'s>(&'s self, seqs: &[Sequence]) -> Result<Vec<Vec<Vec<f32>>>> {
        let cfg = &self.config;
        let d = cfg.d;
        let hd = cfg.head_dim();
        let total: usize = seqs.iter().map(|s| s.ids.len()).sum();
        if total == 0 {
            bail!("empty request");
        }
        if total > self.capacity {
            bail!(
                "{total} tokens exceed the engine capacity {}",
                self.capacity
            );
        }
        let mut ids: Vec<u32> = Vec::with_capacity(total);
        let mut qtypes: Vec<u32> = Vec::with_capacity(total);
        let mut meta: Vec<i32> = Vec::with_capacity(total * 4);
        // Rows to read back: (sequence, slot, workspace row).
        let mut wanted: Vec<(usize, usize, usize)> = Vec::new();
        for (si, s) in seqs.iter().enumerate() {
            if s.ids.is_empty() {
                bail!("sequence {si} is empty");
            }
            if s.ids.len() > cfg.max_len {
                bail!(
                    "sequence {si}: {} tokens > max_len {}",
                    s.ids.len(),
                    cfg.max_len
                );
            }
            if s.qtype >= 3 {
                bail!("sequence {si}: question type {}", s.qtype);
            }
            let start = ids.len();
            for (i, &t) in s.ids.iter().enumerate() {
                if t as usize >= cfg.vocab {
                    bail!("token id {t} outside the vocabulary of {}", cfg.vocab);
                }
                ids.push(t);
                qtypes.push(s.qtype as u32);
                meta.push(i as i32);
                meta.push(si as i32);
                meta.push(start as i32);
                meta.push((start + s.ids.len()) as i32);
            }
            wanted.push((si, 0, start));
            for (j, &m) in s.markers.iter().enumerate() {
                if m >= s.ids.len() {
                    bail!("sequence {si}: marker {m} past its {} tokens", s.ids.len());
                }
                wanted.push((si, j + 1, start + m));
            }
        }
        if wanted.len() > self.max_rows {
            bail!(
                "{} rows requested, engine reads back at most {}",
                wanted.len(),
                self.max_rows
            );
        }
        let t = total;
        self.queue
            .write_buffer(&self.ids, 0, bytemuck::cast_slice(&ids));
        self.queue
            .write_buffer(&self.qtypes, 0, bytemuck::cast_slice(&qtypes));
        self.queue
            .write_buffer(&self.meta, 0, bytemuck::cast_slice(&meta));

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
            // (has_bias, residual, has_rbias, in_place)
            let ln = |bias: bool, residual: bool, rbias: bool, in_place: bool| LnParams {
                t: t as u32,
                d: d as u32,
                eps: cfg.eps,
                has_bias: bias as u32,
                residual: residual as u32,
                has_rbias: rbias as u32,
                in_place: in_place as u32,
                _pad: 0,
            };
            let mm = |n: usize, k: usize| MatmulParams {
                m: t as u32,
                n: n as u32,
                k: k as u32,
                _pad: 0,
            };
            let mm_wg = |n: usize| (div_ceil(n, 128), div_ceil(t, 32));
            // (mode, has_bias, in_place, residual)
            let bias =
                |n: usize, mode: u32, has_bias: bool, in_place: bool, residual: bool| BiasParams {
                    t: t as u32,
                    n: n as u32,
                    mode,
                    has_bias: has_bias as u32,
                    in_place: in_place as u32,
                    residual: residual as u32,
                    _p0: 0,
                    _p1: 0,
                };
            let ew = |n: usize| (div_ceil(t * n, 256), 1);
            let (rows_per_wg, _) = attn_tile_bi(hd);
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

            let off = params.push(EmbedParams {
                t: t as u32,
                d: d as u32,
                scale: 1.0,
                residual: 0,
            });
            run("embed", &self.embed, off, (div_ceil(t * d / 4, 256), 1));
            let off = params.push(ln(false, false, false, false));
            run("embed_norm", &self.embed_norm, off, (t as u32, 1));
            for l in &self.enc {
                let off = params.push(mm(3 * d, d));
                run("mm_qkv", &l.mm_qkv, off, mm_wg(3 * d));
                let off = params.push(RopeParams {
                    t: t as u32,
                    heads: cfg.heads as u32,
                    stride: 3 * d as u32,
                    k_off: d as u32,
                    theta: if l.sliding {
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
                    &l.rope,
                    off,
                    (div_ceil(t * cfg.heads * hd / 2, 256), 1),
                );
                let off = params.push(attn_p(if l.sliding { cfg.window } else { 0 }));
                run("attention", &l.attn, off, attn_wg);
                let off = params.push(mm(d, d));
                run("mm_o", &l.mm_o, off, mm_wg(d));
                let off = params.push(ln(false, true, false, false));
                run("add_o_norm", &l.add_o_norm, off, (t as u32, 1));
                let off = params.push(mm(cfg.ff, d));
                run("mm_wi", &l.mm_wi, off, mm_wg(cfg.ff));
                let off = params.push(mm(d, cfg.ff));
                run("mm_wo", &l.mm_wo, off, mm_wg(d));
                let off = params.push(ln(false, true, false, l.last));
                run("add_wo_norm", &l.add_wo_norm, off, (t as u32, 1));
            }
            if let Some(te) = &self.type_embed {
                let off = params.push(EmbedParams {
                    t: t as u32,
                    d: d as u32,
                    scale: 1.0,
                    residual: 1,
                });
                run("type_emb", te, off, (div_ceil(t * d / 4, 256), 1));
            }
            for (i, l) in self.head.iter().enumerate() {
                if i == 0 {
                    let off = params.push(ln(true, false, false, false));
                    run("norm1", &l.norm1, off, (t as u32, 1));
                }
                let off = params.push(mm(3 * d, d));
                run("mm_in", &l.mm_in, off, mm_wg(3 * d));
                let off = params.push(bias(3 * d, 0, true, true, false));
                run("bias_in", &l.bias_in, off, ew(3 * d));
                let off = params.push(attn_p(0));
                run("head_attention", &l.attn, off, attn_wg);
                let off = params.push(mm(d, d));
                run("mm_out", &l.mm_out, off, mm_wg(d));
                let off = params.push(ln(true, true, true, false));
                run("add_out_norm2", &l.add_out_norm2, off, (t as u32, 1));
                let off = params.push(mm(4 * d, d));
                run("mm_l1", &l.mm_l1, off, mm_wg(4 * d));
                let off = params.push(bias(4 * d, 1, true, true, false));
                run("relu_l1", &l.relu_l1, off, ew(4 * d));
                let off = params.push(mm(d, 4 * d));
                run("mm_l2", &l.mm_l2, off, mm_wg(d));
                if l.last {
                    let off = params.push(bias(d, 0, true, false, true));
                    run("add_l2", &l.add_l2, off, ew(d));
                } else {
                    let off = params.push(ln(true, true, true, false));
                    run("add_l2_norm1", &l.add_l2, off, (t as u32, 1));
                }
            }
        }
        if params.bytes.len() > self.param_slots * PARAM_SLOT as usize {
            bail!("parameter slots exhausted");
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("laya"),
            });
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
        let row_bytes = (d * 4) as u64;
        for (r, &(_, _, idx)) in wanted.iter().enumerate() {
            enc.copy_buffer_to_buffer(
                &self.x,
                idx as u64 * row_bytes,
                &self.rows,
                r as u64 * row_bytes,
                row_bytes,
            );
        }
        let out_bytes = wanted.len() as u64 * row_bytes;
        enc.copy_buffer_to_buffer(&self.rows, 0, &self.staging, 0, out_bytes);
        self.queue.write_buffer(&self.params, 0, &params.bytes);
        self.queue.submit(Some(enc.finish()));

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

        let mut out: Vec<Vec<Vec<f32>>> = seqs
            .iter()
            .map(|s| Vec::with_capacity(s.markers.len() + 1))
            .collect();
        for (r, &(si, slot, _)) in wanted.iter().enumerate() {
            if out[si].len() != slot {
                bail!("row order mismatch for sequence {si}");
            }
            out[si].push(data[r * d..(r + 1) * d].to_vec());
        }
        Ok(out)
    }

    /// Scorer logits per option and the act probability, per sequence.
    pub async fn evaluate(&self, seqs: &[Sequence]) -> Result<Vec<Output>> {
        let hidden = self.hidden(seqs).await?;
        Ok(hidden.iter().map(|rows| self.score(rows)).collect())
    }

    /// The host half of the model on one sequence's rows (`[CLS]`, then
    /// the markers): exactly laya's `DecisionModel.__call__` after the head.
    fn score(&self, rows: &[Vec<f32>]) -> Output {
        let cfg = &self.config;
        let d = cfg.d;
        let hst = &self.host;
        let logits: Vec<f32> = rows[1..]
            .iter()
            .map(|r| {
                let mut z = layer_norm(r, &hst.norm_w, &hst.norm_b, cfg.eps);
                z = linear(&z, &hst.l1, &hst.l1_b, d);
                for v in &mut z {
                    *v = gelu_erf(*v);
                }
                dot(&z, &hst.l2) + hst.l2_b
            })
            .collect();
        if cfg.act_out == 0 {
            return Output {
                logits,
                act_probability: None,
            };
        }
        // Confidence features over the untempered softmax, with the public
        // runtime's padding to two slots for a one-option question.
        let k = logits.len().max(2);
        let mut padded = logits.clone();
        padded.resize(k, -1e4);
        let p = softmax(&padded);
        let entropy = -p.iter().map(|&v| v * v.max(1e-9).ln()).sum::<f32>() / (k as f32).ln();
        let mut sorted = p.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let top1 = sorted[k - 1];
        let top2 = sorted[k - 2];
        let mut pooled = rows[0].clone();
        pooled.extend_from_slice(&[top1, top1 - top2, entropy, k as f32 / 255.0]);
        let mut a = linear(&pooled, &hst.act_l1, &hst.act_l1_b, d + 4);
        for v in &mut a {
            *v = gelu_erf(*v);
        }
        let act = softmax(&linear(&a, &hst.act_l2, &hst.act_l2_b, cfg.act_hidden));
        Output {
            logits,
            act_probability: Some(act[0]),
        }
    }

    /// Probabilities for a sequence's options under the checkpoint's
    /// calibration temperature for its type and option count.
    pub fn probabilities(&self, qtype: usize, logits: &[f32]) -> Vec<f64> {
        let t = self.config.temperature_for(qtype, logits.len()).max(1e-3);
        let z: Vec<f32> = logits.iter().map(|v| v / t).collect();
        softmax(&z).into_iter().map(f64::from).collect()
    }
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(w)
        .zip(b)
        .map(|((v, w), b)| (v - mean) * inv * w + b)
        .collect()
}

/// `y = W x + b`, W row-major `[out, k]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], k: usize) -> Vec<f32> {
    w.chunks(k).zip(b).map(|(row, b)| dot(row, x) + b).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// Exact GELU (erf), as `nn.GELU` / `mlx.nn.gelu`.
fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// erf to 1.5e-7 (Abramowitz & Stegun 7.1.26).
fn erf(x: f32) -> f32 {
    const A: [f32; 5] = [
        0.254_829_6,
        -0.284_496_74,
        1.421_413_7,
        -1.453_152,
        1.061_405_4,
    ];
    let s = x.signum();
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let poly = ((((A[4] * t + A[3]) * t + A[2]) * t + A[1]) * t + A[0]) * t;
    s * (1.0 - poly * (-a * a).exp())
}

/// Manifest of an exported Laya directory (`tools/export_laya.py`): the
/// same layout as [`crate::model::Manifest`] without a per-layer table.
pub type Manifest = crate::model::Manifest;
