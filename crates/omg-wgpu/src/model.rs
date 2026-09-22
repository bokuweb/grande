//! Checkpoint description and host-side tensors.
//!
//! Two model families share one set of kernels:
//! - Gemma 3 text (the trained 270M): one head_dim, every layer has its own
//!   K/V, `(1 + w)` RMSNorm weights, f16 weights from an HF safetensors file.
//! - Gemma 4 text (E2B / E4B): sliding layers at head_dim 256 and global
//!   layers at 512 with partial RoPE, the last `num_kv_shared_layers` layers
//!   reuse the K/V of the last layer of the same type, V is RMS-normalized,
//!   double-wide MLPs on the shared layers, per-layer token embeddings, a
//!   per-layer output scalar, softcapped logits, plain `w` RMSNorm weights,
//!   and quantized (Q8_0 / Q4_0) weights repacked from a GGUF.
//!
//! `Config` describes everything per layer so the engine has one code path;
//! `Config::tensors()` is the catalogue of tensor names and shapes the engine
//! expects, which both loaders (safetensors, exported directory) fill.

use anyhow::{anyhow, bail, Context, Result};
use half::f16;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Gemma3,
    Gemma4,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub arch: Arch,
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    /// Per layer.
    pub head_dim: Vec<usize>,
    /// Per layer: intermediate size.
    pub ff: Vec<usize>,
    /// Per layer: true = sliding window.
    pub sliding: Vec<bool>,
    /// Per layer: the layer whose K/V this layer attends over (itself when
    /// it has its own K/V projection).
    pub kv_source: Vec<usize>,
    /// Per layer: how many of the head_dim dims RoPE rotates (pairs
    /// `(i, i + head_dim/2)` for `i < rope_dims/2`, inv_freq
    /// `theta^(-2i/head_dim)`).
    pub rope_dims: Vec<usize>,
    pub eps: f32,
    pub window: usize,
    pub theta_global: f32,
    pub theta_local: f32,
    pub query_scale: f32,
    /// RMSNorm weight offset: Gemma 3 stores `w` and applies `1 + w`; Gemma 4
    /// (and any GGUF, where the shift is folded in) applies `w`.
    pub norm_offset: f32,
    /// RMS-normalize V (no weight) before attention.
    pub v_norm: bool,
    /// Per-layer input embedding width (0 = none).
    pub per_layer_dim: usize,
    /// Final logit softcapping (0 = none): `cap * tanh(z / cap)`.
    pub softcap: f32,
    pub bos: u32,
}

impl Config {
    pub fn from_json(v: &Value) -> Result<Self> {
        // HF multimodal configs nest the text model.
        let v = if v.get("text_config").is_some() {
            &v["text_config"]
        } else {
            v
        };
        let arch = match v["model_type"].as_str().unwrap_or("gemma3_text") {
            "gemma4_text" | "gemma4" => Arch::Gemma4,
            _ => Arch::Gemma3,
        };
        let n = |k: &str| -> Result<usize> {
            v[k].as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("config.json: missing {k}"))
        };
        let layers = n("num_hidden_layers")?;
        let sliding: Vec<bool> = match v["layer_types"].as_array() {
            Some(a) => a
                .iter()
                .map(|t| t.as_str() == Some("sliding_attention"))
                .collect(),
            None => {
                // Older configs: every k-th layer is global.
                let k = v["_sliding_window_pattern"].as_u64().unwrap_or(6) as usize;
                (0..layers).map(|i| (i + 1) % k != 0).collect()
            }
        };
        if sliding.len() != layers {
            bail!(
                "config.json: layer_types has {} entries for {layers} layers",
                sliding.len()
            );
        }
        let rope = &v["rope_parameters"];
        let theta = |kind: &str, fallback: f64| -> f32 {
            rope[kind]["rope_theta"].as_f64().unwrap_or(fallback) as f32
        };
        let theta_global = theta("full_attention", v["rope_theta"].as_f64().unwrap_or(1e6));
        let theta_local = theta(
            "sliding_attention",
            v["rope_local_base_freq"].as_f64().unwrap_or(1e4),
        );
        let hd_local = n("head_dim")?;
        let hd_global = v["global_head_dim"]
            .as_u64()
            .map(|x| x as usize)
            .unwrap_or(hd_local);
        let head_dim: Vec<usize> = sliding
            .iter()
            .map(|&s| if s { hd_local } else { hd_global })
            .collect();
        let rot =
            |kind: &str| -> f64 { rope[kind]["partial_rotary_factor"].as_f64().unwrap_or(1.0) };
        let (rot_local, rot_global) = (rot("sliding_attention"), rot("full_attention"));
        let rope_dims: Vec<usize> = sliding
            .iter()
            .zip(&head_dim)
            .map(|(&s, &hd)| ((hd as f64) * if s { rot_local } else { rot_global }) as usize)
            .collect();
        let shared = v["num_kv_shared_layers"].as_u64().unwrap_or(0) as usize;
        if shared >= layers {
            bail!("num_kv_shared_layers {shared} >= {layers} layers");
        }
        let first_shared = layers - shared;
        let kv_source: Vec<usize> = (0..layers)
            .map(|i| {
                if i < first_shared {
                    i
                } else {
                    // The last own-K/V layer of the same attention type.
                    (0..first_shared)
                        .rev()
                        .find(|&j| sliding[j] == sliding[i])
                        .unwrap_or(i)
                }
            })
            .collect();
        for (i, &src) in kv_source.iter().enumerate() {
            if src >= first_shared && src != i {
                bail!("layer {i} has no earlier K/V layer of its type to share");
            }
        }
        let ff: Vec<usize> = match &v["intermediate_size"] {
            Value::Array(a) => a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect(),
            Value::Number(x) => {
                let base = x.as_u64().unwrap_or(0) as usize;
                let double = v["use_double_wide_mlp"].as_bool().unwrap_or(false);
                (0..layers)
                    .map(|i| {
                        if double && i >= first_shared {
                            base * 2
                        } else {
                            base
                        }
                    })
                    .collect()
            }
            _ => bail!("config.json: missing intermediate_size"),
        };
        if ff.len() != layers {
            bail!(
                "config.json: intermediate_size has {} entries for {layers} layers",
                ff.len()
            );
        }
        let (query_scale, norm_offset, v_norm) = match arch {
            Arch::Gemma3 => {
                let qps = v["query_pre_attn_scalar"]
                    .as_f64()
                    .unwrap_or(hd_local as f64);
                ((qps as f32).powf(-0.5), 1.0, false)
            }
            // Gemma 4: no query scaling, `w` norms, RMS-normalized V.
            Arch::Gemma4 => (1.0, 0.0, true),
        };
        // A GGUF-derived export has the +1 folded into the norm weights.
        let norm_offset = v["omg_norm_offset"]
            .as_f64()
            .map(|x| x as f32)
            .unwrap_or(norm_offset);
        Ok(Config {
            arch,
            vocab: n("vocab_size")?,
            d: n("hidden_size")?,
            layers,
            heads: n("num_attention_heads")?,
            kv_heads: n("num_key_value_heads")?,
            head_dim,
            ff,
            sliding,
            kv_source,
            rope_dims,
            eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            window: v["sliding_window"].as_u64().unwrap_or(512) as usize,
            theta_global,
            theta_local,
            query_scale,
            norm_offset,
            v_norm,
            per_layer_dim: v["hidden_size_per_layer_input"].as_u64().unwrap_or(0) as usize,
            softcap: v["final_logit_softcapping"].as_f64().unwrap_or(0.0) as f32,
            bos: v["bos_token_id"].as_u64().unwrap_or(2) as u32,
        })
    }

    /// What the kernels can run.
    pub fn validate(&self) -> Result<()> {
        if self.kv_heads == 0 || !self.heads.is_multiple_of(self.kv_heads) {
            bail!(
                "num_key_value_heads {} must divide num_attention_heads {}",
                self.kv_heads,
                self.heads
            );
        }
        // The attention workgroup covers whole tokens of one KV head's query
        // heads, ROWS = 16 rows at a time (attention.wgsl).
        if !matches!(self.heads / self.kv_heads, 1 | 2 | 4 | 8 | 16) {
            bail!(
                "query heads per KV head must divide 16 (got {} / {})",
                self.heads,
                self.kv_heads
            );
        }
        for (i, &hd) in self.head_dim.iter().enumerate() {
            if hd != 256 && hd != 512 {
                bail!("layer {i}: head_dim must be 256 or 512 (got {hd})");
            }
            if self.rope_dims[i] > hd || !self.rope_dims[i].is_multiple_of(2) {
                bail!(
                    "layer {i}: rope_dims {} for head_dim {hd}",
                    self.rope_dims[i]
                );
            }
            if self.kv_source[i] != i && self.head_dim[self.kv_source[i]] != hd {
                bail!(
                    "layer {i}: shares K/V with layer {} of a different head_dim",
                    self.kv_source[i]
                );
            }
        }
        if !self.d.is_multiple_of(32) || self.ff.iter().any(|f| !f.is_multiple_of(32)) {
            bail!("hidden_size and intermediate_size must be multiples of 32");
        }
        if !self.per_layer_dim.is_multiple_of(32) {
            bail!("hidden_size_per_layer_input must be a multiple of 32");
        }
        Ok(())
    }

    pub fn has_kv(&self, layer: usize) -> bool {
        self.kv_source[layer] == layer
    }

    /// Width of layer `l`'s fused projection: q heads, plus the k and v heads
    /// when the layer has its own K/V.
    pub fn qkv_width(&self, l: usize) -> usize {
        let hd = self.head_dim[l];
        if self.has_kv(l) {
            (self.heads + 2 * self.kv_heads) * hd
        } else {
            self.heads * hd
        }
    }

    pub fn max_head_dim(&self) -> usize {
        self.head_dim.iter().copied().max().unwrap_or(256)
    }

    /// Every tensor the engine needs, with its shape (row-major `[out, in]`
    /// for linears). Norm weights are 1-D.
    pub fn tensors(&self) -> Vec<TensorSpec> {
        let d = self.d;
        let mut v = vec![
            TensorSpec::new("embed", vec![self.vocab, d]),
            TensorSpec::new("final_norm", vec![d]),
        ];
        if self.per_layer_dim > 0 {
            let p = self.per_layer_dim;
            v.push(TensorSpec::new("pl_model_proj", vec![p * self.layers, d]));
            v.push(TensorSpec::new("pl_proj_norm", vec![p]));
        }
        for l in 0..self.layers {
            let hd = self.head_dim[l];
            let ff = self.ff[l];
            let n = |s: &str| format!("blk.{l}.{s}");
            v.push(TensorSpec::new(&n("attn_norm"), vec![d]));
            v.push(TensorSpec::new(&n("qkv"), vec![self.qkv_width(l), d]));
            v.push(TensorSpec::new(&n("q_norm"), vec![hd]));
            if self.has_kv(l) {
                v.push(TensorSpec::new(&n("k_norm"), vec![hd]));
            }
            v.push(TensorSpec::new(&n("o"), vec![d, self.heads * hd]));
            v.push(TensorSpec::new(&n("post_attn_norm"), vec![d]));
            v.push(TensorSpec::new(&n("ffn_norm"), vec![d]));
            v.push(TensorSpec::new(&n("gate"), vec![ff, d]));
            v.push(TensorSpec::new(&n("up"), vec![ff, d]));
            v.push(TensorSpec::new(&n("down"), vec![d, ff]));
            v.push(TensorSpec::new(&n("post_ffn_norm"), vec![d]));
            if self.per_layer_dim > 0 {
                let p = self.per_layer_dim;
                v.push(TensorSpec::new(&n("pl_gate"), vec![p, d]));
                v.push(TensorSpec::new(&n("pl_proj"), vec![d, p]));
                v.push(TensorSpec::new(&n("pl_norm"), vec![d]));
                v.push(TensorSpec::new(&n("out_scale"), vec![1]));
            }
        }
        v
    }
}

#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    pub fn new(name: &str, shape: Vec<usize>) -> Self {
        TensorSpec {
            name: name.to_string(),
            shape,
        }
    }
}

/// Storage type of a host tensor. Quantized types use 32-element blocks
/// along the last dimension with one f16 scale per block: the GGUF Q8_0 /
/// Q4_0 codes with the scales split out so the payload is 4-byte aligned.
/// Q8: 32 signed bytes, value `q * scale`. Q4: 16 bytes, low nibbles =
/// elements 0..16, high nibbles = elements 16..32, value `(q - 8) * scale`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F16,
    Q8,
    Q4,
}

impl Dtype {
    pub fn parse(s: &str) -> Result<Dtype> {
        Ok(match s {
            "f16" | "F16" => Dtype::F16,
            "q8" | "q8_0" | "Q8_0" => Dtype::Q8,
            "q4" | "q4_0" | "Q4_0" => Dtype::Q4,
            other => bail!("unsupported dtype {other}"),
        })
    }

    /// Shader constant selecting the weight decoder.
    pub fn code(self) -> u32 {
        match self {
            Dtype::F16 => 0,
            Dtype::Q8 => 1,
            Dtype::Q4 => 2,
        }
    }

    /// Payload bytes for `n` elements (a whole number of blocks).
    pub fn payload_bytes(self, n: usize) -> usize {
        match self {
            Dtype::F16 => n * 2,
            Dtype::Q8 => n,
            Dtype::Q4 => n / 2,
        }
    }
}

pub const BLOCK: usize = 32;

/// A tensor as uploaded: raw little-endian bytes, plus f16 block scales for
/// the quantized types.
#[derive(Debug, Clone)]
pub struct QTensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
    pub scales: Vec<u8>,
}

impl QTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn from_f16(shape: Vec<usize>, values: &[f16]) -> Self {
        let mut data = Vec::with_capacity(values.len() * 2);
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        QTensor {
            dtype: Dtype::F16,
            shape,
            data,
            scales: Vec::new(),
        }
    }

    pub fn from_f32(shape: Vec<usize>, values: &[f32]) -> Self {
        let h: Vec<f16> = values.iter().map(|&v| f16::from_f32(v)).collect();
        Self::from_f16(shape, &h)
    }

    /// Q8_0 quantization of `values` (the GGUF rule: scale = max|v| / 127).
    pub fn quantize_q8(shape: Vec<usize>, values: &[f32]) -> Result<Self> {
        if !values.len().is_multiple_of(BLOCK) {
            bail!("q8: {} values is not a multiple of {BLOCK}", values.len());
        }
        let mut data = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(values.len() / BLOCK * 2);
        for blk in values.chunks(BLOCK) {
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let d = f16::from_f32(d);
            scales.extend_from_slice(&d.to_le_bytes());
            for &v in blk {
                data.push((v * id).round().clamp(-127.0, 127.0) as i8 as u8);
            }
        }
        Ok(QTensor {
            dtype: Dtype::Q8,
            shape,
            data,
            scales,
        })
    }

    /// Q4_0 quantization of `values` (the GGUF rule: scale = max / -8 where
    /// max is the signed value of largest magnitude).
    pub fn quantize_q4(shape: Vec<usize>, values: &[f32]) -> Result<Self> {
        if !values.len().is_multiple_of(BLOCK) {
            bail!("q4: {} values is not a multiple of {BLOCK}", values.len());
        }
        let mut data = Vec::with_capacity(values.len() / 2);
        let mut scales = Vec::with_capacity(values.len() / BLOCK * 2);
        for blk in values.chunks(BLOCK) {
            let mut max = 0.0f32;
            let mut amax = 0.0f32;
            for &v in blk {
                if v.abs() > amax {
                    amax = v.abs();
                    max = v;
                }
            }
            let d = max / -8.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let d = f16::from_f32(d);
            scales.extend_from_slice(&d.to_le_bytes());
            for j in 0..BLOCK / 2 {
                let lo = ((blk[j] * id + 8.5) as i32).clamp(0, 15) as u8;
                let hi = ((blk[j + BLOCK / 2] * id + 8.5) as i32).clamp(0, 15) as u8;
                data.push(lo | (hi << 4));
            }
        }
        Ok(QTensor {
            dtype: Dtype::Q4,
            shape,
            data,
            scales,
        })
    }

    /// Wrap already-encoded bytes (an exported file).
    pub fn from_raw(
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
        scales: Vec<u8>,
    ) -> Result<Self> {
        let n: usize = shape.iter().product();
        let t = QTensor {
            dtype,
            shape,
            data,
            scales,
        };
        if t.data.len() != dtype.payload_bytes(n) {
            bail!(
                "{:?} tensor {:?}: {} payload bytes, expected {}",
                dtype,
                t.shape,
                t.data.len(),
                dtype.payload_bytes(n)
            );
        }
        let want_scales = if dtype == Dtype::F16 {
            0
        } else {
            n / BLOCK * 2
        };
        if t.scales.len() != want_scales {
            bail!(
                "{:?} tensor {:?}: {} scale bytes, expected {want_scales}",
                dtype,
                t.shape,
                t.scales.len()
            );
        }
        Ok(t)
    }

    /// Element `i` as f32 (row-major).
    pub fn get(&self, i: usize) -> f32 {
        match self.dtype {
            Dtype::F16 => f16::from_le_bytes([self.data[2 * i], self.data[2 * i + 1]]).to_f32(),
            Dtype::Q8 => {
                let b = i / BLOCK;
                let d = f16::from_le_bytes([self.scales[2 * b], self.scales[2 * b + 1]]).to_f32();
                (self.data[i] as i8) as f32 * d
            }
            Dtype::Q4 => {
                let b = i / BLOCK;
                let j = i % BLOCK;
                let d = f16::from_le_bytes([self.scales[2 * b], self.scales[2 * b + 1]]).to_f32();
                let byte = self.data[b * 16 + j % 16];
                let q = if j < 16 { byte & 0xf } else { byte >> 4 };
                (q as f32 - 8.0) * d
            }
        }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        (0..self.numel()).map(|i| self.get(i)).collect()
    }

    /// Dequantized row `r` of a 2-D tensor, as f16.
    pub fn row_f16(&self, r: usize, out: &mut [f16]) {
        let k = self.shape[1];
        for (c, o) in out.iter_mut().enumerate().take(k) {
            *o = f16::from_f32(self.get(r * k + c));
        }
    }
}

/// Host copy of a whole checkpoint (small models, tests): every tensor of
/// `Config::tensors()` by name, plus the optional per-layer token table.
pub struct Weights {
    pub config: Config,
    pub tensors: std::collections::HashMap<String, QTensor>,
    /// `[vocab, per_layer_dim * layers]`, gathered on the host per request.
    pub per_layer_table: Option<QTensor>,
}

impl Weights {
    pub fn get(&self, name: &str) -> Result<&QTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    /// Gemma 3 HF checkpoint: `config.json` + `model.safetensors`.
    pub fn load(config_json: &[u8], safetensors: &[u8]) -> Result<Self> {
        let config =
            Config::from_json(&serde_json::from_slice(config_json).context("config.json")?)?;
        config.validate()?;
        if config.arch != Arch::Gemma3 {
            bail!("safetensors loading is for Gemma 3 checkpoints; export Gemma 4 with tools/export_wgpu_gguf.py");
        }
        let st = SafeTensors::parse(safetensors)?;
        // Checkpoints may or may not carry the `model.` prefix.
        let prefix = if st.header.get("model.embed_tokens.weight").is_some() {
            "model."
        } else {
            ""
        };
        let get = |name: &str| st.tensor(&format!("{prefix}{name}"));
        let mut tensors = std::collections::HashMap::new();
        let mut put = |name: &str, t: Tensor16| {
            tensors.insert(name.to_string(), QTensor::from_f16(t.shape, &t.data));
        };
        let embed = get("embed_tokens.weight")?;
        if embed.shape != [config.vocab, config.d] {
            bail!(
                "embed_tokens is {:?}, config says [{}, {}]",
                embed.shape,
                config.vocab,
                config.d
            );
        }
        put("embed", embed);
        put("final_norm", get("norm.weight")?);
        for i in 0..config.layers {
            let l = |n: &str| get(&format!("layers.{i}.{n}"));
            let n = |s: &str| format!("blk.{i}.{s}");
            put(&n("attn_norm"), l("input_layernorm.weight")?);
            put(
                &n("qkv"),
                concat_rows(&[
                    l("self_attn.q_proj.weight")?,
                    l("self_attn.k_proj.weight")?,
                    l("self_attn.v_proj.weight")?,
                ]),
            );
            put(&n("q_norm"), l("self_attn.q_norm.weight")?);
            put(&n("k_norm"), l("self_attn.k_norm.weight")?);
            put(&n("o"), l("self_attn.o_proj.weight")?);
            put(&n("post_attn_norm"), l("post_attention_layernorm.weight")?);
            put(&n("ffn_norm"), l("pre_feedforward_layernorm.weight")?);
            put(&n("gate"), l("mlp.gate_proj.weight")?);
            put(&n("up"), l("mlp.up_proj.weight")?);
            put(&n("down"), l("mlp.down_proj.weight")?);
            put(&n("post_ffn_norm"), l("post_feedforward_layernorm.weight")?);
        }
        let w = Weights {
            config,
            tensors,
            per_layer_table: None,
        };
        w.check()?;
        Ok(w)
    }

    /// Every catalogued tensor present with the right shape.
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
        if self.config.per_layer_dim > 0 {
            let t = self
                .per_layer_table
                .as_ref()
                .ok_or_else(|| anyhow!("missing per-layer token table"))?;
            let want = [
                self.config.vocab,
                self.config.per_layer_dim * self.config.layers,
            ];
            if t.shape != want {
                bail!("per-layer table: shape {:?}, expected {:?}", t.shape, want);
            }
        }
        Ok(())
    }
}

/// f16 tensor, packed row-major (safetensors reader output).
pub struct Tensor16 {
    pub shape: Vec<usize>,
    pub data: Vec<f16>,
}

/// Minimal safetensors reader.
pub struct SafeTensors<'a> {
    header: Value,
    data: &'a [u8],
}

impl<'a> SafeTensors<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 8 {
            bail!("safetensors: too short");
        }
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(
            bytes
                .get(8..8 + n)
                .ok_or_else(|| anyhow!("safetensors: header out of range"))?,
        )
        .context("safetensors header")?;
        Ok(SafeTensors {
            header,
            data: &bytes[8 + n..],
        })
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor16> {
        let t = self
            .header
            .get(name)
            .ok_or_else(|| anyhow!("safetensors: missing {name}"))?;
        let shape: Vec<usize> = t["shape"]
            .as_array()
            .ok_or_else(|| anyhow!("{name}: shape"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let off = t["data_offsets"]
            .as_array()
            .ok_or_else(|| anyhow!("{name}: data_offsets"))?;
        let (a, b) = (
            off[0].as_u64().unwrap_or(0) as usize,
            off[1].as_u64().unwrap_or(0) as usize,
        );
        let raw = self
            .data
            .get(a..b)
            .ok_or_else(|| anyhow!("{name}: offsets out of range"))?;
        let data: Vec<f16> = match t["dtype"].as_str().unwrap_or("") {
            "F32" => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f16::from_f32(f32::from_le_bytes(*c)))
                .collect(),
            "F16" => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_bits(u16::from_le_bytes(*c)))
                .collect(),
            "BF16" => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_f32(f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16)))
                .collect(),
            other => bail!("{name}: unsupported dtype {other}"),
        };
        if data.len() != shape.iter().product::<usize>() {
            bail!("{name}: {} values for shape {shape:?}", data.len());
        }
        Ok(Tensor16 { shape, data })
    }
}

fn concat_rows(parts: &[Tensor16]) -> Tensor16 {
    let cols = parts[0].shape[1];
    let rows: usize = parts.iter().map(|p| p.shape[0]).sum();
    let mut data = Vec::with_capacity(rows * cols);
    for p in parts {
        assert_eq!(p.shape[1], cols);
        data.extend_from_slice(&p.data);
    }
    Tensor16 {
        shape: vec![rows, cols],
        data,
    }
}

/// Manifest of an exported model directory (`tools/export_wgpu_gguf.py`):
/// tensors grouped into files, each entry an encoded `QTensor`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Manifest {
    pub files: Vec<ManifestFile>,
    /// The per-layer token table, if the model has one (its own file).
    #[serde(default)]
    pub per_layer_table: Option<ManifestEntry>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ManifestFile {
    pub path: String,
    pub tensors: Vec<ManifestEntry>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ManifestEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Byte range of the payload within the file.
    pub offset: usize,
    pub nbytes: usize,
    /// Byte range of the scales (quantized types).
    #[serde(default)]
    pub scales_offset: usize,
    #[serde(default)]
    pub scales_nbytes: usize,
    /// For the per-layer table: the file it lives in.
    #[serde(default)]
    pub path: Option<String>,
}

impl ManifestEntry {
    /// Decode this entry out of its file's bytes.
    pub fn tensor(&self, file: &[u8]) -> Result<QTensor> {
        let slice = |off: usize, n: usize| -> Result<Vec<u8>> {
            file.get(off..off + n).map(|s| s.to_vec()).ok_or_else(|| {
                anyhow!(
                    "{}: byte range {off}..{} outside the file",
                    self.name,
                    off + n
                )
            })
        };
        QTensor::from_raw(
            Dtype::parse(&self.dtype)?,
            self.shape.clone(),
            slice(self.offset, self.nbytes)?,
            slice(self.scales_offset, self.scales_nbytes)?,
        )
    }
}
