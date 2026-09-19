//! Gemma 3 text checkpoint: config.json + model.safetensors (HF layout),
//! loaded into f16 host buffers in the shapes the shaders expect.

use anyhow::{anyhow, bail, Context, Result};
use half::f16;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub eps: f32,
    pub window: usize,
    /// Per layer: true = sliding window.
    pub sliding: Vec<bool>,
    pub theta_global: f32,
    pub theta_local: f32,
    pub query_scale: f32,
    pub bos: u32,
}

impl Config {
    pub fn from_json(v: &Value) -> Result<Self> {
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
        let head_dim = n("head_dim")?;
        let qps = v["query_pre_attn_scalar"]
            .as_f64()
            .unwrap_or(head_dim as f64);
        Ok(Config {
            vocab: n("vocab_size")?,
            d: n("hidden_size")?,
            layers,
            heads: n("num_attention_heads")?,
            kv_heads: n("num_key_value_heads")?,
            head_dim,
            ff: n("intermediate_size")?,
            eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            window: v["sliding_window"].as_u64().unwrap_or(512) as usize,
            sliding,
            theta_global,
            theta_local,
            query_scale: (qps as f32).powf(-0.5),
            bos: v["bos_token_id"].as_u64().unwrap_or(2) as u32,
        })
    }
}

/// f16 tensor, packed row-major.
pub struct Tensor16 {
    pub shape: Vec<usize>,
    pub data: Vec<f16>,
}

impl Tensor16 {
    pub fn bytes(&self) -> &[u8] {
        // f16 is a `#[repr(transparent)]` u16.
        // SAFETY: the slice is reinterpreted as its own bytes.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 2) }
    }
}

pub struct Layer {
    pub input_norm: Tensor16,
    /// [q(heads*hd) | k(hd) | v(hd), d]
    pub qkv: Tensor16,
    pub q_norm: Tensor16,
    pub k_norm: Tensor16,
    pub o: Tensor16,
    pub post_attn_norm: Tensor16,
    pub pre_ff_norm: Tensor16,
    pub gate: Tensor16,
    pub up: Tensor16,
    pub down: Tensor16,
    pub post_ff_norm: Tensor16,
}

pub struct Weights {
    pub config: Config,
    pub embed: Tensor16,
    pub layers: Vec<Layer>,
    pub final_norm: Tensor16,
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

impl Weights {
    pub fn load(config_json: &[u8], safetensors: &[u8]) -> Result<Self> {
        let config =
            Config::from_json(&serde_json::from_slice(config_json).context("config.json")?)?;
        if config.kv_heads != 1 {
            bail!("only one KV head is supported (got {})", config.kv_heads);
        }
        if config.head_dim != 256 {
            bail!("head_dim must be 256 (got {})", config.head_dim);
        }
        if !matches!(config.heads, 1 | 2 | 4 | 8 | 16) {
            bail!("num_attention_heads must divide 16 (got {})", config.heads);
        }
        if config.d % 16 != 0 || config.ff % 16 != 0 {
            bail!("hidden_size and intermediate_size must be multiples of 16");
        }
        let st = SafeTensors::parse(safetensors)?;
        // Checkpoints may or may not carry the `model.` prefix.
        let prefix = if st.header.get("model.embed_tokens.weight").is_some() {
            "model."
        } else {
            ""
        };
        let get = |name: &str| st.tensor(&format!("{prefix}{name}"));
        let embed = get("embed_tokens.weight")?;
        if embed.shape != [config.vocab, config.d] {
            bail!(
                "embed_tokens is {:?}, config says [{}, {}]",
                embed.shape,
                config.vocab,
                config.d
            );
        }
        let mut layers = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let l = |n: &str| get(&format!("layers.{i}.{n}"));
            layers.push(Layer {
                input_norm: l("input_layernorm.weight")?,
                qkv: concat_rows(&[
                    l("self_attn.q_proj.weight")?,
                    l("self_attn.k_proj.weight")?,
                    l("self_attn.v_proj.weight")?,
                ]),
                q_norm: l("self_attn.q_norm.weight")?,
                k_norm: l("self_attn.k_norm.weight")?,
                o: l("self_attn.o_proj.weight")?,
                post_attn_norm: l("post_attention_layernorm.weight")?,
                pre_ff_norm: l("pre_feedforward_layernorm.weight")?,
                gate: l("mlp.gate_proj.weight")?,
                up: l("mlp.up_proj.weight")?,
                down: l("mlp.down_proj.weight")?,
                post_ff_norm: l("post_feedforward_layernorm.weight")?,
            });
        }
        Ok(Weights {
            config,
            embed,
            layers,
            final_norm: get("norm.weight")?,
        })
    }
}
