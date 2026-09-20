//! Native Laya runtime: the HF tokenizer of the checkpoint, the prompt
//! builder and the wgpu engine behind grande-core's [`Decider`], so
//! `grande serve --model <laya dir>` speaks `/v1/systemone` with Laya's
//! answers.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use grande_core::{Decider, Diagnostics, Distributions, Error, Mode, Request, Response};
use indexmap::IndexMap;
use serde_json::Value;
use tokenizers::Tokenizer;

use super::prompt::Tokenize;
use super::{config_from_export, LayaBuilder, LayaConfig, LayaEngine, LayaWeights};
use crate::model::Manifest;

pub struct LayaBackend {
    engine: LayaEngine,
    tokenizer: Tokenizer,
    pub config: LayaConfig,
    pub model: String,
    /// Multiplies the checkpoint's own calibration temperatures (1 = as
    /// shipped).
    pub temperature: f32,
}

struct Tok<'a>(&'a Tokenizer);

impl Tokenize for Tok<'_> {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.0
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }
}

impl LayaBackend {
    /// A directory holds a Laya checkpoint when it is an HF / laya-mlx
    /// snapshot (`rl_agent_config.json` + `encoder/config.json`) or an
    /// export of one (`manifest.json` + a `config.json` with `laya_agent`).
    pub fn is_laya_dir(dir: &Path) -> bool {
        if dir.join("rl_agent_config.json").is_file()
            && dir.join("encoder").join("config.json").is_file()
        {
            return true;
        }
        if dir.join("manifest.json").is_file() {
            if let Ok(c) = std::fs::read(dir.join("config.json")) {
                if let Ok(v) = serde_json::from_slice::<Value>(&c) {
                    return v.get("laya_agent").is_some();
                }
            }
        }
        false
    }

    /// `capacity` is the packed-token budget per pass, `max_rows` the rows
    /// read back (one per option plus one per question).
    pub fn load(dir: &Path, capacity: usize, max_rows: usize) -> Result<Self> {
        let read = |p: &Path| std::fs::read(p).with_context(|| format!("reading {}", p.display()));
        let json = |p: &Path| -> Result<Value> {
            serde_json::from_slice(&read(p)?).with_context(|| format!("parsing {}", p.display()))
        };
        let exported = dir.join("manifest.json").is_file();
        let tok_path = if exported {
            dir.join("tokenizer.json")
        } else {
            dir.join("tokenizer").join("tokenizer.json")
        };
        let tokenizer =
            Tokenizer::from_bytes(read(&tok_path)?).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        let special = |s: &str| tokenizer.token_to_id(s);
        let (config, engine) = if exported {
            let config = config_from_export(&json(&dir.join("config.json"))?, special)?;
            let manifest: Manifest = serde_json::from_slice(&read(&dir.join("manifest.json"))?)
                .context("manifest.json")?;
            let mut b = pollster::block_on(LayaBuilder::new(config.clone()))?;
            for file in &manifest.files {
                let bytes = read(&dir.join(&file.path))?;
                for e in &file.tensors {
                    b.push(&e.name, &e.tensor(&bytes)?)?;
                }
            }
            (config, b.finish(capacity, max_rows)?)
        } else {
            let config = LayaConfig::from_json(
                &json(&dir.join("encoder").join("config.json"))?,
                &json(&dir.join("rl_agent_config.json"))?,
                &json(&dir.join("tokenizer").join("tokenizer_config.json"))?,
                special,
            )?;
            let weights =
                LayaWeights::load(config.clone(), &read(&dir.join("model.safetensors"))?)?;
            (
                config,
                pollster::block_on(LayaEngine::new(&weights, capacity, max_rows))?,
            )
        };
        pollster::block_on(engine.warmup())?;
        // The directory's name, unless it is a Hub snapshot hash.
        let model = dir
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|n| !(n.len() >= 32 && n.chars().all(|c| c.is_ascii_hexdigit())))
            .unwrap_or("laya")
            .to_lowercase();
        Ok(LayaBackend {
            engine,
            tokenizer,
            config,
            model,
            temperature: 1.0,
        })
    }

    fn run(
        &mut self,
        req: &Request,
        orders: &IndexMap<String, Vec<usize>>,
    ) -> Result<(Response, Distributions)> {
        let tok = Tok(&self.tokenizer);
        let (resp, dists, diag) = pollster::block_on(self.engine.decide(
            &tok,
            &self.model,
            self.temperature,
            req,
            orders,
        ))?;
        Ok((resp, (dists, diag)))
    }
}

fn backend_err(e: anyhow::Error) -> Error {
    Error::Backend(format!("{e:#}"))
}

impl Decider for LayaBackend {
    fn model(&self) -> &str {
        &self.model
    }

    fn answer(
        &mut self,
        req: &Request,
        _mode: Mode,
    ) -> grande_core::Result<(Response, Diagnostics)> {
        let (resp, (_, diag)) = self.run(req, &IndexMap::new()).map_err(backend_err)?;
        Ok((resp, diag))
    }

    fn distributions(
        &mut self,
        req: &Request,
        orders: &IndexMap<String, Vec<usize>>,
        _mode: Mode,
    ) -> grande_core::Result<Distributions> {
        self.run(req, orders).map(|(_, d)| d).map_err(backend_err)
    }
}
