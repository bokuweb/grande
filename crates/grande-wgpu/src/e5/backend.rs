//! Native e5 runtime: the HF tokenizer of the export, the wgpu engine and
//! the head behind grande-core's [`Decider`], so `grande serve --model <e5
//! dir>` speaks `/v1/systemone` with the embedding model's answers.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use grande_core::{Decider, Diagnostics, Distributions, Error, Mode, Request, Response};
use indexmap::IndexMap;
use serde_json::Value;
use tokenizers::Tokenizer;

use super::{E5Builder, E5Config, E5Engine, Manifest, Tokenize};

pub struct E5Backend {
    engine: E5Engine,
    tokenizer: Tokenizer,
    pub config: E5Config,
    pub model: String,
    /// Multiplies the head's own calibration temperature (1 = as exported).
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

impl E5Backend {
    /// A directory holds an e5 export (tools/export_e5.py) when it has a
    /// `manifest.json` and a `config.json` with `grande_e5`.
    pub fn is_e5_dir(dir: &Path) -> bool {
        if dir.join("manifest.json").is_file() {
            if let Ok(c) = std::fs::read(dir.join("config.json")) {
                if let Ok(v) = serde_json::from_slice::<Value>(&c) {
                    return v.get("grande_e5").is_some();
                }
            }
        }
        false
    }

    /// `capacity` is the packed-token budget per pass, `max_seqs` the
    /// texts (state + options) per pass.
    pub fn load(dir: &Path, capacity: usize, max_seqs: usize) -> Result<Self> {
        let read = |p: &Path| std::fs::read(p).with_context(|| format!("reading {}", p.display()));
        let json = |p: &Path| -> Result<Value> {
            serde_json::from_slice(&read(p)?).with_context(|| format!("parsing {}", p.display()))
        };
        let tokenizer = Tokenizer::from_bytes(read(&dir.join("tokenizer.json"))?)
            .map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        let config = E5Config::from_json(&json(&dir.join("config.json"))?)?;
        let manifest: Manifest =
            serde_json::from_slice(&read(&dir.join("manifest.json"))?).context("manifest.json")?;
        let mut b = pollster::block_on(E5Builder::new(config.clone()))?;
        for file in &manifest.files {
            let bytes = read(&dir.join(&file.path))?;
            for e in &file.tensors {
                b.push(&e.name, &e.tensor(&bytes)?)?;
            }
        }
        let engine = b.finish(capacity, max_seqs)?;
        pollster::block_on(engine.warmup())?;
        let model = config.name.clone();
        Ok(E5Backend {
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

impl Decider for E5Backend {
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

    /// Queued requests share an embedding batch.
    fn answer_many(
        &mut self,
        reqs: &[&Request],
        _mode: Mode,
    ) -> Vec<grande_core::Result<(Response, Diagnostics)>> {
        let tok = Tok(&self.tokenizer);
        pollster::block_on(
            self.engine
                .decide_many(&tok, &self.model, self.temperature, reqs),
        )
        .into_iter()
        .map(|r| r.map(|(resp, _, diag)| (resp, diag)).map_err(backend_err))
        .collect()
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
