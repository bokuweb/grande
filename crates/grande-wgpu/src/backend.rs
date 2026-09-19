//! Native `Backend`: the HF tokenizer of the checkpoint plus a blocking
//! `evaluate` over the wgpu engine.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Context};
use grande_core::{Backend, BranchOutput, BranchTokens, Error, Token, Want};
use tokenizers::Tokenizer;

use crate::{Engine, Weights};

pub struct WgpuBackend {
    engine: Engine,
    tokenizer: Tokenizer,
    specials: Mutex<HashMap<String, Token>>,
}

impl WgpuBackend {
    /// Load `config.json`, `model.safetensors` and `tokenizer.json` from a
    /// checkpoint directory. `capacity` is the packed-token budget.
    pub fn load(dir: &Path, capacity: usize, max_rows: usize) -> anyhow::Result<Self> {
        let read = |name: &str| {
            std::fs::read(dir.join(name))
                .with_context(|| format!("reading {}", dir.join(name).display()))
        };
        let weights = Weights::load(&read("config.json")?, &read("model.safetensors")?)?;
        let tokenizer = Tokenizer::from_bytes(read("tokenizer.json")?)
            .map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        let engine = pollster::block_on(Engine::new(&weights, capacity, max_rows))?;
        Ok(WgpuBackend {
            engine,
            tokenizer,
            specials: Mutex::new(HashMap::new()),
        })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }
}

fn core_err(e: anyhow::Error) -> Error {
    Error::Backend(format!("{e:#}"))
}

/// Break `<name>`-style control-token surface forms in caller text (same
/// rule as the llama.cpp backend): the tokenizer matches added tokens
/// anywhere in the text, so a zero-width non-joiner after `<` keeps user
/// text from forging a delimiter.
fn neutralize_specials(text: &str) -> String {
    if !text.contains('<') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len() + 8);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        out.push(c);
        if c == '<' {
            if let Some(&n) = chars.peek() {
                if n.is_ascii_alphabetic() || n == '|' || n == '/' {
                    out.push('\u{200c}');
                }
            }
        }
    }
    out
}

impl Backend for WgpuBackend {
    fn tokenize(&self, text: &str) -> grande_core::Result<Vec<Token>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let enc = self
            .tokenizer
            .encode(neutralize_specials(text), false)
            .map_err(|e| Error::Backend(format!("tokenize: {e}")))?;
        Ok(enc.get_ids().iter().map(|&i| Token(i as i32)).collect())
    }

    fn special(&self, name: &str) -> grande_core::Result<Token> {
        if let Some(t) = self.specials.lock().unwrap().get(name) {
            return Ok(*t);
        }
        let id = self
            .tokenizer
            .token_to_id(name)
            .ok_or_else(|| Error::Backend(format!("{name:?} is not a token in this vocabulary")))?;
        let t = Token(id as i32);
        self.specials.lock().unwrap().insert(name.to_string(), t);
        Ok(t)
    }

    fn bos(&self) -> Token {
        Token(self.engine.config.bos as i32)
    }

    fn n_embd(&self) -> usize {
        self.engine.config.d
    }

    fn n_vocab(&self) -> usize {
        self.engine.config.vocab
    }

    fn evaluate(
        &mut self,
        prefix: &[Token],
        branches: &[BranchTokens],
        want: Want,
    ) -> grande_core::Result<Vec<BranchOutput>> {
        let prefix: Vec<u32> = prefix.iter().map(|t| t.0 as u32).collect();
        pollster::block_on(self.engine.evaluate(&prefix, branches, want)).map_err(core_err)
    }
}
