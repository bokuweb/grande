//! llama.cpp backend for grande.
//!
//! The state prefix is decoded once into sequence 0. Every question branch
//! gets its own sequence id, receives the prefix cells by `llama_memory_seq_cp`
//! (no copy in the unified cache: a cell can belong to many sequences), and
//! all branches are appended to one batch and decoded together. A branch token
//! attends to prefix cells and to its own sequence only, which is exactly the
//! block-causal mask kev uses in PyTorch. Positions restart at `prefix_len`
//! per branch, so every branch is "state, then this question".

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Context};
use grande_core::{Backend, BranchOutput, BranchTokens, Error, Token, Want};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

pub struct Options {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_gpu_layers: u32,
    pub n_threads: i32,
    /// Keep the full sliding-window cache. Verified unnecessary for branch
    /// isolation on Gemma 4 E2B (packed vs separate identical either way);
    /// left as an escape hatch for other SWA models.
    pub swa_full: bool,
    /// Flash attention: None = llama.cpp auto, Some(true/false) = force.
    pub flash: Option<bool>,
    /// Expose hidden states (pointer readout). Off for the label readout:
    /// llama.cpp then computes logits only at requested positions, whereas
    /// embeddings mode marks every token as an output.
    pub embeddings: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            n_ctx: 8192,
            n_batch: 8192,
            n_ubatch: 512,
            n_seq_max: 64,
            n_gpu_layers: 999,
            n_threads: 8,
            swa_full: false,
            flash: None,
            embeddings: false,
        }
    }
}

pub struct LlamaEngine {
    // The context borrows the model, so the model is boxed and leaked for a
    // `'static` borrow, then reclaimed in `Drop` after the context is gone.
    // Metal asserts at process exit if a model buffer is still resident, so
    // the order ctx → model → backend is not optional.
    ctx: ManuallyDrop<LlamaContext<'static>>,
    model: *mut LlamaModel,
    backend: *mut LlamaBackend,
    opts: Options,
    n_seq_max: usize,
    specials: Mutex<HashMap<String, Token>>,
    /// Prefix currently resident in sequence 0, if any. A request over the
    /// same state skips the prefix pass entirely.
    resident_prefix: Option<Vec<Token>>,
}

// SAFETY: every use of the context and model goes through `&mut self` or
// `&self` on one `LlamaEngine`; callers that share it across threads wrap it
// in a `Mutex`, so llama.cpp never sees concurrent access to one context.
// Moving the engine between threads is fine: nothing inside is thread-affine.
unsafe impl Send for LlamaEngine {}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        // SAFETY: ctx is dropped exactly once here, before the model it
        // borrows; model and backend were created by Box::into_raw in `load`.
        unsafe {
            ManuallyDrop::drop(&mut self.ctx);
            drop(Box::from_raw(self.model));
            drop(Box::from_raw(self.backend));
        }
    }
}

impl LlamaEngine {
    /// Load a GGUF and open one context. The model and backend are leaked so
    /// the context can hold a `'static` borrow; an engine lives for the
    /// process anyway.
    pub fn load(path: &Path, opts: Options) -> anyhow::Result<Self> {
        let backend_ptr = Box::into_raw(Box::new(LlamaBackend::init()?));
        // SAFETY: the pointer is valid until `Drop` reclaims it.
        let backend: &'static LlamaBackend = unsafe { &*backend_ptr };
        let mparams = LlamaModelParams::default().with_n_gpu_layers(opts.n_gpu_layers);
        let model_ptr = match LlamaModel::load_from_file(backend, path, &mparams) {
            Ok(m) => Box::into_raw(Box::new(m)),
            Err(e) => {
                unsafe { drop(Box::from_raw(backend_ptr)) };
                return Err(e).with_context(|| format!("loading {}", path.display()));
            }
        };
        let model: &'static LlamaModel = unsafe { &*model_ptr };
        let cparams = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(opts.n_ctx))
            .with_n_batch(opts.n_batch)
            .with_n_ubatch(opts.n_ubatch)
            .with_n_seq_max(opts.n_seq_max)
            .with_n_threads(opts.n_threads)
            .with_n_threads_batch(opts.n_threads)
            .with_swa_full(opts.swa_full)
            // One shared cell pool: `seq_cp` then only adds a sequence id to
            // the prefix cells (no copy), and n_ctx is the request's total
            // budget instead of being split per sequence.
            .with_kv_unified(true)
            .with_flash_attention_policy(match opts.flash {
                None => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO,
                Some(true) => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED,
                Some(false) => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED,
            })
            .with_embeddings(opts.embeddings);
        let ctx = match model.new_context(backend, cparams) {
            Ok(c) => c,
            Err(e) => {
                unsafe {
                    drop(Box::from_raw(model_ptr));
                    drop(Box::from_raw(backend_ptr));
                }
                return Err(e.into());
            }
        };
        let n_seq_max = opts.n_seq_max as usize;
        Ok(LlamaEngine {
            ctx: ManuallyDrop::new(ctx),
            model: model_ptr,
            backend: backend_ptr,
            opts,
            n_seq_max,
            specials: Mutex::new(HashMap::new()),
            resident_prefix: None,
        })
    }

    pub fn model(&self) -> &LlamaModel {
        // SAFETY: valid for the lifetime of self.
        unsafe { &*self.model }
    }

    /// Surface form of a token, control tokens included.
    pub fn piece(&self, t: Token) -> String {
        let mut dec = encoding_rs::UTF_8.new_decoder();
        self.model()
            .token_to_piece(LlamaToken(t.0), &mut dec, true, None)
            .unwrap_or_else(|_| "<?>".into())
    }

    pub fn n_ctx(&self) -> u32 {
        self.ctx.n_ctx()
    }

    fn decode_prefix(&mut self, prefix: &[Token]) -> anyhow::Result<()> {
        if self.resident_prefix.as_deref() == Some(prefix) {
            return Ok(());
        }
        self.resident_prefix = None;
        self.ctx.clear_kv_cache();
        let n_batch = self.opts.n_batch as usize;
        let mut pos = 0i32;
        for chunk in prefix.chunks(n_batch) {
            let mut batch = LlamaBatch::new(chunk.len(), 1);
            for t in chunk {
                batch.add(LlamaToken(t.0), pos, &[0], false)?;
                pos += 1;
            }
            self.ctx.decode(&mut batch).context("decoding prefix")?;
        }
        self.resident_prefix = Some(prefix.to_vec());
        Ok(())
    }

    /// Whether the last request's state is still resident (diagnostics).
    pub fn prefix_resident(&self, prefix: &[Token]) -> bool {
        self.resident_prefix.as_deref() == Some(prefix)
    }

    /// Decode all branches, chunking at `n_batch`, collecting requested rows.
    fn decode_branches(
        &mut self,
        prefix_len: usize,
        branches: &[BranchTokens],
        want: Want,
    ) -> anyhow::Result<Vec<BranchOutput>> {
        let n_batch = self.opts.n_batch as usize;
        let mut outputs: Vec<BranchOutput> = branches
            .iter()
            .map(|b| BranchOutput {
                rows: Vec::with_capacity(b.want.len()),
            })
            .collect();

        // pending: (branch, want slot, batch-local index) for every token
        // whose output was requested in the batch being filled.
        let mut batch = LlamaBatch::new(n_batch, self.n_seq_max as i32);
        let mut pending: Vec<(usize, usize, usize)> = Vec::new();
        for (bi, b) in branches.iter().enumerate() {
            let seq = (bi + 1) as i32;
            self.ctx
                .kv_cache_seq_cp(0, seq, None, None)
                .map_err(|e| anyhow!("seq_cp: {e:?}"))?;
            let mut want_iter = b.want.iter().enumerate().peekable();
            for (j, t) in b.tokens.iter().enumerate() {
                if batch.n_tokens() as usize >= n_batch {
                    self.flush(&mut batch, &mut pending, &mut outputs, want)?;
                }
                let wanted = matches!(want_iter.peek(), Some((_, &w)) if w == j);
                batch.add(LlamaToken(t.0), (prefix_len + j) as i32, &[seq], wanted)?;
                if wanted {
                    let (slot, _) = want_iter.next().unwrap();
                    pending.push((bi, slot, batch.n_tokens() as usize - 1));
                }
            }
        }
        if batch.n_tokens() > 0 {
            self.flush(&mut batch, &mut pending, &mut outputs, want)?;
        }
        Ok(outputs)
    }

    fn flush(
        &mut self,
        batch: &mut LlamaBatch,
        pending: &mut Vec<(usize, usize, usize)>,
        outputs: &mut [BranchOutput],
        want: Want,
    ) -> anyhow::Result<()> {
        self.ctx.decode(batch).context("decoding branches")?;
        for &(bi, slot, pos) in pending.iter() {
            let row: Vec<f32> = match want {
                Want::Logits => self.ctx.get_logits_ith(pos as i32).to_vec(),
                Want::Hidden => self
                    .ctx
                    .embeddings_ith(pos as i32)
                    .map_err(|e| anyhow!("embeddings at {pos}: {e:?}"))?
                    .to_vec(),
            };
            let out = &mut outputs[bi];
            if out.rows.len() != slot {
                return Err(anyhow!("row order mismatch for branch {bi}"));
            }
            out.rows.push(row);
        }
        pending.clear();
        batch.clear();
        Ok(())
    }
}

fn core_err(e: anyhow::Error) -> Error {
    Error::Backend(format!("{e:#}"))
}

impl Backend for LlamaEngine {
    fn tokenize(&self, text: &str) -> grande_core::Result<Vec<Token>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        // `str_to_token` parses special tokens; neutralize their surface forms
        // in user text so delimiters cannot be forged. Gemma's specials are
        // `<name>`; a zero-width joiner after `<` breaks the match without
        // changing what a reader sees.
        let safe = neutralize_specials(text);
        let toks = self
            .model()
            .str_to_token(&safe, AddBos::Never)
            .map_err(|e| core_err(e.into()))?;
        Ok(toks.into_iter().map(|t| Token(t.0)).collect())
    }

    fn special(&self, name: &str) -> grande_core::Result<Token> {
        if let Some(t) = self.specials.lock().unwrap().get(name) {
            return Ok(*t);
        }
        // Tokens the tokenizer treats as control tokens parse directly
        // (`<|turn>`). Gemma's reserved `<unusedN>` tokens are plain vocab
        // entries in the GGUF, so fall back to an exact vocabulary lookup.
        let toks = self
            .model()
            .str_to_token(name, AddBos::Never)
            .map_err(|e| core_err(e.into()))?;
        let found = if toks.len() == 1 && self.piece(Token(toks[0].0)) == name {
            Some(Token(toks[0].0))
        } else {
            let n = self.model().n_vocab();
            (0..n).map(Token).find(|t| self.piece(*t) == name)
        };
        let t = found
            .ok_or_else(|| Error::Backend(format!("{name:?} is not a token in this vocabulary")))?;
        self.specials.lock().unwrap().insert(name.to_string(), t);
        Ok(t)
    }

    fn bos(&self) -> Token {
        Token(self.model().token_bos().0)
    }

    fn n_embd(&self) -> usize {
        self.model().n_embd() as usize
    }

    fn n_vocab(&self) -> usize {
        self.model().n_vocab() as usize
    }

    fn evaluate(
        &mut self,
        prefix: &[Token],
        branches: &[BranchTokens],
        want: Want,
    ) -> grande_core::Result<Vec<BranchOutput>> {
        if branches.len() + 1 > self.n_seq_max {
            return Err(Error::Backend(format!(
                "{} branches exceed n_seq_max {}",
                branches.len(),
                self.n_seq_max - 1
            )));
        }
        let total = prefix.len() + branches.iter().map(|b| b.tokens.len()).sum::<usize>();
        if total > self.ctx.n_ctx() as usize {
            return Err(Error::Backend(format!(
                "{total} tokens exceed n_ctx {}",
                self.ctx.n_ctx()
            )));
        }
        self.decode_prefix(prefix).map_err(core_err)?;
        let out = self
            .decode_branches(prefix.len(), branches, want)
            .map_err(core_err);
        // Drop the branch sequences; the prefix cells (sequence 0) stay
        // resident for the next request over the same state.
        for seq in 1..=branches.len() as i32 {
            self.ctx.kv_cache_seq_rm(seq, None, None).ok();
        }
        out
    }
}

/// Break `<name>`-style control-token surface forms in caller text.
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

#[cfg(test)]
mod tests {
    use super::neutralize_specials;

    #[test]
    fn specials_in_user_text_are_broken_up() {
        assert_eq!(
            neutralize_specials("a <unused0> b"),
            "a <\u{200c}unused0> b"
        );
        assert_eq!(neutralize_specials("x < 3"), "x < 3");
        assert_eq!(
            neutralize_specials("<start_of_turn>"),
            "<\u{200c}start_of_turn>"
        );
    }
}
