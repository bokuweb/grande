//! The only contract an inference engine has to satisfy.
//!
//! There is no `generate`. A backend tokenizes text, evaluates a shared prefix
//! plus N isolated branches in one pass, and hands back either the logits or
//! the final hidden state at the positions the readout asked for.

use crate::Result;

/// A vocabulary id in the backend's tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Token(pub i32);

/// What the readout needs from each requested position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Want {
    /// Full-vocabulary logits (label readout).
    Logits,
    /// Final-norm hidden state, `n_embd` wide (pointer readout).
    Hidden,
}

/// One question branch, already tokenized. Positions are branch-relative and
/// the backend must assign absolute positions `prefix_len + i` so every branch
/// looks like "state, then this question" to the model.
#[derive(Debug, Clone)]
pub struct BranchTokens {
    pub tokens: Vec<Token>,
    /// Branch-relative indices whose output the readout wants, ascending.
    pub want: Vec<usize>,
}

/// Rows returned for one branch, in the order of [`BranchTokens::want`].
#[derive(Debug, Clone)]
pub struct BranchOutput {
    pub rows: Vec<Vec<f32>>,
}

/// One request's worth of work for [`Backend::evaluate_many`]: a prefix and
/// the branches that continue from it. Groups in one call are isolated from
/// each other exactly as branches within a group are.
#[derive(Debug, Clone, Copy)]
pub struct Group<'a> {
    pub prefix: &'a [Token],
    pub branches: &'a [BranchTokens],
}

impl Group<'_> {
    /// Tokens the group puts in a pass.
    pub fn tokens(&self) -> usize {
        self.prefix.len() + self.branches.iter().map(|b| b.tokens.len()).sum::<usize>()
    }
    /// Sequences the group occupies (prefix + one per branch).
    pub fn sequences(&self) -> usize {
        1 + self.branches.len()
    }
    /// Rows the group reads back.
    pub fn rows(&self) -> usize {
        self.branches.iter().map(|b| b.want.len()).sum()
    }
}

/// What one group of an [`Backend::evaluate_many`] call produced.
#[derive(Debug, Clone)]
pub struct GroupOutput {
    pub branches: Vec<BranchOutput>,
    /// Where this group's prefix came from, if the backend keeps states.
    pub prefix_source: Option<PrefixSource>,
}

/// How much one `evaluate_many` call may hold; an [`crate::Engine`] splits
/// larger sets into several calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Total tokens (every group's prefix and branches).
    pub tokens: usize,
    /// Total sequences (per group, 1 + its branches).
    pub sequences: usize,
    /// Total rows read back (per branch, the length of its `want`).
    pub rows: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            tokens: usize::MAX,
            sequences: usize::MAX,
            rows: usize::MAX,
        }
    }
}

/// Where the prefix (state) came from in the last `evaluate` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PrefixSource {
    /// Still in the cache from the previous request over the same state.
    Resident,
    /// Restored from the in-memory state cache.
    Ram,
    /// Restored from the on-disk state cache.
    Disk,
    /// Evaluated from scratch.
    Decoded,
}

impl PrefixSource {
    pub fn as_str(self) -> &'static str {
        match self {
            PrefixSource::Resident => "resident",
            PrefixSource::Ram => "ram",
            PrefixSource::Disk => "disk",
            PrefixSource::Decoded => "decoded",
        }
    }
}

pub trait Backend {
    /// Tokenize caller text. Never adds BOS and never emits control tokens for
    /// text that merely looks like one.
    fn tokenize(&self, text: &str) -> Result<Vec<Token>>;
    /// Look up a control / special token by its surface form (`<bos>`, `<unused0>`).
    fn special(&self, name: &str) -> Result<Token>;
    /// Beginning-of-sequence token.
    fn bos(&self) -> Token;
    /// Width of a hidden-state row.
    fn n_embd(&self) -> usize;
    /// Vocabulary size (width of a logits row).
    fn n_vocab(&self) -> usize;
    /// Evaluate `prefix` once, then every branch on top of it, isolated from
    /// each other, in as few passes as the engine allows (ideally one).
    fn evaluate(
        &mut self,
        prefix: &[Token],
        branches: &[BranchTokens],
        want: Want,
    ) -> Result<Vec<BranchOutput>>;
    /// Evaluate several prefix + branches groups in one pass where the
    /// engine can (different requests over different states, batched by a
    /// server). Group `g`'s branches see `g`'s prefix and themselves only.
    /// The default runs one `evaluate` per group; engines override it to
    /// pack everything into a single pass.
    fn evaluate_many(&mut self, groups: &[Group<'_>], want: Want) -> Result<Vec<GroupOutput>> {
        let mut out = Vec::with_capacity(groups.len());
        for g in groups {
            let branches = self.evaluate(g.prefix, g.branches, want)?;
            out.push(GroupOutput {
                branches,
                prefix_source: self.prefix_source(),
            });
        }
        Ok(out)
    }
    /// What one `evaluate_many` call may hold. Unbounded by default (the
    /// default `evaluate_many` runs the groups one by one anyway).
    fn limits(&self) -> Limits {
        Limits::default()
    }
    /// How the last `evaluate` obtained its prefix, if the backend keeps
    /// states around at all.
    fn prefix_source(&self) -> Option<PrefixSource> {
        None
    }
    /// Forget the resident state (diagnostics: lets a benchmark measure a
    /// restore or a cold pass without changing the request). No-op for
    /// backends that keep nothing between requests.
    fn evict_resident(&mut self) {}
}

impl<B: Backend + ?Sized> Backend for Box<B> {
    fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
        (**self).tokenize(text)
    }
    fn special(&self, name: &str) -> Result<Token> {
        (**self).special(name)
    }
    fn bos(&self) -> Token {
        (**self).bos()
    }
    fn n_embd(&self) -> usize {
        (**self).n_embd()
    }
    fn n_vocab(&self) -> usize {
        (**self).n_vocab()
    }
    fn evaluate(
        &mut self,
        prefix: &[Token],
        branches: &[BranchTokens],
        want: Want,
    ) -> Result<Vec<BranchOutput>> {
        (**self).evaluate(prefix, branches, want)
    }
    fn evaluate_many(&mut self, groups: &[Group<'_>], want: Want) -> Result<Vec<GroupOutput>> {
        (**self).evaluate_many(groups, want)
    }
    fn limits(&self) -> Limits {
        (**self).limits()
    }
    fn prefix_source(&self) -> Option<PrefixSource> {
        (**self).prefix_source()
    }
    fn evict_resident(&mut self) {
        (**self).evict_resident()
    }
}
