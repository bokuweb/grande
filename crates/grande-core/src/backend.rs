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
}
