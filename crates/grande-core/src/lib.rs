//! grande-core: a System One style decision model core.
//!
//! State in, typed questions in, probability distributions out. Nothing here
//! generates text, touches a file, or knows which inference engine is
//! underneath. The crate renders a request into a shared prefix plus one
//! branch per question, asks a [`Backend`] to evaluate them in one pass, and
//! turns the returned logits / hidden states into typed answers.

pub mod api;
pub mod backend;
pub mod calibration;
pub mod engine;
pub mod math;
pub mod plan;
pub mod readout;
pub mod render;

pub use api::{Answer, Question, Request, Response, Usage};
pub use backend::{Backend, BranchOutput, BranchTokens, PrefixSource, Token, Want};
pub use engine::{Diagnostics, Engine, Mode};
pub use plan::{Folded, Plan};
pub use readout::Readout;
pub use render::{Rendered, RenderedBranch, Renderer, Segment};

use thiserror::Error;

/// Errors surfaced to callers of the core.
#[derive(Debug, Error)]
pub enum Error {
    /// A request failed validation. `path` mirrors the TypeSafe `detail[].loc`.
    #[error("invalid request at {path}: {message}")]
    Invalid { path: String, message: String },
    /// The backend refused or failed the evaluation.
    #[error("backend: {0}")]
    Backend(String),
    /// A label could not be represented as a single token by this backend.
    #[error("label {0:?} is not a single token")]
    LabelNotSingleToken(String),
}

impl Error {
    pub fn invalid(path: impl Into<String>, message: impl Into<String>) -> Self {
        Error::Invalid {
            path: path.into(),
            message: message.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
