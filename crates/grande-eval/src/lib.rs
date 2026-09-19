//! Evaluation: JGLUE (JNLI, JCommonsenseQA) as TypeSafe-shaped requests,
//! plus calibration metrics on the returned option logits.
//!
//! Prompts match jev_local's `tools/benchmark_jglue.py` so numbers are
//! comparable across runtimes. Zero-shot, fixed option order, one question per
//! record.

pub mod jglue;
pub mod report;
