//! wgpu backend for grande: a Gemma 3 / Gemma 4 forward pass written in WGSL
//! that runs the same on Metal / Vulkan (native) and WebGPU (browser).
//!
//! The whole request — state prefix plus every question branch — is one
//! packed sequence and one command buffer. Each token carries a (position,
//! sequence) pair: prefix tokens are sequence 0 at positions 0..P, branch b's
//! tokens are sequence b at positions P.. — so every branch reads as "state,
//! then this question" and the attention kernel's visibility rule (prefix or
//! own sequence, causal, window) is the block-causal mask kev describes.
//! There is no KV cache to manage and no batching constraint to work around:
//! the mask is the isolation. Several requests over different states share a
//! pass the same way (`Engine::evaluate_groups`): the high bits of the
//! sequence id number the request, and a query sees its own request's
//! prefix and branch only.
//!
//! Grouped-query attention: `kv_heads` K/V heads (one on the 270M and E2B,
//! two on E4B), each shared by `heads / kv_heads` query heads. Gemma 3 (the
//! trained 270M) loads f16 weights from an HF safetensors checkpoint; Gemma 4
//! (E2B, E4B) loads Q8_0 / Q4_0 weights from a directory exported by
//! tools/export_wgpu_gguf.py (shared K/V, head_dim 512 global layers,
//! per-layer embeddings, see model.rs).
//!
//! Debugging (native only): `GRANDE_WGPU_PROFILE=1` prints GPU time per
//! kernel after each request (timestamp queries), `GRANDE_WGPU_LAYERS=n` runs
//! only the first n layers.

pub mod engine;
#[cfg(feature = "laya")]
pub mod laya;
pub mod model;

pub use engine::{Engine, EngineBuilder, Group, SavedState, SEQ_GROUP_SHIFT};
pub use model::{Config, Dtype, QTensor, Weights};

#[cfg(feature = "native")]
mod backend;
#[cfg(feature = "native")]
pub use backend::WgpuBackend;
#[cfg(all(feature = "native", feature = "laya"))]
pub use laya::LayaBackend;
