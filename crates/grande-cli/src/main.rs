//! `grande`: run a System One style request against a local GGUF.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use grande_core::{Backend, Engine, Mode, Readout, Renderer, Request};
use grande_llama::{LlamaEngine, Options};

#[derive(Parser)]
#[command(
    name = "grande",
    about = "Typed questions in, probabilities out, one pass."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum ModeArg {
    Packed,
    Separate,
    /// Run both and report the largest probability difference.
    Check,
}

#[derive(Subcommand)]
enum Cmd {
    /// Answer a request JSON file (TypeSafe `/v1/systemone` shape).
    Probe {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        request: PathBuf,
        #[arg(long, value_enum, default_value = "packed")]
        mode: ModeArg,
        #[arg(long, default_value_t = 1.0)]
        temperature: f32,
        #[arg(long, default_value_t = 8192)]
        n_ctx: u32,
        #[arg(long, default_value_t = 999)]
        n_gpu_layers: u32,
        /// Keep the full sliding-window cache. Not needed for branch isolation
        /// on Gemma 4 (verified: same numbers either way); costs memory.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        swa_full: bool,
    },
    /// Print a GGUF metadata value (e.g. tokenizer.chat_template).
    Meta {
        #[arg(long)]
        model: PathBuf,
        key: String,
    },
    /// Print the surface form of token ids (debugging the vocabulary).
    Pieces {
        #[arg(long)]
        model: PathBuf,
        ids: Vec<i32>,
    },
    /// Show how the model tokenizes a string (no BOS, specials neutralized).
    Tokens {
        #[arg(long)]
        model: PathBuf,
        text: String,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    match Cli::parse().cmd {
        Cmd::Meta { model, key } => {
            let engine = LlamaEngine::load(
                &model,
                Options {
                    n_ctx: 512,
                    n_batch: 512,
                    ..Default::default()
                },
            )?;
            println!("{}", engine.model().meta_val_str(&key)?);
        }
        Cmd::Pieces { model, ids } => {
            let engine = LlamaEngine::load(
                &model,
                Options {
                    n_ctx: 512,
                    n_batch: 512,
                    ..Default::default()
                },
            )?;
            for id in ids {
                println!("{id:>7}  {:?}", engine.piece(grande_core::Token(id)));
            }
        }
        Cmd::Tokens { model, text } => {
            let engine = LlamaEngine::load(
                &model,
                Options {
                    n_ctx: 512,
                    n_batch: 512,
                    ..Default::default()
                },
            )?;
            let toks = engine.tokenize(&text)?;
            for t in &toks {
                let piece = engine.piece(*t);
                println!("{:>7}  {:?}", t.0, piece);
            }
            println!("{} tokens", toks.len());
        }
        Cmd::Probe {
            model,
            request,
            mode,
            temperature,
            n_ctx,
            n_gpu_layers,
            swa_full,
        } => {
            let req: Request = serde_json::from_slice(&std::fs::read(&request)?)
                .with_context(|| format!("parsing {}", request.display()))?;
            let t0 = Instant::now();
            let backend = LlamaEngine::load(
                &model,
                Options {
                    n_ctx,
                    n_batch: n_ctx,
                    n_gpu_layers,
                    swa_full,
                    ..Default::default()
                },
            )?;
            eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f32());
            let name = model
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_lowercase();
            let mut engine = Engine::new(backend, Renderer::gemma_label(), Readout::Label, name);
            engine.temperature = temperature;

            match mode {
                ModeArg::Packed | ModeArg::Separate => {
                    let m = if matches!(mode, ModeArg::Packed) {
                        Mode::Packed
                    } else {
                        Mode::Separate
                    };
                    let t = Instant::now();
                    let (resp, diag) = engine.answer(&req, m)?;
                    let ms = t.elapsed().as_millis();
                    println!("{}", serde_json::to_string_pretty(&resp)?);
                    eprintln!(
                        "{:?}: {} ms, prefix {} tok, branches {:?} tok, passes {}",
                        m, ms, diag.prefix_tokens, diag.branch_tokens, diag.passes
                    );
                    for (id, m) in &diag.candidate_mass {
                        eprintln!("  candidate_mass {id}: {m:.4}");
                    }
                }
                ModeArg::Check => {
                    let orders = Default::default();
                    let t = Instant::now();
                    let (packed, dp) = engine.distributions(&req, &orders, Mode::Packed)?;
                    let packed_ms = t.elapsed().as_millis();
                    let t = Instant::now();
                    let (separate, _) = engine.distributions(&req, &orders, Mode::Separate)?;
                    let separate_ms = t.elapsed().as_millis();
                    let mut worst = 0f64;
                    for ((b, p), (_, s)) in packed.iter().zip(&separate) {
                        let d = p
                            .probs
                            .iter()
                            .zip(&s.probs)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0, f64::max);
                        worst = worst.max(d);
                        let show = |v: &[f64]| {
                            v.iter()
                                .map(|x| format!("{x:.4}"))
                                .collect::<Vec<_>>()
                                .join(" ")
                        };
                        println!(
                            "{:<24} packed [{}]  separate [{}]  Δmax {:.2e}",
                            b.id,
                            show(&p.probs),
                            show(&s.probs),
                            d
                        );
                    }
                    println!(
                        "packed {} ms ({} pass) vs separate {} ms ({} passes); max |Δp| = {:.2e}",
                        packed_ms,
                        dp.passes,
                        separate_ms,
                        packed.len(),
                        worst
                    );
                }
            }
        }
    }
    Ok(())
}
