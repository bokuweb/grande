//! Glue: render → tokenize/pack → evaluate → read out → typed answers.

use std::collections::HashMap;

use indexmap::IndexMap;
use serde::Serialize;

use crate::api::{Answer, Question, Request, Response, Usage};
use crate::backend::{Backend, BranchTokens, PrefixSource, Token, Want};
use crate::math::{argmax, confidence, expected_index, option_orders, softmax};
use crate::readout::{Distribution, Readout};
use crate::render::{Kind, Mark, Rendered, RenderedBranch, Renderer, Segment};
use crate::Result;

/// Per-request diagnostics, surfaced as headers by a server or printed by the CLI.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Diagnostics {
    pub prefix_tokens: usize,
    /// Every branch that was evaluated, including extra option orders and
    /// the group / finalist branches of a two-stage Choice.
    pub branch_tokens: Vec<usize>,
    /// Label readout only.
    pub candidate_mass: IndexMap<String, f64>,
    /// Number of `evaluate` calls (1 when packed; +1 for a baseline pass
    /// that was not served from the cache; +1 for a two-stage Choice).
    pub passes: usize,
    /// Where the backend got the state from (resident / ram / disk / decoded).
    pub prefix_source: Option<PrefixSource>,
    /// Contextual calibration only: per question, the option logits over
    /// the content-free state that were subtracted.
    pub baseline: IndexMap<String, Vec<f32>>,
    /// Option orders averaged per Choice / Noul (see [`Engine::orders`]).
    pub orders: usize,
    /// Per question asked under more than one order: the largest |Δp| any
    /// option showed between two orders — the position bias that was
    /// averaged out.
    pub order_spread: IndexMap<String, f64>,
    /// Choices with more options than the label readout can letter: the
    /// finalist keys that went into the second stage.
    pub two_stage: IndexMap<String, Vec<String>>,
}

/// How branches are evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Every branch in one `evaluate` call: state once, one pass.
    Packed,
    /// One `evaluate` call per branch, re-evaluating the prefix each time.
    /// Only for checking that packing does not change the numbers.
    Separate,
}

/// Logit given to an option that did not reach the second stage of a
/// two-stage Choice: zero probability at any temperature, still finite so
/// rows serialize and a refit sees a number.
pub const EXCLUDED_LOGIT: f32 = -1.0e4;

pub struct Engine<B: Backend> {
    pub backend: B,
    pub renderer: Renderer,
    pub readout: Readout,
    pub temperature: f32,
    pub model: String,
    /// Contextual calibration: when set, every question is also asked over
    /// this content-free state and the option logits it yields are
    /// subtracted from the live ones (see [`crate::calibration::contextual`]).
    /// The baseline depends on the question alone, so it is cached per
    /// rendered branch: a fixed question set over changing states pays for
    /// it once.
    pub baseline: Option<String>,
    /// Ask every Choice and Noul under this many option orders (rotations,
    /// then reversals; see [`crate::math::option_orders`]) in the same pass
    /// and average the option logits. Removes position bias — the pull of
    /// letter A, the lean to the first-listed option — at the cost of extra
    /// branch tokens; the state is read once regardless. 1 = off.
    pub orders: usize,
    baseline_cache: HashMap<String, Vec<f32>>,
}

/// A tokenized request ready for the backend.
#[derive(Debug, Clone)]
pub struct Packed {
    pub prefix: Vec<Token>,
    pub branches: Vec<BranchTokens>,
}

/// One evaluated branch: which option (original index) sat in each slot,
/// and what the readout made of it.
struct Scored {
    order: Vec<usize>,
    dist: Distribution,
}

impl<B: Backend> Engine<B> {
    pub fn new(backend: B, renderer: Renderer, readout: Readout, model: impl Into<String>) -> Self {
        Engine {
            backend,
            renderer,
            readout,
            temperature: 1.0,
            model: model.into(),
            baseline: None,
            orders: 1,
            baseline_cache: HashMap::new(),
        }
    }

    /// Option logits for every branch over the content-free state `cf`,
    /// from the cache where possible. Returns whether the backend was called.
    fn baseline_logits(
        &mut self,
        req: &Request,
        branches: &[RenderedBranch],
        packed: &[BranchTokens],
        label_ids: &[Token],
        cf: &str,
    ) -> Result<(Vec<Vec<f32>>, bool)> {
        let keys: Vec<String> = branches
            .iter()
            .map(|b| serde_json::to_string(&(cf, &b.segments, &b.keys)).unwrap_or_default())
            .collect();
        let missing: Vec<usize> = (0..keys.len())
            .filter(|&i| !self.baseline_cache.contains_key(&keys[i]))
            .collect();
        let evaluated = !missing.is_empty();
        if evaluated {
            let mut cf_req = req.clone();
            cf_req.state = serde_json::Value::String(cf.to_string());
            let cf_prefix = self.renderer.render(&cf_req).prefix;
            let (cf_prefix, _) = self.tokenize_segments(&cf_prefix)?;
            let wanted: Vec<BranchTokens> = missing.iter().map(|&i| packed[i].clone()).collect();
            let outs = self
                .backend
                .evaluate(&cf_prefix, &wanted, self.readout.want())?;
            for (&i, out) in missing.iter().zip(outs) {
                let dist = self
                    .readout
                    .distribution(&branches[i], &out, label_ids, 1.0)?;
                self.baseline_cache.insert(keys[i].clone(), dist.logits);
            }
        }
        Ok((
            keys.iter()
                .map(|k| self.baseline_cache[k].clone())
                .collect(),
            evaluated,
        ))
    }

    fn tokenize_segments(&self, segments: &[Segment]) -> Result<(Vec<Token>, Vec<usize>)> {
        // Returns tokens and, per segment, the index of its last token.
        let mut tokens = Vec::new();
        let mut ends = Vec::with_capacity(segments.len());
        for s in segments {
            match s {
                Segment::Bos => tokens.push(self.backend.bos()),
                Segment::Special(name) => tokens.push(self.backend.special(name)?),
                Segment::Text(t) => tokens.extend(self.backend.tokenize(t)?),
            }
            ends.push(tokens.len().saturating_sub(1));
        }
        Ok((tokens, ends))
    }

    pub fn pack(&self, rendered: &Rendered) -> Result<Packed> {
        let (prefix, _) = self.tokenize_segments(&rendered.prefix)?;
        Ok(Packed {
            prefix,
            branches: self.pack_branches(&rendered.branches)?,
        })
    }

    fn pack_branches(&self, branches: &[RenderedBranch]) -> Result<Vec<BranchTokens>> {
        let mut out = Vec::with_capacity(branches.len());
        for b in branches {
            let (tokens, ends) = self.tokenize_segments(&b.segments)?;
            let want = b
                .marks
                .iter()
                .map(|(seg, mark)| match mark {
                    Mark::Last => tokens.len() - 1,
                    _ => ends[*seg],
                })
                .collect();
            out.push(BranchTokens { tokens, want });
        }
        Ok(out)
    }

    /// Evaluate `branches` over the prefix (one pass, or one per branch)
    /// and read every one out. The baseline pass, when configured, goes
    /// first so the live state is what stays resident in the backend.
    fn score_branches(
        &mut self,
        req: &Request,
        prefix: &[Token],
        branches: Vec<RenderedBranch>,
        mode: Mode,
        diag: &mut Diagnostics,
    ) -> Result<Vec<(RenderedBranch, Distribution)>> {
        if branches.is_empty() {
            return Ok(Vec::new());
        }
        let packed = self.pack_branches(&branches)?;
        let max_k = branches.iter().map(|b| b.keys.len()).max().unwrap_or(0);
        let label_ids = match self.readout {
            Readout::Label => Readout::label_ids(&self.backend, max_k)?,
            Readout::Pointer(_) => Vec::new(),
        };
        let want: Want = self.readout.want();
        diag.branch_tokens
            .extend(packed.iter().map(|b| b.tokens.len()));
        let baseline = match self.baseline.clone() {
            Some(cf) => {
                let (b, evaluated) =
                    self.baseline_logits(req, &branches, &packed, &label_ids, &cf)?;
                diag.passes += usize::from(evaluated);
                Some(b)
            }
            None => None,
        };
        let outputs = match mode {
            Mode::Packed => {
                diag.passes += 1;
                self.backend.evaluate(prefix, &packed, want)?
            }
            Mode::Separate => {
                let mut outs = Vec::with_capacity(packed.len());
                for b in &packed {
                    outs.extend(
                        self.backend
                            .evaluate(prefix, std::slice::from_ref(b), want)?,
                    );
                }
                diag.passes += packed.len();
                outs
            }
        };
        diag.prefix_source = self.backend.prefix_source();
        let mut result = Vec::with_capacity(outputs.len());
        for (i, (branch, out)) in branches.into_iter().zip(outputs).enumerate() {
            let mut dist =
                self.readout
                    .distribution(&branch, &out, &label_ids, self.temperature)?;
            if let Some(m) = dist.candidate_mass {
                let e = diag.candidate_mass.entry(branch.id.clone()).or_insert(m);
                *e = e.min(m);
            }
            if let Some(b) = &baseline {
                dist.calibrate(b[i].clone(), self.temperature);
                diag.baseline
                    .entry(branch.id.clone())
                    .or_insert_with(|| b[i].clone());
            }
            result.push((branch, dist));
        }
        Ok(result)
    }

    /// Distributions for every question, in request order. Each question's
    /// distribution is over the options in the order `orders[qid]` gives
    /// (request order by default), whatever branches were evaluated to get
    /// it: several option orders when [`Engine::orders`] > 1, and a group
    /// pass plus a finalist pass when a Choice has more options than the
    /// label readout can letter.
    pub fn distributions(
        &mut self,
        req: &Request,
        orders: &IndexMap<String, Vec<usize>>,
        mode: Mode,
    ) -> Result<(Vec<(RenderedBranch, Distribution)>, Diagnostics)> {
        req.validate()?;
        let rendered = self.renderer.render_with(req, orders);
        let (prefix, _) = self.tokenize_segments(&rendered.prefix)?;
        let mut diag = Diagnostics {
            prefix_tokens: prefix.len(),
            orders: self.orders.max(1),
            ..Default::default()
        };
        let cap = match self.readout {
            Readout::Label => Some(crate::readout::LABELS.len()),
            Readout::Pointer(_) => None,
        };
        let orders_wanted = self.orders.max(1);
        let n_orders = move |kind: Kind| match kind {
            Kind::Choice | Kind::Noul => orders_wanted,
            // Levels are ordered; the model must see them in order.
            Kind::Score => 1,
        };

        // Pass 1: every question under its orders, except Choices over the
        // label cap, which go in as groups of at most `cap` options.
        let mut first: Vec<RenderedBranch> = Vec::new();
        let mut grouped: Vec<usize> = Vec::new();
        for (qi, (id, q)) in req.questions.iter().enumerate() {
            let primary = &rendered.branches[qi];
            let k = primary.order.len();
            match cap {
                Some(cap) if k > cap && primary.kind == Kind::Choice => {
                    let groups = k.div_ceil(cap);
                    for chunk in primary.order.chunks(k.div_ceil(groups)) {
                        first.push(self.renderer.branch(id, q, Some(chunk)));
                    }
                    grouped.push(qi);
                }
                Some(cap) if k > cap => {
                    return Err(crate::Error::invalid(
                        format!("questions.{id}.criteria"),
                        format!("label readout supports at most {cap} levels"),
                    ));
                }
                _ => {
                    for o in option_orders(k, n_orders(primary.kind)) {
                        let order: Vec<usize> = o.iter().map(|&s| primary.order[s]).collect();
                        first.push(self.renderer.branch(id, q, Some(&order)));
                    }
                }
            }
        }
        let mut scored: Vec<Vec<Scored>> =
            (0..rendered.branches.len()).map(|_| Vec::new()).collect();
        let index_of = |id: &str| req.questions.get_index_of(id).expect("rendered id");
        for (b, d) in self.score_branches(req, &prefix, first, mode, &mut diag)? {
            scored[index_of(&b.id)].push(Scored {
                order: b.order,
                dist: d,
            });
        }

        // Pass 2: the finalists of every grouped Choice — the top options of
        // each group, as many as fit under the cap together — asked once
        // more, against each other, under the configured orders.
        if !grouped.is_empty() {
            let mut second: Vec<RenderedBranch> = Vec::new();
            for &qi in &grouped {
                let (id, q) = req.questions.get_index(qi).expect("question");
                let groups = std::mem::take(&mut scored[qi]);
                let per_group = cap.expect("label readout") / groups.len();
                let mut finalists: Vec<usize> = Vec::new();
                for g in &groups {
                    let mut ranked: Vec<(usize, f64)> = g
                        .order
                        .iter()
                        .copied()
                        .zip(g.dist.probs.iter().copied())
                        .collect();
                    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                    finalists.extend(ranked.iter().take(per_group).map(|x| x.0));
                }
                // Keep the caller's relative order among the finalists.
                let primary = &rendered.branches[qi];
                let slot_of =
                    |o: usize| primary.order.iter().position(|&x| x == o).expect("option");
                finalists.sort_by_key(|&o| slot_of(o));
                diag.two_stage.insert(
                    id.clone(),
                    finalists
                        .iter()
                        .map(|&o| primary.keys[slot_of(o)].clone())
                        .collect(),
                );
                for o in option_orders(finalists.len(), n_orders(primary.kind)) {
                    let order: Vec<usize> = o.iter().map(|&s| finalists[s]).collect();
                    second.push(self.renderer.branch(id, q, Some(&order)));
                }
            }
            for (b, d) in self.score_branches(req, &prefix, second, mode, &mut diag)? {
                scored[index_of(&b.id)].push(Scored {
                    order: b.order,
                    dist: d,
                });
            }
        }

        // Fold every question's branches into one distribution in the
        // primary branch's slot order: mean logit per option over the orders
        // it was asked in, options that were never asked at EXCLUDED_LOGIT.
        let mut result = Vec::with_capacity(rendered.branches.len());
        for (qi, primary) in rendered.branches.into_iter().enumerate() {
            let k = primary.order.len();
            let runs = &scored[qi];
            let mut sum = vec![0.0f64; k];
            let mut base_sum = vec![0.0f64; k];
            let mut count = vec![0usize; k];
            let mut per_run_probs: Vec<Vec<f64>> = Vec::with_capacity(runs.len());
            let mut has_baseline = !runs.is_empty();
            let mut candidate_mass: Option<f64> = None;
            for r in runs {
                let mut p = vec![0.0; k];
                for (slot, &orig) in r.order.iter().enumerate() {
                    sum[orig] += f64::from(r.dist.logits[slot]);
                    count[orig] += 1;
                    p[orig] = r.dist.probs[slot];
                    match &r.dist.baseline {
                        Some(b) => base_sum[orig] += f64::from(b[slot]),
                        None => has_baseline = false,
                    }
                }
                per_run_probs.push(p);
                if let Some(m) = r.dist.candidate_mass {
                    candidate_mass = Some(candidate_mass.map_or(m, |c: f64| c.min(m)));
                }
            }
            let mean = |s: &[f64], orig: usize| -> f32 {
                if count[orig] == 0 {
                    EXCLUDED_LOGIT
                } else {
                    (s[orig] / count[orig] as f64) as f32
                }
            };
            let logits: Vec<f32> = primary.order.iter().map(|&o| mean(&sum, o)).collect();
            let baseline =
                has_baseline.then(|| primary.order.iter().map(|&o| mean(&base_sum, o)).collect());
            if per_run_probs.len() > 1 {
                let mut spread = 0.0f64;
                for (a, pa) in per_run_probs.iter().enumerate() {
                    for pb in &per_run_probs[a + 1..] {
                        for (x, y) in pa.iter().zip(pb) {
                            spread = spread.max((x - y).abs());
                        }
                    }
                }
                diag.order_spread.insert(primary.id.clone(), spread);
            }
            let probs = softmax(&logits, self.temperature);
            result.push((
                primary,
                Distribution {
                    logits,
                    baseline,
                    probs,
                    candidate_mass,
                },
            ));
        }
        Ok((result, diag))
    }

    /// Full TypeSafe-shaped response.
    pub fn answer(&mut self, req: &Request, mode: Mode) -> Result<(Response, Diagnostics)> {
        let (dists, diag) = self.distributions(req, &IndexMap::new(), mode)?;
        let mut answers = IndexMap::with_capacity(dists.len());
        for (branch, dist) in &dists {
            let q = &req.questions[&branch.id];
            answers.insert(branch.id.clone(), to_answer(q, branch, dist));
        }
        let input_tokens = diag.prefix_tokens + diag.branch_tokens.iter().sum::<usize>();
        let response = Response {
            model: self.model.clone(),
            answers,
            usage: Usage {
                input_tokens: input_tokens as u64,
                output_tokens: 0,
            },
        };
        Ok((response, diag))
    }
}

/// Map a distribution over rendered options back onto the question's own keys.
pub fn to_answer(q: &Question, branch: &RenderedBranch, dist: &Distribution) -> Answer {
    // p_orig[i] = probability of the option that was at original index i.
    let k = branch.keys.len();
    let mut p_orig = vec![0.0; k];
    for (slot, &orig) in branch.order.iter().enumerate() {
        p_orig[orig] = dist.probs[slot];
    }
    match (q, branch.kind) {
        (Question::Noul { .. }, Kind::Noul) => Answer::Noul { noul: p_orig[0] },
        (Question::Choice { criteria, .. }, Kind::Choice) => {
            let probabilities: IndexMap<String, f64> = criteria
                .keys()
                .cloned()
                .zip(p_orig.iter().copied())
                .collect();
            let best = argmax(&p_orig);
            Answer::Choice {
                choice: criteria.keys().nth(best).cloned().unwrap_or_default(),
                probabilities,
                confidence: confidence(&p_orig),
            }
        }
        (Question::Score { criteria, .. }, Kind::Score) => {
            let legend = criteria
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), v.clone()))
                .collect();
            let probabilities = p_orig
                .iter()
                .enumerate()
                .map(|(i, &p)| (i.to_string(), p))
                .collect();
            Answer::Score {
                score: expected_index(&p_orig),
                legend,
                probabilities,
                confidence: confidence(&p_orig),
            }
        }
        _ => unreachable!("renderer kind must match the question type"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BranchOutput;
    use std::cell::Cell;

    /// A backend with a fixed "yes" lean: every label-readout row puts label
    /// A 3 logits above B, plus a bonus for A when the prefix contains the
    /// evidence token. Counts `evaluate` calls.
    struct Leaning {
        calls: Cell<usize>,
    }

    const EVIDENCE: i32 = 999;

    impl Backend for Leaning {
        fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
            Ok(text
                .split_whitespace()
                .map(|w| {
                    Token(match w {
                        "A" => 1,
                        "B" => 2,
                        "evidence" => EVIDENCE,
                        _ => 3,
                    })
                })
                .collect())
        }
        fn special(&self, _: &str) -> Result<Token> {
            Ok(Token(2))
        }
        fn bos(&self) -> Token {
            Token(0)
        }
        fn n_embd(&self) -> usize {
            1
        }
        fn n_vocab(&self) -> usize {
            4
        }
        fn evaluate(
            &mut self,
            prefix: &[Token],
            branches: &[BranchTokens],
            _: Want,
        ) -> Result<Vec<BranchOutput>> {
            self.calls.set(self.calls.get() + 1);
            let seen = prefix.iter().any(|t| t.0 == EVIDENCE);
            Ok(branches
                .iter()
                .map(|_| BranchOutput {
                    // row[1] = label A, row[2] = label B (see `tokenize`).
                    rows: vec![vec![0.0, 3.0 + if seen { 6.0 } else { 0.0 }, 0.0, 0.0]],
                })
                .collect())
        }
    }

    fn request(state: &str) -> Request {
        serde_json::from_value(serde_json::json!({
            "state": state,
            "questions": {"q": {"type": "noul", "instructions": "is it so"}}
        }))
        .unwrap()
    }

    #[test]
    fn baseline_removes_the_lean_and_is_cached() {
        let backend = Leaning {
            calls: Cell::new(0),
        };
        let mut engine = Engine::new(backend, Renderer::gemma_label(), Readout::Label, "t");
        // Uncalibrated: the lean reads as 95% yes with nothing in the state.
        let (d, _) = engine
            .distributions(&request("nothing here"), &IndexMap::new(), Mode::Packed)
            .unwrap();
        assert!(d[0].1.probs[0] > 0.9);

        engine.baseline = Some("N/A".into());
        let (d, diag) = engine
            .distributions(&request("nothing here"), &IndexMap::new(), Mode::Packed)
            .unwrap();
        assert!((d[0].1.probs[0] - 0.5).abs() < 1e-9, "{:?}", d[0].1.probs);
        assert_eq!(diag.passes, 2);
        assert_eq!(diag.baseline["q"], vec![3.0, 0.0]);

        // Same question over a state with evidence: the baseline is served
        // from the cache (one pass) and the evidence survives calibration.
        let calls = engine.backend.calls.get();
        let (d, diag) = engine
            .distributions(&request("the evidence"), &IndexMap::new(), Mode::Packed)
            .unwrap();
        assert_eq!(engine.backend.calls.get(), calls + 1);
        assert_eq!(diag.passes, 1);
        assert!(d[0].1.probs[0] > 0.99);
    }

    #[test]
    fn averaging_orders_cancels_the_letter_pull() {
        let backend = Leaning {
            calls: Cell::new(0),
        };
        let mut engine = Engine::new(backend, Renderer::gemma_label(), Readout::Label, "t");
        engine.orders = 2;
        let (d, diag) = engine
            .distributions(&request("nothing here"), &IndexMap::new(), Mode::Packed)
            .unwrap();
        // A is 3 logits ahead in both orders, so true and false each get
        // (3 + 0) / 2 and the lean is gone; one pass, two branches.
        assert!((d[0].1.probs[0] - 0.5).abs() < 1e-9, "{:?}", d[0].1.probs);
        assert_eq!(d[0].1.logits, vec![1.5, 1.5]);
        assert_eq!(diag.passes, 1);
        assert_eq!(diag.branch_tokens.len(), 2);
        assert!(diag.order_spread["q"] > 0.9);
    }

    /// Label readout backend for the two-stage test: every label letter is
    /// its own token, and a branch scores 5 on the label that introduces
    /// the option named `gold`, 0 elsewhere.
    struct Lettered;

    const GOLD: i32 = 999;

    impl Backend for Lettered {
        fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
            Ok(text
                .split_whitespace()
                .map(|w| {
                    let w = w.trim_end_matches(':');
                    let mut c = w.chars();
                    match (c.next(), c.next()) {
                        (Some(ch), None) if ch.is_ascii_alphabetic() => Token(
                            1 + crate::readout::LABELS
                                .iter()
                                .position(|&l| l == ch)
                                .unwrap() as i32,
                        ),
                        _ if w == "gold" => Token(GOLD),
                        _ => Token(GOLD + 1),
                    }
                })
                .collect())
        }
        fn special(&self, _: &str) -> Result<Token> {
            Ok(Token(GOLD + 2))
        }
        fn bos(&self) -> Token {
            Token(0)
        }
        fn n_embd(&self) -> usize {
            1
        }
        fn n_vocab(&self) -> usize {
            (GOLD + 3) as usize
        }
        fn evaluate(
            &mut self,
            _: &[Token],
            branches: &[BranchTokens],
            _: Want,
        ) -> Result<Vec<BranchOutput>> {
            Ok(branches
                .iter()
                .map(|b| {
                    let mut row = vec![0.0; self.n_vocab()];
                    let mut label = None;
                    for t in &b.tokens {
                        if (1..=52).contains(&t.0) {
                            label = Some(t.0 as usize);
                        } else if t.0 == GOLD {
                            if let Some(l) = label {
                                row[l] = 5.0;
                            }
                        }
                    }
                    BranchOutput { rows: vec![row] }
                })
                .collect())
        }
    }

    #[test]
    fn choice_over_the_label_cap_runs_two_stages() {
        let mut criteria = serde_json::Map::new();
        for i in 0..60 {
            let key = if i == 45 {
                "gold".to_string()
            } else {
                format!("k{i}")
            };
            criteria.insert(key, serde_json::Value::Null);
        }
        let req: Request = serde_json::from_value(serde_json::json!({
            "state": "s",
            "questions": {"q": {"type": "choice", "instructions": "which", "criteria": criteria}}
        }))
        .unwrap();
        let mut engine = Engine::new(Lettered, Renderer::gemma_label(), Readout::Label, "t");
        let (resp, diag) = engine.answer(&req, Mode::Packed).unwrap();
        assert_eq!(diag.passes, 2);
        // Two groups of 30, 26 finalists from each, one finalist branch.
        assert_eq!(diag.branch_tokens.len(), 3);
        let finalists = &diag.two_stage["q"];
        assert_eq!(finalists.len(), 52);
        assert!(finalists.contains(&"gold".to_string()));
        let Answer::Choice {
            choice,
            probabilities,
            ..
        } = &resp.answers["q"]
        else {
            panic!()
        };
        assert_eq!(choice, "gold");
        assert_eq!(probabilities.len(), 60);
        assert!(probabilities["gold"] > 0.7);
        let excluded = probabilities
            .iter()
            .filter(|(k, _)| !finalists.contains(k))
            .count();
        assert_eq!(excluded, 8);
        assert!(probabilities
            .iter()
            .filter(|(k, _)| !finalists.contains(k))
            .all(|(_, p)| *p == 0.0));
        let total: f64 = probabilities.values().sum();
        assert!((total - 1.0).abs() < 1e-9);
    }
}
