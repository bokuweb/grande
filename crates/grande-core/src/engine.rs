//! Glue: render → tokenize/pack → evaluate → read out → typed answers.

use std::collections::HashMap;

use indexmap::IndexMap;
use serde::Serialize;

use crate::api::{Answer, Question, Request, Response, Usage};
use crate::backend::{Backend, BranchOutput, BranchTokens, Group, PrefixSource, Token, Want};
use crate::math::{argmax, confidence, expected_index};
use crate::plan::Plan;
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
    /// Requests that shared a backend pass with this one (1 = alone); see
    /// [`Engine::distributions_many`].
    pub batch: usize,
    /// Gated re-read ([`Engine::recheck`]): the questions whose first read
    /// fell below the threshold and were re-asked under the other orders.
    pub rechecked: Vec<String>,
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

pub use crate::plan::EXCLUDED_LOGIT;

/// What [`Engine::distributions`] returns: one distribution per question
/// (with its rendered branch) plus the request's diagnostics.
pub type Distributions = (Vec<(RenderedBranch, Distribution)>, Diagnostics);

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
    /// Gate the extra `orders` on need: ask each question once, and only
    /// re-ask (under the remaining orders, in a second pass) the ones whose
    /// confidence ([`crate::math::confidence`]) came back below this. What
    /// DiffusionGemma-as-Jev calls its auto policy. None = every question
    /// gets every order in the first pass.
    pub recheck: Option<f64>,
    baseline_cache: HashMap<String, Vec<f32>>,
}

/// A tokenized request ready for the backend.
#[derive(Debug, Clone)]
pub struct Packed {
    pub prefix: Vec<Token>,
    pub branches: Vec<BranchTokens>,
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
            recheck: None,
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

    /// Hidden-state rows at every mark of every branch, one packed pass,
    /// under explicit option orders. Feature extraction for training a
    /// pointer head on this backend's own numbers; no readout involved.
    pub fn hidden_rows(
        &mut self,
        req: &Request,
        orders: &IndexMap<String, Vec<usize>>,
    ) -> Result<Vec<(RenderedBranch, BranchOutput)>> {
        let rendered = self.renderer.render_with(req, orders);
        let packed = self.pack(&rendered)?;
        let outs = self
            .backend
            .evaluate(&packed.prefix, &packed.branches, Want::Hidden)?;
        Ok(rendered.branches.into_iter().zip(outs).collect())
    }

    /// Score one set of branches per job: pack, baseline, one backend pass
    /// for as many jobs as the backend's limits allow, read out. Jobs whose
    /// branches fail to pack or whose pass fails are moved to `failed` and
    /// dropped from `jobs`; the rest come back as `(job, distributions)`.
    /// `Mode::Separate` evaluates one branch per pass and never batches.
    fn score_many<'r>(
        &mut self,
        work: Vec<(usize, Vec<RenderedBranch>)>,
        jobs: &mut [Option<Job<'r>>],
        mode: Mode,
        failed: &mut Vec<(usize, crate::Error)>,
    ) -> Vec<(usize, Vec<(RenderedBranch, Distribution)>)> {
        struct Item {
            job: usize,
            branches: Vec<RenderedBranch>,
            packed: Vec<BranchTokens>,
            label_ids: Vec<Token>,
            baseline: Option<Vec<Vec<f32>>>,
            outputs: Option<Vec<BranchOutput>>,
        }
        let want: Want = self.readout.want();
        let mut done: Vec<(usize, Vec<(RenderedBranch, Distribution)>)> = Vec::new();
        let mut items: Vec<Item> = Vec::with_capacity(work.len());
        let fail = |jobs: &mut [Option<Job<'r>>], failed: &mut Vec<_>, i: usize, e| {
            jobs[i] = None;
            failed.push((i, e));
        };
        for (i, branches) in work {
            let Some(job) = jobs[i].as_mut() else {
                continue;
            };
            if branches.is_empty() {
                done.push((i, Vec::new()));
                continue;
            }
            let prepared = (|| {
                let packed = self.pack_branches(&branches)?;
                let max_k = branches.iter().map(|b| b.keys.len()).max().unwrap_or(0);
                let label_ids = match self.readout {
                    Readout::Label => Readout::label_ids(&self.backend, max_k)?,
                    Readout::Pointer(_) => Vec::new(),
                };
                job.diag
                    .branch_tokens
                    .extend(packed.iter().map(|b| b.tokens.len()));
                let baseline = match self.baseline.clone() {
                    Some(cf) => {
                        let (b, evaluated) =
                            self.baseline_logits(job.req, &branches, &packed, &label_ids, &cf)?;
                        job.diag.passes += usize::from(evaluated);
                        Some(b)
                    }
                    None => None,
                };
                Ok((packed, label_ids, baseline))
            })();
            match prepared {
                Ok((packed, label_ids, baseline)) => items.push(Item {
                    job: i,
                    branches,
                    packed,
                    label_ids,
                    baseline,
                    outputs: None,
                }),
                Err(e) => fail(jobs, failed, i, e),
            }
        }
        match mode {
            Mode::Separate => {
                for it in &mut items {
                    let job = jobs[it.job].as_mut().expect("live job");
                    let mut outs = Vec::with_capacity(it.packed.len());
                    let mut err = None;
                    for b in &it.packed {
                        match self
                            .backend
                            .evaluate(&job.prefix, std::slice::from_ref(b), want)
                        {
                            Ok(o) => outs.extend(o),
                            Err(e) => {
                                err = Some(e);
                                break;
                            }
                        }
                    }
                    job.diag.passes += it.packed.len();
                    job.diag.prefix_source = self.backend.prefix_source();
                    match err {
                        None => it.outputs = Some(outs),
                        Some(e) => fail(jobs, failed, it.job, e),
                    }
                }
            }
            Mode::Packed => {
                // Greedy chunks under the backend's limits; a lone item that
                // exceeds them still gets its own call (and its own error).
                let limits = self.backend.limits();
                let mut chunks: Vec<Vec<usize>> = Vec::new();
                let (mut toks, mut seqs, mut rows) = (0usize, 0usize, 0usize);
                for (k, it) in items.iter().enumerate() {
                    let job = jobs[it.job].as_ref().expect("live job");
                    let g = Group {
                        prefix: &job.prefix,
                        branches: &it.packed,
                    };
                    let (t, q, r) = (g.tokens(), g.sequences(), g.rows());
                    let fits = toks + t <= limits.tokens
                        && seqs + q <= limits.sequences
                        && rows + r <= limits.rows;
                    match chunks.last_mut() {
                        Some(c) if fits => {
                            c.push(k);
                            toks += t;
                            seqs += q;
                            rows += r;
                        }
                        _ => {
                            chunks.push(vec![k]);
                            toks = t;
                            seqs = q;
                            rows = r;
                        }
                    }
                }
                for chunk in chunks {
                    let groups: Vec<Group<'_>> = chunk
                        .iter()
                        .map(|&k| Group {
                            prefix: &jobs[items[k].job].as_ref().expect("live job").prefix,
                            branches: &items[k].packed,
                        })
                        .collect();
                    match self.backend.evaluate_many(&groups, want) {
                        Ok(outs) => {
                            for (&k, out) in chunk.iter().zip(outs) {
                                let job = jobs[items[k].job].as_mut().expect("live job");
                                job.diag.passes += 1;
                                job.diag.batch = job.diag.batch.max(chunk.len());
                                job.diag.prefix_source = out.prefix_source;
                                items[k].outputs = Some(out.branches);
                            }
                        }
                        Err(e) => {
                            for &k in &chunk {
                                fail(jobs, failed, items[k].job, e.clone());
                            }
                        }
                    }
                }
            }
        }
        for it in items {
            let Some(outputs) = it.outputs else { continue };
            let Some(job) = jobs[it.job].as_mut() else {
                continue;
            };
            let mut result = Vec::with_capacity(outputs.len());
            let mut err = None;
            for (i, (branch, out)) in it.branches.into_iter().zip(outputs).enumerate() {
                let mut dist =
                    match self
                        .readout
                        .distribution(&branch, &out, &it.label_ids, self.temperature)
                    {
                        Ok(d) => d,
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    };
                if let Some(m) = dist.candidate_mass {
                    let e = job
                        .diag
                        .candidate_mass
                        .entry(branch.id.clone())
                        .or_insert(m);
                    *e = e.min(m);
                }
                if let Some(b) = &it.baseline {
                    dist.calibrate(b[i].clone(), self.temperature);
                    job.diag
                        .baseline
                        .entry(branch.id.clone())
                        .or_insert_with(|| b[i].clone());
                }
                result.push((branch, dist));
            }
            match err {
                None => done.push((it.job, result)),
                Some(e) => fail(jobs, failed, it.job, e),
            }
        }
        done
    }

    /// Distributions for every question, in request order. Each question's
    /// distribution is over the options in the order `orders[qid]` gives
    /// (request order by default), whatever branches the [`Plan`] evaluated
    /// to get it: several option orders when [`Engine::orders`] > 1, and a
    /// group pass plus a finalist pass when a Choice has more options than
    /// the label readout can letter.
    pub fn distributions(
        &mut self,
        req: &Request,
        orders: &IndexMap<String, Vec<usize>>,
        mode: Mode,
    ) -> Result<Distributions> {
        self.distributions_many(&[(req, orders)], mode)
            .pop()
            .expect("one result per request")
    }

    /// [`Engine::distributions`] for several requests at once: every
    /// request's first-stage branches go to the backend together (as few
    /// [`Backend::evaluate_many`] calls as its [`Limits`] allow), then every
    /// second stage. Requests stay independent: one that fails validation,
    /// packing or its pass gets its own error and the others go on. One
    /// result per request, in order. `diag.batch` says how many requests
    /// shared a pass with it.
    pub fn distributions_many(
        &mut self,
        reqs: &[(&Request, &IndexMap<String, Vec<usize>>)],
        mode: Mode,
    ) -> Vec<Result<Distributions>> {
        let cap = match self.readout {
            Readout::Label => Some(crate::readout::LABELS.len()),
            Readout::Pointer(_) => None,
        };
        let mut failed: Vec<(usize, crate::Error)> = Vec::new();
        let mut jobs: Vec<Option<Job<'_>>> = Vec::with_capacity(reqs.len());
        for (i, (req, orders)) in reqs.iter().enumerate() {
            let planned = Plan::new(&self.renderer, req, orders, cap, self.orders, self.recheck)
                .and_then(|plan| {
                    let (prefix, _) = self.tokenize_segments(&plan.rendered.prefix)?;
                    Ok((plan, prefix))
                });
            match planned {
                Ok((plan, prefix)) => {
                    let diag = Diagnostics {
                        prefix_tokens: prefix.len(),
                        orders: self.orders.max(1),
                        batch: 1,
                        ..Default::default()
                    };
                    jobs.push(Some(Job {
                        req,
                        plan,
                        prefix,
                        diag,
                    }));
                }
                Err(e) => {
                    jobs.push(None);
                    failed.push((i, e));
                }
            }
        }
        let work: Vec<(usize, Vec<RenderedBranch>)> = jobs
            .iter()
            .enumerate()
            .filter_map(|(i, j)| j.as_ref().map(|j| (i, j.plan.first.clone())))
            .collect();
        let firsts = self.score_many(work, &mut jobs, mode, &mut failed);
        let mut first_dists: Vec<Option<Vec<Distribution>>> =
            (0..reqs.len()).map(|_| None).collect();
        let mut work = Vec::with_capacity(firsts.len());
        for (i, dists) in firsts {
            let Some(job) = jobs[i].as_mut() else {
                continue;
            };
            let first: Vec<Distribution> = dists.into_iter().map(|(_, d)| d).collect();
            let (second, finalists, rechecked) = job.plan.second(&self.renderer, job.req, &first);
            job.diag.two_stage = finalists;
            job.diag.rechecked = rechecked;
            first_dists[i] = Some(first);
            work.push((i, second));
        }
        let seconds = self.score_many(work, &mut jobs, mode, &mut failed);
        let mut out: Vec<Option<Result<Distributions>>> = (0..reqs.len()).map(|_| None).collect();
        for (i, second) in seconds {
            let Some(job) = jobs[i].take() else { continue };
            let first = first_dists[i].take().expect("first stage scored");
            let folded = job.plan.fold(first, second, self.temperature);
            let mut diag = job.diag;
            diag.order_spread = folded.order_spread;
            out[i] = Some(Ok((folded.results, diag)));
        }
        for (i, e) in failed {
            out[i] = Some(Err(e));
        }
        out.into_iter()
            .map(|r| r.expect("every request resolved"))
            .collect()
    }

    /// Full TypeSafe-shaped response.
    pub fn answer(&mut self, req: &Request, mode: Mode) -> Result<(Response, Diagnostics)> {
        self.answer_many(&[req], mode)
            .pop()
            .expect("one result per request")
    }

    /// [`Engine::answer`] for several requests in as few passes as the
    /// backend allows (see [`Engine::distributions_many`]). One result per
    /// request, in order.
    pub fn answer_many(
        &mut self,
        reqs: &[&Request],
        mode: Mode,
    ) -> Vec<Result<(Response, Diagnostics)>> {
        let none = IndexMap::new();
        let pairs: Vec<(&Request, &IndexMap<String, Vec<usize>>)> =
            reqs.iter().map(|r| (*r, &none)).collect();
        self.distributions_many(&pairs, mode)
            .into_iter()
            .zip(reqs)
            .map(|(r, req)| {
                let (dists, diag) = r?;
                let mut answers = IndexMap::with_capacity(dists.len());
                for (branch, dist) in &dists {
                    let q = &req.questions[&branch.id];
                    answers.insert(branch.id.clone(), to_answer(q, branch, dist));
                }
                let input_tokens = diag.prefix_tokens + diag.branch_tokens.iter().sum::<usize>();
                Ok((
                    Response {
                        model: self.model.clone(),
                        answers,
                        usage: Usage {
                            input_tokens: input_tokens as u64,
                            output_tokens: 0,
                        },
                    },
                    diag,
                ))
            })
            .collect()
    }
}

/// One request between the passes of [`Engine::distributions_many`].
struct Job<'r> {
    req: &'r Request,
    plan: Plan,
    prefix: Vec<Token>,
    diag: Diagnostics,
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

    /// Counts groups per `evaluate_many` call and caps a call at two
    /// sequences' worth of tokens, so three requests take two calls.
    struct Counting {
        calls: Cell<Vec<usize>>,
    }

    impl Backend for Counting {
        fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
            Ok(text.split_whitespace().map(|_| Token(3)).collect())
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
            _: &[Token],
            branches: &[BranchTokens],
            _: Want,
        ) -> Result<Vec<BranchOutput>> {
            Ok(branches
                .iter()
                .map(|_| BranchOutput {
                    rows: vec![vec![0.0, 3.0, 0.0, 0.0]],
                })
                .collect())
        }
        fn evaluate_many(
            &mut self,
            groups: &[Group<'_>],
            want: Want,
        ) -> Result<Vec<crate::backend::GroupOutput>> {
            let mut c = self.calls.take();
            c.push(groups.len());
            self.calls.set(c);
            let mut out = Vec::new();
            for g in groups {
                out.push(crate::backend::GroupOutput {
                    branches: self.evaluate(g.prefix, g.branches, want)?,
                    prefix_source: None,
                });
            }
            Ok(out)
        }
        fn limits(&self) -> crate::backend::Limits {
            crate::backend::Limits {
                sequences: 4,
                ..Default::default()
            }
        }
    }

    #[test]
    fn many_requests_share_passes_and_fail_alone() {
        let mut engine = Engine::new(
            Counting {
                calls: Cell::new(Vec::new()),
            },
            Renderer::gemma_label(),
            Readout::Label,
            "t",
        );
        let good = request("fine");
        // A choice with no options fails validation on its own without
        // touching the others.
        let bad: Request = serde_json::from_value(serde_json::json!({
            "state": "x",
            "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {}}}
        }))
        .unwrap();
        let empty: Request = serde_json::from_value(serde_json::json!({
            "state": "x", "questions": {}
        }))
        .unwrap();
        let results = engine.answer_many(&[&good, &bad, &good, &empty, &good], Mode::Packed);
        assert!(results[0].is_ok());
        assert!(
            matches!(&results[1], Err(crate::Error::Invalid { .. })),
            "{:?}",
            results[1].as_ref().err()
        );
        assert!(results[2].is_ok());
        assert!(results[3].is_err());
        assert!(results[4].is_ok());
        // Each good request is 2 sequences; the limit of 4 puts two in the
        // first call and one in the second.
        assert_eq!(engine.backend.calls.take(), vec![2, 1]);
        let (_, d0) = results[0].as_ref().unwrap();
        let (_, d4) = results[4].as_ref().unwrap();
        assert_eq!(d0.batch, 2);
        assert_eq!(d4.batch, 1);
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

    /// Position bias only: label A gets +1 unless an option is "gold", which
    /// gets +6 wherever it sits.
    struct Pulled;

    impl Backend for Pulled {
        fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
            Lettered.tokenize(text)
        }
        fn special(&self, n: &str) -> Result<Token> {
            Lettered.special(n)
        }
        fn bos(&self) -> Token {
            Lettered.bos()
        }
        fn n_embd(&self) -> usize {
            1
        }
        fn n_vocab(&self) -> usize {
            Lettered.n_vocab()
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
                    row[1] = 1.0;
                    let mut label = None;
                    for t in &b.tokens {
                        if (1..=52).contains(&t.0) {
                            label = Some(t.0 as usize);
                        } else if t.0 == GOLD {
                            if let Some(l) = label {
                                row[l] += 6.0;
                            }
                        }
                    }
                    BranchOutput { rows: vec![row] }
                })
                .collect())
        }
    }

    #[test]
    fn recheck_re_asks_only_the_uncertain_question() {
        let req: Request = serde_json::from_value(serde_json::json!({
            "state": "s",
            "questions": {
                "sure": {"type": "choice", "instructions": "which", "criteria": {"x": null, "gold": null, "y": null}},
                "unsure": {"type": "choice", "instructions": "which", "criteria": {"p": null, "q": null, "r": null}}
            }
        }))
        .unwrap();
        let mut engine = Engine::new(Pulled, Renderer::gemma_label(), Readout::Label, "t");
        engine.orders = 3;
        // Ungated: both questions get all three orders in one pass.
        let (_, diag) = engine.answer(&req, Mode::Packed).unwrap();
        assert_eq!(diag.passes, 1);
        assert_eq!(diag.branch_tokens.len(), 6);
        assert!(diag.rechecked.is_empty());

        engine.recheck = Some(0.5);
        let (resp, diag) = engine.answer(&req, Mode::Packed).unwrap();
        assert_eq!(diag.rechecked, vec!["unsure".to_string()]);
        // First pass 2 branches, second pass the two other orders of "unsure".
        assert_eq!(diag.passes, 2);
        assert_eq!(diag.branch_tokens.len(), 4);
        let Answer::Choice { choice, .. } = &resp.answers["sure"] else {
            panic!()
        };
        assert_eq!(choice, "gold");
        // Averaged over the three rotations, the pull of A cancels: uniform.
        let Answer::Choice { probabilities, .. } = &resp.answers["unsure"] else {
            panic!()
        };
        for p in probabilities.values() {
            assert!((p - 1.0 / 3.0).abs() < 1e-6, "{probabilities:?}");
        }
        assert!(diag.order_spread["unsure"] > 0.3);

        // A threshold nothing falls under: one pass, one branch each.
        engine.recheck = Some(0.0);
        let (_, diag) = engine.answer(&req, Mode::Packed).unwrap();
        assert_eq!(diag.passes, 1);
        assert_eq!(diag.branch_tokens.len(), 2);
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
