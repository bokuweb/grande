//! Which branches to evaluate for a request and how to fold what comes back
//! into one distribution per question. Backend-agnostic on purpose: the
//! native engine and the browser (through the wasm surface) run this same
//! code, so an option order or a two-stage Choice is decided in one place.
//!
//! Two things make a request need more than one branch per question:
//!
//! - **Order averaging.** A Choice or Noul is asked under several option
//!   orders ([`crate::math::option_orders`]) and the option logits are
//!   averaged, which removes position bias (the pull of letter A, the lean
//!   to the first-listed option). Score levels are ordered and never
//!   permuted.
//! - **Two-stage Choice.** With a label readout at most `cap` (52) options
//!   can be lettered. A larger Choice is asked in groups of at most `cap`
//!   in the first pass; the top options of every group — as many as fit
//!   under `cap` together — are asked once more against each other in a
//!   second pass. The answer is the second pass's distribution with the
//!   eliminated options at zero.
//! - **Gated re-read.** With `recheck` set, the first pass asks each
//!   question once, and only the questions whose answer came back with a
//!   confidence below the threshold are re-asked under the remaining
//!   orders in the second pass (the policy of DiffusionGemma-as-Jev: a
//!   second sample only when the first read is uncertain). A confident
//!   answer costs one branch; an uncertain one gets the full averaging.

use indexmap::IndexMap;

use crate::api::Request;
use crate::math::{option_orders, softmax};
use crate::readout::Distribution;
use crate::render::{Kind, Rendered, RenderedBranch, Renderer};
use crate::Result;

/// Logit given to an option that did not reach the second stage of a
/// two-stage Choice: zero probability at any temperature, still finite so
/// rows serialize and a refit sees a number.
pub const EXCLUDED_LOGIT: f32 = -1.0e4;

#[derive(Debug, Clone)]
pub struct Plan {
    /// The prefix and one branch per question in the caller's option order.
    /// Every folded distribution is over that branch's slots.
    pub rendered: Rendered,
    /// Branches of the first pass, each carrying its question's id.
    pub first: Vec<RenderedBranch>,
    /// Questions (index into `rendered.branches`) that went in as groups.
    pub grouped: Vec<usize>,
    cap: Option<usize>,
    orders: usize,
    /// Re-ask a question under the other orders only when its first read's
    /// confidence is below this.
    recheck: Option<f64>,
}

/// Distributions per question after folding, plus what was averaged out.
#[derive(Debug, Clone)]
pub struct Folded {
    pub results: Vec<(RenderedBranch, Distribution)>,
    /// Per question asked under more than one order: the largest |Δp| any
    /// option showed between two orders.
    pub order_spread: IndexMap<String, f64>,
}

impl Plan {
    /// `cap` is the label readout's option limit (`None` for the pointer
    /// readout); `orders` how many option orders to average (1 = off);
    /// `recheck` gates the extra orders on the first read's confidence
    /// ([`crate::math::confidence`]) being below the value.
    pub fn new(
        renderer: &Renderer,
        req: &Request,
        orders: &IndexMap<String, Vec<usize>>,
        cap: Option<usize>,
        n_orders: usize,
        recheck: Option<f64>,
    ) -> Result<Plan> {
        req.validate()?;
        let rendered = renderer.render_with(req, orders);
        let n_orders = n_orders.max(1);
        // Gated: one order in the first pass, the rest on demand.
        let first_orders = if recheck.is_some() && n_orders > 1 {
            1
        } else {
            n_orders
        };
        let mut first = Vec::new();
        let mut grouped = Vec::new();
        for (qi, (id, q)) in req.questions.iter().enumerate() {
            let primary = &rendered.branches[qi];
            let k = primary.order.len();
            match cap {
                Some(cap) if k > cap && primary.kind == Kind::Choice => {
                    let groups = k.div_ceil(cap);
                    for chunk in primary.order.chunks(k.div_ceil(groups)) {
                        first.push(renderer.branch(id, q, Some(chunk)));
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
                    for o in option_orders(k, orders_for(primary.kind, first_orders)) {
                        let order: Vec<usize> = o.iter().map(|&s| primary.order[s]).collect();
                        first.push(renderer.branch(id, q, Some(&order)));
                    }
                }
            }
        }
        Ok(Plan {
            rendered,
            first,
            grouped,
            cap,
            orders: n_orders,
            recheck,
        })
    }

    fn question_index(&self, id: &str) -> usize {
        self.rendered
            .branches
            .iter()
            .position(|b| b.id == id)
            .expect("branch id is a question id")
    }

    /// The second pass, given one distribution per `first` branch: the
    /// finalist branches of every grouped Choice (empty when nothing was
    /// grouped) and, per such question, the finalists' keys; plus, with
    /// `recheck`, the remaining orders of every question whose first read
    /// was not confident enough, and those questions' ids.
    pub fn second(
        &self,
        renderer: &Renderer,
        req: &Request,
        first: &[Distribution],
    ) -> (
        Vec<RenderedBranch>,
        IndexMap<String, Vec<String>>,
        Vec<String>,
    ) {
        let mut branches = Vec::new();
        let mut finalists_of = IndexMap::new();
        let mut rechecked = Vec::new();
        if let Some(threshold) = self.recheck {
            if self.orders > 1 {
                for (qi, (id, q)) in req.questions.iter().enumerate() {
                    if self.grouped.contains(&qi) {
                        continue;
                    }
                    let primary = &self.rendered.branches[qi];
                    let k = primary.order.len();
                    if orders_for(primary.kind, self.orders) < 2 {
                        continue;
                    }
                    let (_, d) = self
                        .first
                        .iter()
                        .zip(first)
                        .find(|(b, _)| b.id == *id)
                        .expect("every question has a first branch");
                    if crate::math::confidence(&d.probs) >= threshold {
                        continue;
                    }
                    rechecked.push(id.clone());
                    // The first pass asked the natural order; add the rest.
                    for o in option_orders(k, self.orders).into_iter().skip(1) {
                        let order: Vec<usize> = o.iter().map(|&s| primary.order[s]).collect();
                        branches.push(renderer.branch(id, q, Some(&order)));
                    }
                }
            }
        }
        for &qi in &self.grouped {
            let (id, q) = req.questions.get_index(qi).expect("question");
            let primary = &self.rendered.branches[qi];
            let groups: Vec<(&RenderedBranch, &Distribution)> = self
                .first
                .iter()
                .zip(first)
                .filter(|(b, _)| b.id == *id)
                .collect();
            let per_group = self.cap.expect("grouped means a label cap") / groups.len();
            let mut finalists: Vec<usize> = Vec::new();
            for (b, d) in &groups {
                let mut ranked: Vec<(usize, f64)> = b
                    .order
                    .iter()
                    .copied()
                    .zip(d.probs.iter().copied())
                    .collect();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                finalists.extend(ranked.iter().take(per_group).map(|x| x.0));
            }
            // Keep the caller's relative order among the finalists.
            let slot_of = |o: usize| primary.order.iter().position(|&x| x == o).expect("option");
            finalists.sort_by_key(|&o| slot_of(o));
            finalists_of.insert(
                id.clone(),
                finalists
                    .iter()
                    .map(|&o| primary.keys[slot_of(o)].clone())
                    .collect(),
            );
            for o in option_orders(finalists.len(), orders_for(primary.kind, self.orders)) {
                let order: Vec<usize> = o.iter().map(|&s| finalists[s]).collect();
                branches.push(renderer.branch(id, q, Some(&order)));
            }
        }
        (branches, finalists_of, rechecked)
    }

    /// Fold every question's branches into one distribution in its primary
    /// branch's slot order: the mean logit per option over the orders it
    /// was asked in (the second pass's for a grouped Choice), options that
    /// were never asked at [`EXCLUDED_LOGIT`], then a softmax at
    /// `temperature`. `first` holds one distribution per `self.first`
    /// branch; `second` the branches [`Plan::second`] returned with theirs.
    pub fn fold(
        self,
        first: Vec<Distribution>,
        second: Vec<(RenderedBranch, Distribution)>,
        temperature: f32,
    ) -> Folded {
        let n = self.rendered.branches.len();
        let mut runs: Vec<Vec<(Vec<usize>, Distribution)>> = (0..n).map(|_| Vec::new()).collect();
        for (b, d) in self.first.iter().zip(first) {
            runs[self.question_index(&b.id)].push((b.order.clone(), d));
        }
        // A grouped question's first pass only chose the finalists.
        for &qi in &self.grouped {
            runs[qi].clear();
        }
        for (b, d) in second {
            runs[self.question_index(&b.id)].push((b.order, d));
        }
        let mut order_spread = IndexMap::new();
        let mut results = Vec::with_capacity(n);
        for (primary, runs) in self.rendered.branches.into_iter().zip(runs) {
            let k = primary.order.len();
            let mut sum = vec![0.0f64; k];
            let mut base_sum = vec![0.0f64; k];
            let mut count = vec![0usize; k];
            let mut per_run_probs: Vec<Vec<f64>> = Vec::with_capacity(runs.len());
            let mut has_baseline = !runs.is_empty();
            let mut candidate_mass: Option<f64> = None;
            for (order, dist) in &runs {
                let mut p = vec![0.0; k];
                for (slot, &orig) in order.iter().enumerate() {
                    sum[orig] += f64::from(dist.logits[slot]);
                    count[orig] += 1;
                    p[orig] = dist.probs[slot];
                    match &dist.baseline {
                        Some(b) => base_sum[orig] += f64::from(b[slot]),
                        None => has_baseline = false,
                    }
                }
                per_run_probs.push(p);
                if let Some(m) = dist.candidate_mass {
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
                order_spread.insert(primary.id.clone(), spread);
            }
            let probs = softmax(&logits, temperature);
            results.push((
                primary,
                Distribution {
                    logits,
                    baseline,
                    probs,
                    candidate_mass,
                },
            ));
        }
        Folded {
            results,
            order_spread,
        }
    }
}

fn orders_for(kind: Kind, n: usize) -> usize {
    match kind {
        Kind::Choice | Kind::Noul => n,
        Kind::Score => 1,
    }
}
