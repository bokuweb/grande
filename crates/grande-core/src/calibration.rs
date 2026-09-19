//! Contextual calibration, one-parameter temperature scaling and
//! calibration metrics.
//!
//! Fit the temperature on a calibration split only and apply it unchanged
//! elsewhere; a temperature fitted on the data you report is not a result.

use crate::math::softmax;

/// The content-free state used for contextual calibration by default.
pub const CONTENT_FREE: &str = "N/A";

/// Contextual calibration (Zhao et al. 2021, "Calibrate Before Use"): the
/// same question asked over a content-free state gives the model's prior
/// over the options — the "yes" bias of an instruct model, the pull of the
/// first letter, the majority label of the few-shot block. Dividing the
/// live distribution by that prior removes it; in logit space that is a
/// subtraction, and the constant the two log-normalizers differ by cancels
/// in the softmax that follows.
pub fn contextual(logits: &[f32], baseline: &[f32]) -> Vec<f32> {
    debug_assert_eq!(logits.len(), baseline.len());
    logits.iter().zip(baseline).map(|(z, b)| z - b).collect()
}

/// A scored example: option logits and the index of the correct option.
#[derive(Debug, Clone)]
pub struct Labeled {
    pub logits: Vec<f32>,
    pub label: usize,
}

/// Mean negative log-likelihood at temperature `t`.
pub fn nll(data: &[Labeled], t: f32) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter()
        .map(|x| -softmax(&x.logits, t)[x.label].max(1e-12).ln())
        .sum::<f64>()
        / data.len() as f64
}

/// Golden-section search for the temperature minimizing NLL on `data`.
pub fn fit_temperature(data: &[Labeled]) -> f32 {
    let (mut lo, mut hi) = (0.05f64, 10.0f64);
    let phi = (5f64.sqrt() - 1.0) / 2.0;
    let mut a = hi - phi * (hi - lo);
    let mut b = lo + phi * (hi - lo);
    let (mut fa, mut fb) = (nll(data, a as f32), nll(data, b as f32));
    for _ in 0..80 {
        if fa < fb {
            hi = b;
            b = a;
            fb = fa;
            a = hi - phi * (hi - lo);
            fa = nll(data, a as f32);
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + phi * (hi - lo);
            fb = nll(data, b as f32);
        }
    }
    ((lo + hi) / 2.0) as f32
}

/// Expected calibration error with equal-width bins over max probability.
pub fn ece(data: &[Labeled], t: f32, bins: usize) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut conf_sum = vec![0.0; bins];
    let mut acc_sum = vec![0.0; bins];
    let mut count = vec![0usize; bins];
    for x in data {
        let p = softmax(&x.logits, t);
        let (arg, conf) = p
            .iter()
            .enumerate()
            .fold((0, 0.0), |m, (i, &v)| if v > m.1 { (i, v) } else { m });
        let bin = ((conf * bins as f64) as usize).min(bins - 1);
        conf_sum[bin] += conf;
        acc_sum[bin] += f64::from(u8::from(arg == x.label));
        count[bin] += 1;
    }
    let n = data.len() as f64;
    (0..bins)
        .filter(|&b| count[b] > 0)
        .map(|b| {
            let c = count[b] as f64;
            (c / n) * ((acc_sum[b] / c) - (conf_sum[b] / c)).abs()
        })
        .sum()
}

/// Mean Brier score (multi-class, summed over options).
pub fn brier(data: &[Labeled], t: f32) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter()
        .map(|x| {
            softmax(&x.logits, t)
                .iter()
                .enumerate()
                .map(|(i, &p)| (p - f64::from(u8::from(i == x.label))).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / data.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contextual_removes_the_prior() {
        // The model says "yes" 73% with no evidence; the same lean over the
        // live state is no evidence either and lands at 50%.
        let prior = [1.0, 0.0];
        let p = softmax(&contextual(&[1.0, 0.0], &prior), 1.0);
        assert!((p[0] - 0.5).abs() < 1e-9);
        // Evidence on top of the prior survives.
        let p = softmax(&contextual(&[5.0, 0.0], &prior), 1.0);
        assert!(p[0] > 0.98);
        // A "no" prior is lifted the same way.
        let p = softmax(&contextual(&[0.0, 11.0], &[0.0, 11.0]), 1.0);
        assert!((p[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn overconfident_logits_get_a_temperature_above_one() {
        // Half the "confident" answers are wrong: the fit must flatten.
        let mut data = Vec::new();
        for i in 0..100 {
            data.push(Labeled {
                logits: vec![4.0, 0.0],
                label: if i % 2 == 0 { 0 } else { 1 },
            });
        }
        let t = fit_temperature(&data);
        assert!(t > 3.0, "t = {t}");
        assert!(ece(&data, t, 10) < ece(&data, 1.0, 10));
    }

    #[test]
    fn perfect_predictions_keep_low_temperature() {
        let data: Vec<Labeled> = (0..50)
            .map(|_| Labeled {
                logits: vec![6.0, 0.0],
                label: 0,
            })
            .collect();
        assert!(fit_temperature(&data) < 0.5);
    }
}
