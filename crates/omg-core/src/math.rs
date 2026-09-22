//! Small numeric helpers. Hand-written on purpose: no linear algebra crate is
//! worth pulling in for a softmax and a 256-wide dot product.

/// Softmax of `z / temperature`, numerically stable.
pub fn softmax(z: &[f32], temperature: f32) -> Vec<f64> {
    let t = f64::from(temperature.max(1e-6));
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = z.iter().map(|&v| ((v as f64 - m) / t).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// Log-sum-exp over a full logits row, used for candidate mass diagnostics.
pub fn log_sum_exp(z: &[f32]) -> f64 {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    m + z.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
}

/// Entropy-based concentration: `1 - H(p) / ln K`. 0 for uniform, 1 for a
/// point mass. A single option is 1 by definition. This is jev_local's
/// formula; TypeSafe has not published theirs.
pub fn confidence(p: &[f64]) -> f64 {
    let k = p.len();
    if k <= 1 {
        return 1.0;
    }
    let h: f64 = p.iter().filter(|&&v| v > 0.0).map(|&v| -v * v.ln()).sum();
    (1.0 - h / (k as f64).ln()).clamp(0.0, 1.0)
}

/// Expected level index for a Score distribution.
pub fn expected_index(p: &[f64]) -> f64 {
    p.iter().enumerate().map(|(i, &v)| i as f64 * v).sum()
}

/// Distinct option orders for a `k`-way question: the identity, its
/// rotations, then the reversals of those — `2k` orders for `k >= 3`, every
/// permutation for `k <= 2`. Returns at most `n`. Rotations move every
/// option through every slot, reversals flip the neighbours, which is what a
/// position-bias probe (or an average over orders) needs.
pub fn option_orders(k: usize, n: usize) -> Vec<Vec<usize>> {
    let distinct = match k {
        0 | 1 => 1,
        2 => 2,
        _ => 2 * k,
    };
    (0..n.min(distinct))
        .map(|r| {
            let mut o: Vec<usize> = (0..k).collect();
            if r >= k {
                o.reverse();
            }
            o.rotate_left(r % k);
            o
        })
        .collect()
}

pub fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_orders() {
        let p = softmax(&[1.0, 2.0, 3.0], 1.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn temperature_flattens() {
        let sharp = softmax(&[0.0, 4.0], 1.0);
        let flat = softmax(&[0.0, 4.0], 4.0);
        assert!(sharp[1] > flat[1]);
    }

    #[test]
    fn confidence_bounds() {
        assert!((confidence(&[0.5, 0.5])).abs() < 1e-12);
        assert!((confidence(&[1.0, 0.0]) - 1.0).abs() < 1e-12);
        assert_eq!(confidence(&[1.0]), 1.0);
    }

    #[test]
    fn option_orders_are_distinct_and_capped() {
        assert_eq!(option_orders(1, 4), vec![vec![0]]);
        assert_eq!(option_orders(2, 4), vec![vec![0, 1], vec![1, 0]]);
        let o = option_orders(3, 8);
        assert_eq!(o.len(), 6);
        let mut set = o.clone();
        set.sort();
        set.dedup();
        assert_eq!(set.len(), 6);
        assert_eq!(
            option_orders(5, 3),
            vec![
                vec![0, 1, 2, 3, 4],
                vec![1, 2, 3, 4, 0],
                vec![2, 3, 4, 0, 1]
            ]
        );
    }

    #[test]
    fn expected_index_is_weighted_mean() {
        assert!((expected_index(&[0.0, 0.5, 0.5]) - 1.5).abs() < 1e-12);
    }
}
