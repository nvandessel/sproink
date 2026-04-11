//! Sigmoid squashing function for post-propagation normalization.

/// Applies sigmoid squashing element-wise in-place.
///
/// For each activation `x` greater than a small epsilon (`1e-9`), replaces it
/// with `1 / (1 + exp(-gain × (x − center)))`. Values at or below the epsilon
/// are left unchanged so that exactly-zero and numerically-negligible
/// activations are not amplified up to the sigmoid baseline (which, at the
/// default gain/center, would otherwise map `0.0 -> ~0.047`).
pub fn squash_sigmoid(activations: &mut [f64], gain: f64, center: f64) {
    const ZERO_EPSILON: f64 = 1e-9;
    for v in activations.iter_mut() {
        if *v > ZERO_EPSILON {
            *v = 1.0 / (1.0 + (-gain * (*v - center)).exp());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_center_returns_half() {
        let mut v = vec![0.3];
        squash_sigmoid(&mut v, 10.0, 0.3);
        assert!((v[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn well_above_center_near_one() {
        let mut v = vec![1.0];
        squash_sigmoid(&mut v, 10.0, 0.3);
        assert!(v[0] > 0.99);
    }

    #[test]
    fn well_below_center_near_zero() {
        // Use a small positive value below the 0.3 center; 0.0 would be skipped
        // by the zero-skip branch and not exercise sigmoid at all.
        let mut v = vec![0.05];
        squash_sigmoid(&mut v, 10.0, 0.3);
        assert!(
            v[0] < 0.1,
            "expected sigmoid at 0.05 (well below center 0.3) to be near 0, got {}",
            v[0]
        );
    }

    #[test]
    fn higher_gain_steeper_curve() {
        let mut low_gain = vec![0.5];
        let mut high_gain = vec![0.5];
        squash_sigmoid(&mut low_gain, 5.0, 0.3);
        squash_sigmoid(&mut high_gain, 20.0, 0.3);
        assert!(high_gain[0] > low_gain[0]);
    }

    #[test]
    fn empty_slice_is_noop() {
        let mut v: Vec<f64> = vec![];
        squash_sigmoid(&mut v, 10.0, 0.3);
    }

    #[test]
    fn multiple_values() {
        let mut v = vec![0.0, 0.3, 0.6, 1.0];
        squash_sigmoid(&mut v, 10.0, 0.3);
        for i in 1..v.len() {
            assert!(v[i] >= v[i - 1]);
        }
        assert_eq!(v[0], 0.0); // zero inputs stay zero (skipped)
        for &val in &v[1..] {
            assert!(val > 0.0 && val < 1.0);
        }
    }
}
