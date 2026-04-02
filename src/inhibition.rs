//! Lateral inhibition: top-M winners suppress weaker competitors.

/// Configuration for lateral inhibition.
#[derive(Debug, Clone)]
pub struct InhibitionConfig {
    /// Suppression strength (beta). Default: 0.15.
    pub strength: f64,
    /// Number of top nodes that compete (M). Default: 7.
    pub breadth: usize,
    /// Whether inhibition is enabled.
    pub enabled: bool,
}

impl Default for InhibitionConfig {
    fn default() -> Self {
        InhibitionConfig {
            strength: 0.15,
            breadth: 7,
            enabled: true,
        }
    }
}

/// Apply lateral inhibition in-place.
///
/// Algorithm:
/// 1. Find top-M activated nodes (winners).
/// 2. For each non-winner (loser):
///    suppression = beta * (mean_winner_activation - loser_activation)
///    loser_activation = max(0, loser_activation - suppression)
pub fn apply_inhibition(activation: &mut [f64], config: &InhibitionConfig) {
    if !config.enabled || activation.is_empty() {
        return;
    }

    // Count active nodes.
    let mut indices: Vec<usize> = activation
        .iter()
        .enumerate()
        .filter(|&(_, a)| *a > 0.0)
        .map(|(i, _)| i)
        .collect();

    if indices.len() <= config.breadth {
        return; // Everyone is a winner.
    }

    // Sort by activation descending, ties broken by index ascending (deterministic).
    indices.sort_by(|&a, &b| {
        activation[b]
            .partial_cmp(&activation[a])
            .unwrap()
            .then(a.cmp(&b))
    });

    let m = config.breadth;

    // Mean winner activation.
    let winner_sum: f64 = indices[..m].iter().map(|&i| activation[i]).sum();
    let mean_winner_act = winner_sum / m as f64;

    // Suppress losers.
    for &loser_idx in &indices[m..] {
        let loser_act = activation[loser_idx];
        let suppression = config.strength * (mean_winner_act - loser_act);
        let suppressed = loser_act - suppression;
        activation[loser_idx] = if suppressed < 0.0 { 0.0 } else { suppressed };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_suppression_when_fewer_than_breadth() {
        let mut act = vec![0.0, 0.5, 0.0, 0.3, 0.0];
        let cfg = InhibitionConfig::default(); // breadth=7
        apply_inhibition(&mut act, &cfg);
        // Only 2 active nodes < 7, so no change.
        assert_eq!(act[1], 0.5);
        assert_eq!(act[3], 0.3);
    }

    #[test]
    fn losers_are_suppressed() {
        // 10 active nodes, breadth=3.
        let mut act = vec![1.0, 0.9, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02];
        let cfg = InhibitionConfig {
            strength: 0.15,
            breadth: 3,
            enabled: true,
        };
        let original = act.clone();
        apply_inhibition(&mut act, &cfg);

        // Winners (indices 0,1,2) should be unchanged.
        assert_eq!(act[0], original[0]);
        assert_eq!(act[1], original[1]);
        assert_eq!(act[2], original[2]);

        // Losers should be suppressed (lower than original).
        for i in 3..10 {
            assert!(act[i] <= original[i], "loser {} should be suppressed", i);
        }
    }

    #[test]
    fn disabled_is_noop() {
        let mut act = vec![1.0, 0.5, 0.3];
        let cfg = InhibitionConfig {
            enabled: false,
            ..Default::default()
        };
        let original = act.clone();
        apply_inhibition(&mut act, &cfg);
        assert_eq!(act, original);
    }
}
