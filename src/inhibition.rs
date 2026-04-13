//! Lateral inhibition (top-M winner-take-all).
//!
//! After propagation, suppresses activations outside the top-M active nodes
//! proportionally to the gap between their activation and the winners' mean.

use typed_builder::TypedBuilder;

use crate::error::SproinkError;

/// Configuration for lateral inhibition.
#[derive(Debug, Clone, TypedBuilder)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InhibitionConfig {
    /// Suppression multiplier applied to losers. Default: `0.15`.
    #[builder(default = 0.15)]
    pub strength: f64,
    /// Number of top-M winners that are immune to suppression. Default: `7`.
    #[builder(default = 7)]
    pub breadth: usize,
}

impl Default for InhibitionConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl InhibitionConfig {
    /// Validates that the config has sane values.
    pub fn validate(&self) -> Result<(), SproinkError> {
        if self.breadth == 0 {
            return Err(SproinkError::InvalidValue {
                field: "inhibition_breadth",
                value: 0.0,
            });
        }
        if !self.strength.is_finite() || self.strength < 0.0 {
            return Err(SproinkError::InvalidValue {
                field: "inhibition_strength",
                value: self.strength,
            });
        }
        Ok(())
    }
}

/// Trait for inhibition strategies.
pub trait Inhibitor: Send + Sync {
    /// Applies inhibition to the activation vector in-place.
    fn inhibit(&self, activations: &mut [f64], config: &InhibitionConfig);
}

/// Top-M winner-take-all inhibitor.
///
/// The top `breadth` nodes keep their activation unchanged. All other active
/// nodes are suppressed by `strength × (mean_winner − their_activation)`.
pub struct TopMInhibitor;

impl Inhibitor for TopMInhibitor {
    fn inhibit(&self, activations: &mut [f64], config: &InhibitionConfig) {
        let mut active: Vec<usize> = activations
            .iter()
            .enumerate()
            .filter(|&(_, &a)| a > 0.0)
            .map(|(i, _)| i)
            .collect();

        if active.len() <= config.breadth {
            return;
        }

        // Sort by activation descending, tie-break by index ascending
        active.sort_by(|&a, &b| {
            activations[b]
                .partial_cmp(&activations[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let mean_winner: f64 = active[..config.breadth]
            .iter()
            .map(|&i| activations[i])
            .sum::<f64>()
            / config.breadth as f64;

        for &i in &active[config.breadth..] {
            let gap = (mean_winner - activations[i]).max(0.0);
            let suppression = config.strength * gap;
            activations[i] = (activations[i] - suppression).max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(strength: f64, breadth: usize) -> InhibitionConfig {
        InhibitionConfig::builder()
            .strength(strength)
            .breadth(breadth)
            .build()
    }

    #[test]
    fn fewer_active_than_breadth_is_noop() {
        let mut activations = vec![0.8, 0.6, 0.4, 0.0, 0.0];
        let original = activations.clone();
        let inhibitor = TopMInhibitor;
        inhibitor.inhibit(&mut activations, &config(0.15, 5));
        assert_eq!(activations, original);
    }

    #[test]
    fn losers_are_suppressed() {
        let mut activations = vec![0.8, 0.6, 0.4, 0.2];
        let inhibitor = TopMInhibitor;
        inhibitor.inhibit(&mut activations, &config(0.15, 2));
        assert_eq!(activations[0], 0.8);
        assert_eq!(activations[1], 0.6);
        assert!((activations[2] - 0.355).abs() < 1e-10);
        assert!((activations[3] - 0.125).abs() < 1e-10);
    }

    #[test]
    fn suppression_floors_at_zero() {
        let mut activations = vec![0.9, 0.01];
        let inhibitor = TopMInhibitor;
        inhibitor.inhibit(&mut activations, &config(1.0, 1));
        assert!(activations[1] >= 0.0);
    }

    #[test]
    fn tie_breaking_by_node_id() {
        // Four nodes: node 0 clearly wins. Nodes 1, 2, 3 all tie at 0.4.
        // With breadth=2, the top 2 winners should be nodes 0 and 1 (lower ID
        // wins the tie). Nodes 2 and 3 should both be suppressed.
        let mut activations = vec![0.9, 0.4, 0.4, 0.4];
        let inhibitor = TopMInhibitor;
        inhibitor.inhibit(&mut activations, &config(0.5, 2));

        // Node 0 is a clear winner -> unchanged
        assert!((activations[0] - 0.9).abs() < 1e-10);
        // Node 1 wins the tie (lowest ID among the tied) -> unchanged
        assert!(
            (activations[1] - 0.4).abs() < 1e-10,
            "node 1 should win tie and be immune, got {}",
            activations[1]
        );
        // Nodes 2 and 3 lose the tie -> must be strictly suppressed.
        // mean_winner = (0.9 + 0.4) / 2 = 0.65; gap = 0.25; suppression = 0.125;
        // new value = 0.275.
        assert!(
            activations[2] < 0.4 - 1e-10,
            "node 2 should be suppressed, got {}",
            activations[2]
        );
        assert!(
            activations[3] < 0.4 - 1e-10,
            "node 3 should be suppressed, got {}",
            activations[3]
        );
        assert!((activations[2] - 0.275).abs() < 1e-10);
        assert!((activations[3] - 0.275).abs() < 1e-10);
    }

    #[test]
    fn loser_above_mean_winner_is_not_amplified() {
        // Winners have high variance: [1.0, 0.05, 0.05], mean_winner ≈ 0.367.
        // A loser at 0.4 sits above that mean; before the gap-clamp fix the
        // negative gap would flip the suppression sign and amplify the
        // loser. After the fix the clamped gap is 0 and the value must not
        // increase.
        let mut activations = vec![1.0, 0.05, 0.05, 0.4];
        let inhibitor = TopMInhibitor;
        inhibitor.inhibit(&mut activations, &config(0.5, 3));
        assert!(
            activations[3] <= 0.4 + 1e-10,
            "loser was amplified to {}",
            activations[3]
        );
    }

    #[test]
    fn validate_rejects_zero_breadth() {
        let cfg = config(0.15, 0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_nan_strength() {
        let cfg = config(f64::NAN, 3);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_accepts_defaults() {
        assert!(InhibitionConfig::default().validate().is_ok());
    }

    #[test]
    fn zero_activations_not_counted_as_active() {
        let mut activations = vec![0.8, 0.0, 0.0, 0.0];
        let original = activations.clone();
        let inhibitor = TopMInhibitor;
        inhibitor.inhibit(&mut activations, &config(0.15, 2));
        assert_eq!(activations, original);
    }
}
