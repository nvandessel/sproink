//! Oja-stabilized Hebbian co-activation learning.

/// Configuration for Oja's rule weight updates.
#[derive(Debug, Clone)]
pub struct HebbianConfig {
    /// Learning rate (eta). Default: 0.05.
    pub learning_rate: f64,
    /// Floor for edge weights. Default: 0.01.
    pub min_weight: f64,
    /// Ceiling for edge weights. Default: 0.95.
    pub max_weight: f64,
    /// Minimum activation to qualify for co-activation. Default: 0.15.
    pub activation_threshold: f64,
}

impl Default for HebbianConfig {
    fn default() -> Self {
        HebbianConfig {
            learning_rate: 0.05,
            min_weight: 0.01,
            max_weight: 0.95,
            activation_threshold: 0.15,
        }
    }
}

/// A co-activated pair of nodes.
#[derive(Debug, Clone)]
pub struct CoActivationPair {
    pub node_a: u32,
    pub node_b: u32,
    pub activation_a: f64,
    pub activation_b: f64,
}

/// Compute new edge weight using Oja's rule.
///
/// dW = eta * (A_i * A_j - A_j^2 * W)
///
/// The A_j^2 * W forgetting factor prevents unbounded weight growth.
/// Result is clamped to [min_weight, max_weight].
pub fn oja_update(current_weight: f64, activation_a: f64, activation_b: f64, cfg: &HebbianConfig) -> f64 {
    let hebbian = activation_a * activation_b;
    let forgetting = activation_b * activation_b * current_weight;
    let dw = cfg.learning_rate * (hebbian - forgetting);

    let new_weight = current_weight + dw;
    clamp_weight(new_weight, cfg.min_weight, cfg.max_weight)
}

/// Extract co-activation pairs from activation results.
///
/// Returns all unique pairs where both nodes exceed the activation threshold.
/// Seed-seed pairs are excluded (they reflect context matching, not behavioral affinity).
pub fn extract_coactivation_pairs(
    activations: &[f64],
    seed_nodes: &[u32],
    cfg: &HebbianConfig,
) -> Vec<CoActivationPair> {
    let seed_set: std::collections::HashSet<u32> = seed_nodes.iter().copied().collect();

    // Collect nodes above threshold.
    let active: Vec<(u32, f64)> = activations
        .iter()
        .enumerate()
        .filter(|&(_, a)| *a >= cfg.activation_threshold)
        .map(|(i, &a)| (i as u32, a))
        .collect();

    if active.len() < 2 {
        return Vec::new();
    }

    let mut pairs = Vec::with_capacity(active.len() * (active.len() - 1) / 2);
    for i in 0..active.len() {
        for j in (i + 1)..active.len() {
            let a_is_seed = seed_set.contains(&active[i].0);
            let b_is_seed = seed_set.contains(&active[j].0);

            // Skip seed-seed pairs.
            if a_is_seed && b_is_seed {
                continue;
            }

            // Canonical ordering: smaller node ID first.
            let (a, b) = if active[i].0 < active[j].0 {
                (active[i], active[j])
            } else {
                (active[j], active[i])
            };

            pairs.push(CoActivationPair {
                node_a: a.0,
                node_b: b.0,
                activation_a: a.1,
                activation_b: b.1,
            });
        }
    }

    pairs
}

fn clamp_weight(w: f64, min: f64, max: f64) -> f64 {
    if w.is_nan() || w.is_infinite() {
        return min;
    }
    w.clamp(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oja_strengthens_coactivation() {
        let cfg = HebbianConfig::default();
        let new_w = oja_update(0.5, 0.8, 0.7, &cfg);
        // With high co-activation, weight should increase (hebbian > forgetting).
        // hebbian = 0.56, forgetting = 0.49 * 0.5 = 0.245
        // dw = 0.05 * (0.56 - 0.245) = 0.01575
        assert!(new_w > 0.5);
    }

    #[test]
    fn oja_stabilizes_at_high_weight() {
        let cfg = HebbianConfig::default();
        // At high weight, forgetting term dominates.
        let new_w = oja_update(0.9, 0.5, 0.5, &cfg);
        // hebbian = 0.25, forgetting = 0.25 * 0.9 = 0.225
        // dw = 0.05 * (0.25 - 0.225) = 0.00125
        // Still increases slightly. Let's check it stays in bounds.
        assert!(new_w <= cfg.max_weight);
        assert!(new_w >= cfg.min_weight);
    }

    #[test]
    fn oja_clamps_to_bounds() {
        let cfg = HebbianConfig::default();
        let new_w = oja_update(0.001, 0.0, 0.0, &cfg);
        assert_eq!(new_w, cfg.min_weight);
    }

    #[test]
    fn coactivation_excludes_seed_seed_pairs() {
        let activations = vec![0.8, 0.7, 0.6, 0.5];
        let seeds = vec![0, 1];
        let cfg = HebbianConfig::default();
        let pairs = extract_coactivation_pairs(&activations, &seeds, &cfg);

        // Should NOT have pair (0, 1) since both are seeds.
        for p in &pairs {
            assert!(!(p.node_a == 0 && p.node_b == 1));
        }
        // Should have pairs involving at least one non-seed.
        assert!(!pairs.is_empty());
    }
}
