use typed_builder::TypedBuilder;

#[derive(Debug, Clone, TypedBuilder)]
pub struct InhibitionConfig {
    #[builder(default = 0.15)]
    pub strength: f64,
    #[builder(default = 7)]
    pub breadth: usize,
}

impl Default for InhibitionConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

pub trait Inhibitor: Send + Sync {
    fn inhibit(&self, activations: &mut [f64], config: &InhibitionConfig);
}

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
            let suppression = config.strength * (mean_winner - activations[i]);
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
        let mut activations = vec![0.5, 0.5, 0.5];
        let inhibitor = TopMInhibitor;
        inhibitor.inhibit(&mut activations, &config(0.15, 2));
        assert_eq!(activations[0], 0.5);
        assert_eq!(activations[1], 0.5);
        assert_eq!(activations[2], 0.5);
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
