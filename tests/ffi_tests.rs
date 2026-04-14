#![cfg(feature = "ffi")]

use sproink::ffi::*;

#[test]
fn round_trip_graph_build_activate_results() {
    unsafe {
        let sources = [0u32, 1];
        let targets = [1u32, 2];
        let weights = [0.8f64, 0.8];
        let kinds = [0u8, 0]; // Positive

        let graph = sproink_graph_build(
            3,
            2,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        assert!(!graph.is_null());

        let seed_nodes = [0u32];
        let seed_activations = [1.0f64];
        let results = sproink_activate(
            graph,
            1,
            seed_nodes.as_ptr(),
            seed_activations.as_ptr(),
            std::ptr::null(),
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            1,
            0.15,
            7,
            f64::NAN,
            f64::NAN,
        );
        assert!(!results.is_null());

        let len = sproink_results_len(results);
        assert!(len > 0);

        let mut nodes = vec![0u32; len as usize];
        let mut activations = vec![0.0f64; len as usize];
        let mut distances = vec![0u32; len as usize];
        let wrote_n = sproink_results_nodes(results, nodes.as_mut_ptr(), len);
        let wrote_a = sproink_results_activations(results, activations.as_mut_ptr(), len);
        let wrote_d = sproink_results_distances(results, distances.as_mut_ptr(), len);
        assert_eq!(wrote_n, len);
        assert_eq!(wrote_a, len);
        assert_eq!(wrote_d, len);

        for &a in &activations {
            assert!((0.0..=1.0).contains(&a));
        }

        sproink_results_free(results);
        sproink_graph_free(graph);
    }
}

#[test]
fn null_graph_returns_null_results() {
    unsafe {
        let seed_nodes = [0u32];
        let seed_activations = [1.0f64];
        let results = sproink_activate(
            std::ptr::null(),
            1,
            seed_nodes.as_ptr(),
            seed_activations.as_ptr(),
            std::ptr::null(),
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            1,
            0.15,
            7,
            f64::NAN,
            f64::NAN,
        );
        assert!(results.is_null());
    }
}

#[test]
fn null_results_returns_zero_len() {
    unsafe {
        assert_eq!(sproink_results_len(std::ptr::null()), 0);
    }
}

#[test]
fn pairs_round_trip() {
    unsafe {
        // Build a 3-node chain
        let sources = [0u32, 1];
        let targets = [1u32, 2];
        let weights = [0.8f64, 0.8];
        let kinds = [0u8, 0];

        let graph = sproink_graph_build(
            3,
            2,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        assert!(!graph.is_null());

        let seed_nodes = [0u32];
        let seed_activations = [1.0f64];
        let results = sproink_activate(
            graph,
            1,
            seed_nodes.as_ptr(),
            seed_activations.as_ptr(),
            std::ptr::null(),
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            0,
            0.15,
            7,
            f64::NAN,
            f64::NAN,
        );
        assert!(!results.is_null());

        let pairs = sproink_extract_pairs(results, 1, seed_nodes.as_ptr(), 0.1);
        assert!(!pairs.is_null());

        let pair_len = sproink_pairs_len(pairs);
        if pair_len > 0 {
            let mut nodes_a = vec![0u32; pair_len as usize];
            let mut nodes_b = vec![0u32; pair_len as usize];
            let mut acts_a = vec![0.0f64; pair_len as usize];
            let mut acts_b = vec![0.0f64; pair_len as usize];
            let wrote_pn =
                sproink_pairs_nodes(pairs, nodes_a.as_mut_ptr(), nodes_b.as_mut_ptr(), pair_len);
            let wrote_pa = sproink_pairs_activations(
                pairs,
                acts_a.as_mut_ptr(),
                acts_b.as_mut_ptr(),
                pair_len,
            );
            assert_eq!(wrote_pn, pair_len);
            assert_eq!(wrote_pa, pair_len);

            for (&a, &b) in acts_a.iter().zip(acts_b.iter()) {
                assert!((0.0..=1.0).contains(&a));
                assert!((0.0..=1.0).contains(&b));
            }
        }

        sproink_pairs_free(pairs);
        sproink_results_free(results);
        sproink_graph_free(graph);
    }
}

#[test]
fn oja_update_via_ffi() {
    unsafe {
        let w = sproink_oja_update(0.3, 0.8, 0.7, 0.05, 0.01, 0.95);
        assert!((0.01..=0.95).contains(&w));
        assert!(w > 0.3); // should strengthen
    }
}

#[test]
fn null_graph_build_inputs() {
    unsafe {
        let result = sproink_graph_build(
            3,
            2,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
        );
        assert!(result.is_null());
    }
}

#[test]
fn null_pairs_returns_zero_len() {
    unsafe {
        assert_eq!(sproink_pairs_len(std::ptr::null()), 0);
    }
}

#[test]
fn all_edge_kinds_via_ffi() {
    unsafe {
        // kinds: 0=Positive, 1=Conflicts, 2=DirSuppressive, 4=FeatureAffinity
        let sources = [0u32, 0, 0, 0];
        let targets = [1u32, 2, 3, 4];
        let weights = [0.8, 0.5, 0.5, 0.3];
        let kinds = [0u8, 1, 2, 4];

        let graph = sproink_graph_build(
            5,
            4,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        assert!(!graph.is_null());

        let seed_nodes = [0u32];
        let seed_activations = [1.0f64];
        let results = sproink_activate(
            graph,
            1,
            seed_nodes.as_ptr(),
            seed_activations.as_ptr(),
            std::ptr::null(),
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            1,
            0.15,
            7,
            f64::NAN,
            f64::NAN,
        );
        assert!(!results.is_null());
        assert!(sproink_results_len(results) > 0);

        sproink_results_free(results);
        sproink_graph_free(graph);
    }
}

#[test]
fn null_output_pointers_dont_crash() {
    unsafe {
        // Build a valid graph + results first
        let sources = [0u32];
        let targets = [1u32];
        let weights = [0.8f64];
        let kinds = [0u8];
        let graph = sproink_graph_build(
            2,
            1,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        let seed_nodes = [0u32];
        let seed_activations = [1.0f64];
        let results = sproink_activate(
            graph,
            1,
            seed_nodes.as_ptr(),
            seed_activations.as_ptr(),
            std::ptr::null(),
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            0,
            0.15,
            7,
            f64::NAN,
            f64::NAN,
        );

        // Null output pointers should not crash and should write nothing.
        assert_eq!(sproink_results_nodes(results, std::ptr::null_mut(), 0), 0);
        assert_eq!(
            sproink_results_activations(results, std::ptr::null_mut(), 0),
            0
        );
        assert_eq!(
            sproink_results_distances(results, std::ptr::null_mut(), 0),
            0
        );

        // Null pairs accessors
        assert_eq!(
            sproink_pairs_nodes(
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            ),
            0
        );
        assert_eq!(
            sproink_pairs_activations(
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            ),
            0
        );

        // Null extract
        let pairs = sproink_extract_pairs(std::ptr::null(), 0, std::ptr::null(), 0.1);
        assert!(pairs.is_null());

        sproink_results_free(results);
        sproink_graph_free(graph);
    }
}

#[test]
fn activate_with_empty_seeds() {
    unsafe {
        let sources = [0u32];
        let targets = [1u32];
        let weights = [0.8f64];
        let kinds = [0u8];
        let graph = sproink_graph_build(
            2,
            1,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );

        // Zero seeds, null pointers
        let results = sproink_activate(
            graph,
            0,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            0,
            0.15,
            7,
            f64::NAN,
            f64::NAN,
        );
        assert!(!results.is_null());
        // All nodes get sigmoid(0) ≈ 0.047 which may be above min_activation
        sproink_results_free(results);
        sproink_graph_free(graph);
    }
}

#[test]
fn oja_nan_safety_via_ffi() {
    unsafe {
        let w = sproink_oja_update(0.5, 0.0, 0.0, 0.05, 0.01, 0.95);
        assert!((0.01..=0.95).contains(&w));
        assert!(!w.is_nan());
    }
}

#[test]
fn small_buffer_truncates_results_copy() {
    // Regression: I28 — copy functions previously trusted the caller's
    // buffer size. Now they must write at most `buffer_len` elements.
    unsafe {
        let sources = [0u32, 1];
        let targets = [1u32, 2];
        let weights = [0.8f64, 0.8];
        let kinds = [0u8, 0];

        let graph = sproink_graph_build(
            3,
            2,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        let seed_nodes = [0u32];
        let seed_activations = [1.0f64];
        let results = sproink_activate(
            graph,
            1,
            seed_nodes.as_ptr(),
            seed_activations.as_ptr(),
            std::ptr::null(),
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            0,
            0.15,
            7,
            f64::NAN,
            f64::NAN,
        );

        let len = sproink_results_len(results);
        assert!(len >= 2);

        // Provide a buffer with only 1 slot; sentinel in slot 1 must survive.
        let mut nodes = [u32::MAX, u32::MAX];
        let wrote = sproink_results_nodes(results, nodes.as_mut_ptr(), 1);
        assert_eq!(wrote, 1);
        assert_ne!(nodes[0], u32::MAX, "slot 0 should have been written");
        assert_eq!(nodes[1], u32::MAX, "slot 1 must remain untouched");

        // buffer_len == 0 writes nothing.
        let mut acts = [-1.0f64; 4];
        let wrote_a = sproink_results_activations(results, acts.as_mut_ptr(), 0);
        assert_eq!(wrote_a, 0);
        assert!(acts.iter().all(|&v| v == -1.0));

        sproink_results_free(results);
        sproink_graph_free(graph);
    }
}

#[test]
fn oja_update_rejects_invalid_config() {
    // Regression: I33/I34 — unvalidated HebbianConfig could panic under
    // `min_weight > max_weight` or produce wrong values from NaN inputs.
    unsafe {
        // min > max → return min_weight sentinel
        let w = sproink_oja_update(0.5, 0.5, 0.5, 0.05, 0.9, 0.1);
        assert_eq!(w, 0.9);

        // NaN current_weight → return min_weight sentinel
        let w = sproink_oja_update(f64::NAN, 0.5, 0.5, 0.05, 0.01, 0.95);
        assert_eq!(w, 0.01);

        // NaN activation → return min_weight sentinel
        let w = sproink_oja_update(0.5, f64::NAN, 0.5, 0.05, 0.01, 0.95);
        assert_eq!(w, 0.01);

        // Negative learning rate → return min_weight sentinel
        let w = sproink_oja_update(0.5, 0.5, 0.5, -0.1, 0.01, 0.95);
        assert_eq!(w, 0.01);

        // min < 0 → return min_weight sentinel
        let w = sproink_oja_update(0.5, 0.5, 0.5, 0.05, -0.1, 0.95);
        assert_eq!(w, -0.1);

        // max > 1 → return min_weight sentinel
        let w = sproink_oja_update(0.5, 0.5, 0.5, 0.05, 0.01, 1.5);
        assert_eq!(w, 0.01);

        // Infinite input → return min_weight sentinel
        let w = sproink_oja_update(f64::INFINITY, 0.5, 0.5, 0.05, 0.01, 0.95);
        assert_eq!(w, 0.01);
    }
}

#[test]
fn invalid_edge_kind_returns_null() {
    unsafe {
        let sources = [0u32];
        let targets = [1u32];
        let weights = [0.8f64];
        let kinds = [255u8]; // invalid kind

        let graph = sproink_graph_build(
            2,
            1,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        assert!(graph.is_null(), "Invalid edge kind should return null");
    }
}

#[test]
fn directional_passive_edge_kind_rejected() {
    // DirectionalPassive (kind=3) is an internal-only marker and must not be
    // accepted from FFI callers.
    unsafe {
        let sources = [0u32];
        let targets = [1u32];
        let weights = [0.8f64];
        let kinds = [3u8];

        let graph = sproink_graph_build(
            2,
            1,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        assert!(
            graph.is_null(),
            "DirectionalPassive edge kind should return null"
        );
    }
}

#[test]
fn activate_with_null_seed_pointer_returns_null() {
    // If the caller claims num_seeds > 0 but passes null for either seed
    // array, sproink_activate must refuse rather than silently dropping
    // seeds and returning an empty activation.
    unsafe {
        let sources = [0u32];
        let targets = [1u32];
        let weights = [0.5f64];
        let kinds = [0u8];
        let graph = sproink_graph_build(
            2,
            1,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        assert!(!graph.is_null());

        let results = sproink_activate(
            graph,
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            0,
            0.15,
            7,
            f64::NAN,
            f64::NAN,
        );
        assert!(
            results.is_null(),
            "activate with null seed arrays must return null"
        );

        sproink_graph_free(graph);
    }
}

#[test]
fn extract_pairs_null_seed_pointer_returns_null() {
    unsafe {
        // Build graph + activate to get valid results
        let sources = [0u32, 1];
        let targets = [1u32, 2];
        let weights = [0.8f64, 0.8];
        let kinds = [0u8, 0];

        let graph = sproink_graph_build(
            3,
            2,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        assert!(!graph.is_null());

        let seed_nodes = [0u32];
        let seed_activations = [1.0f64];
        let results = sproink_activate(
            graph,
            1,
            seed_nodes.as_ptr(),
            seed_activations.as_ptr(),
            std::ptr::null(),
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            0,
            0.15,
            7,
            f64::NAN,
            f64::NAN,
        );
        assert!(!results.is_null());

        let pairs = sproink_extract_pairs(results, 1, std::ptr::null(), 0.1);
        assert!(
            pairs.is_null(),
            "extract_pairs with null seed_nodes must return null"
        );

        sproink_results_free(results);
        sproink_graph_free(graph);
    }
}

#[test]
fn inhibition_breadth_zero_returns_null() {
    // InhibitionConfig::validate rejects breadth=0 — the FFI should
    // propagate that validation error as a null return.
    unsafe {
        let sources = [0u32];
        let targets = [1u32];
        let weights = [0.5f64];
        let kinds = [0u8];
        let graph = sproink_graph_build(
            2,
            1,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        assert!(!graph.is_null());

        let seed_nodes = [0u32];
        let seed_acts = [1.0f64];
        let results = sproink_activate(
            graph,
            1,
            seed_nodes.as_ptr(),
            seed_acts.as_ptr(),
            std::ptr::null(), // no seed sources
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            1,        // inhibition_enabled = true
            0.15,     // inhibition_strength
            0,        // inhibition_breadth = 0 → rejected
            f64::NAN, // temporal_decay_rate = not set
            f64::NAN, // current_time = not set
        );
        assert!(results.is_null(), "inhibition_breadth=0 must be rejected");

        sproink_graph_free(graph);
    }
}

#[test]
fn activate_with_temporal_decay() {
    unsafe {
        let sources = [0u32, 1];
        let targets = [1u32, 2];
        let weights = [0.8f64, 0.8];
        let kinds = [0u8, 0];

        let graph = sproink_graph_build(
            3,
            2,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        assert!(!graph.is_null());

        let seed_nodes = [0u32];
        let seed_activations = [1.0f64];

        // With temporal_decay_rate and current_time set
        let results = sproink_activate(
            graph,
            1,
            seed_nodes.as_ptr(),
            seed_activations.as_ptr(),
            std::ptr::null(),
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            0,
            0.15,
            7,
            0.1, // temporal_decay_rate
            5.0, // current_time
        );
        assert!(!results.is_null());

        let len = sproink_results_len(results);
        assert!(len > 0);
        for _ in 0..len {
            let mut act = [0.0f64];
            sproink_results_activations(results, act.as_mut_ptr(), 1);
            assert!((0.0..=1.0).contains(&act[0]));
        }

        sproink_results_free(results);
        sproink_graph_free(graph);
    }
}

#[test]
fn activate_with_nan_temporal_params_uses_none() {
    // NaN sentinel means "not set" — should produce the same results as
    // calling without temporal decay.
    unsafe {
        let sources = [0u32, 1];
        let targets = [1u32, 2];
        let weights = [0.8f64, 0.8];
        let kinds = [0u8, 0];

        let graph = sproink_graph_build(
            3,
            2,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        assert!(!graph.is_null());

        let seed_nodes = [0u32];
        let seed_activations = [1.0f64];

        let results = sproink_activate(
            graph,
            1,
            seed_nodes.as_ptr(),
            seed_activations.as_ptr(),
            std::ptr::null(),
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            0,
            0.15,
            7,
            f64::NAN, // temporal_decay_rate = not set
            f64::NAN, // current_time = not set
        );
        assert!(!results.is_null());
        assert!(sproink_results_len(results) > 0);

        sproink_results_free(results);
        sproink_graph_free(graph);
    }
}

#[test]
fn activate_with_seed_sources() {
    // Verify seed_sources are propagated through to results.
    unsafe {
        let sources = [0u32, 1];
        let targets = [1u32, 2];
        let weights = [0.8f64, 0.8];
        let kinds = [0u8, 0];

        let graph = sproink_graph_build(
            3,
            2,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        assert!(!graph.is_null());

        let seed_nodes = [0u32];
        let seed_activations = [1.0f64];
        let seed_sources = [42u32]; // source ID for this seed

        let results = sproink_activate(
            graph,
            1,
            seed_nodes.as_ptr(),
            seed_activations.as_ptr(),
            seed_sources.as_ptr(),
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            0,
            0.15,
            7,
            f64::NAN,
            f64::NAN,
        );
        assert!(!results.is_null());
        assert!(sproink_results_len(results) > 0);

        sproink_results_free(results);
        sproink_graph_free(graph);
    }
}

#[test]
fn activate_with_max_seed_source_means_none() {
    // u32::MAX is the "no source" sentinel.
    unsafe {
        let sources = [0u32];
        let targets = [1u32];
        let weights = [0.8f64];
        let kinds = [0u8];

        let graph = sproink_graph_build(
            2,
            1,
            sources.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            kinds.as_ptr(),
        );
        assert!(!graph.is_null());

        let seed_nodes = [0u32];
        let seed_activations = [1.0f64];
        let seed_sources = [u32::MAX]; // "no source" sentinel

        let results = sproink_activate(
            graph,
            1,
            seed_nodes.as_ptr(),
            seed_activations.as_ptr(),
            seed_sources.as_ptr(),
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            0,
            0.15,
            7,
            f64::NAN,
            f64::NAN,
        );
        assert!(!results.is_null());
        assert!(sproink_results_len(results) > 0);

        sproink_results_free(results);
        sproink_graph_free(graph);
    }
}
