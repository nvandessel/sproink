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
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            true,
            0.15,
            7,
        );
        assert!(!results.is_null());

        let len = sproink_results_len(results);
        assert!(len > 0);

        let mut nodes = vec![0u32; len as usize];
        let mut activations = vec![0.0f64; len as usize];
        let mut distances = vec![0u32; len as usize];
        sproink_results_nodes(results, nodes.as_mut_ptr());
        sproink_results_activations(results, activations.as_mut_ptr());
        sproink_results_distances(results, distances.as_mut_ptr());

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
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            true,
            0.15,
            7,
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
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            false,
            0.15,
            7,
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
            sproink_pairs_nodes(pairs, nodes_a.as_mut_ptr(), nodes_b.as_mut_ptr());
            sproink_pairs_activations(pairs, acts_a.as_mut_ptr(), acts_b.as_mut_ptr());

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
        // kinds: 0=Positive, 1=Conflicts, 2=DirSuppressive, 4=FeatureAffinity, 255=unknown→Positive
        let sources = [0u32, 0, 0, 0, 0];
        let targets = [1u32, 2, 3, 4, 5];
        let weights = [0.8, 0.5, 0.5, 0.3, 0.7];
        let kinds = [0u8, 1, 2, 4, 255];

        let graph = sproink_graph_build(
            6,
            5,
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
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            true,
            0.15,
            7,
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
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            false,
            0.15,
            7,
        );

        // Null output pointers should not crash
        sproink_results_nodes(results, std::ptr::null_mut());
        sproink_results_activations(results, std::ptr::null_mut());
        sproink_results_distances(results, std::ptr::null_mut());

        // Null pairs accessors
        sproink_pairs_nodes(std::ptr::null(), std::ptr::null_mut(), std::ptr::null_mut());
        sproink_pairs_activations(std::ptr::null(), std::ptr::null_mut(), std::ptr::null_mut());

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
            3,
            0.7,
            0.85,
            0.01,
            10.0,
            0.3,
            false,
            0.15,
            7,
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
        assert!(w >= 0.01 && w <= 0.95);
        assert!(!w.is_nan());
    }
}
