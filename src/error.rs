use crate::types::NodeId;

#[derive(Debug, thiserror::Error)]
pub enum SproinkError {
    #[error("invalid {field}: {value} (expected finite value in valid range)")]
    InvalidValue { field: &'static str, value: f64 },

    #[error("node {node:?} out of bounds (graph has {num_nodes} nodes)")]
    NodeOutOfBounds { node: NodeId, num_nodes: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_value_displays_field_and_value() {
        let err = SproinkError::InvalidValue {
            field: "activation",
            value: 1.5,
        };
        let msg = err.to_string();
        assert!(msg.contains("activation"));
        assert!(msg.contains("1.5"));
    }

    #[test]
    fn node_out_of_bounds_displays_details() {
        let err = SproinkError::NodeOutOfBounds {
            node: NodeId(42),
            num_nodes: 10,
        };
        let msg = err.to_string();
        assert!(msg.contains("42"));
        assert!(msg.contains("10"));
    }
}
