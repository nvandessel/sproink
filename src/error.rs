//! Error types for the sproink engine.

/// Errors that can occur during graph construction or propagation.
#[derive(Debug, thiserror::Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SproinkError {
    #[error("invalid {field}: {value} (expected finite value in valid range)")]
    InvalidValue { field: &'static str, value: f64 },
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
}
