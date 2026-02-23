//! Error types for deSnark.

use displaydoc::Display;

/// Errors for the deSnark protocol.
#[derive(Display, Debug)]
pub enum DeSnarkError {
    /// HyperPlonk error: {0}
    HyperPlonkError(String),
    /// Invalid parameters: {0}
    InvalidParameters(String),
    /// Network error: {0}
    NetworkError(String),
}

impl std::error::Error for DeSnarkError {}

