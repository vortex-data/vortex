//! Build-time error type for the lowering API.
//!
//! Reuses `EngineError` from the existing engine error module for the
//! runtime side, but defines `BuildError` separately for lowering-time
//! diagnostics. This keeps the lowering API independent of the
//! turn-based scheduler's existing error vocabulary.

use std::error::Error;
use std::fmt::Display;
use std::fmt::{self};

pub type BuildResult<T> = Result<T, BuildError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    Message(String),
    OutputContractMismatch { expected: String, actual: String },
    MissingDomain(String),
    InvalidPipelineDependency(String),
}

impl BuildError {
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => write!(f, "{message}"),
            Self::OutputContractMismatch { expected, actual } => write!(
                f,
                "output contract mismatch: expected {expected}, got {actual}"
            ),
            Self::MissingDomain(domain) => write!(f, "missing domain {domain}"),
            Self::InvalidPipelineDependency(message) => {
                write!(f, "invalid pipeline dependency: {message}")
            }
        }
    }
}

impl Error for BuildError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanValidationError {
    MissingRoot,
    DomainInvalid(String),
    OutputContractMismatch(String),
    PipelineLoweringInvalid(String),
    UnsupportedOperator(String),
}
