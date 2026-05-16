use std::error::Error;
use std::fmt::Display;
use std::fmt::{self};

pub type EngineResult<T> = Result<T, EngineError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    Message(String),
    InvalidGraph(String),
    InvalidRequirement(String),
}

impl EngineError {
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => write!(f, "{message}"),
            Self::InvalidGraph(message) => write!(f, "invalid operator graph: {message}"),
            Self::InvalidRequirement(message) => write!(f, "invalid requirement: {message}"),
        }
    }
}

impl Error for EngineError {}
