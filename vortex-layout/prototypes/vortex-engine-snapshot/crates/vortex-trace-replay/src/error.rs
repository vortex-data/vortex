use thiserror::Error;

use crate::TimelinePos;

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("file does not start with VTRX magic")]
    BadMagic,

    #[error(
        "unsupported trace format version: file is v{found}, this replayer expects v{expected}"
    )]
    UnsupportedVersion { found: u32, expected: u32 },

    #[error("trace file truncated at offset {offset}")]
    Truncated { offset: u64 },

    #[error("postcard decode error: {0}")]
    Postcard(#[from] postcard::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid timeline position: {0:?}")]
    InvalidPosition(TimelinePos),

    #[error("invalid record kind byte: {0}")]
    InvalidRecordKind(u8),
}
