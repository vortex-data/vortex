// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Error types for nvcomp operations.

use crate::sys;

/// Error type for nvcomp operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvcompError {
    /// Invalid value provided.
    InvalidValue,
    /// Operation not supported.
    NotSupported,
    /// Cannot decompress the data.
    CannotDecompress,
    /// Bad checksum in compressed data.
    BadChecksum,
    /// Cannot verify checksums.
    CannotVerifyChecksums,
    /// Output buffer too small.
    OutputBufferTooSmall,
    /// Wrong header length.
    WrongHeaderLength,
    /// Alignment error.
    Alignment,
    /// Chunk size too large.
    ChunkSizeTooLarge,
    /// Cannot compress the data.
    CannotCompress,
    /// Wrong input length.
    WrongInputLength,
    /// Batch size too large.
    BatchSizeTooLarge,
    /// CUDA error.
    CudaError,
    /// Internal error in nvcomp.
    InternalError,
    /// Unknown error with raw status code.
    Unknown(u32),
}

impl std::fmt::Display for NvcompError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidValue => write!(f, "nvcomp: invalid value"),
            Self::NotSupported => write!(f, "nvcomp: operation not supported"),
            Self::CannotDecompress => write!(f, "nvcomp: cannot decompress"),
            Self::BadChecksum => write!(f, "nvcomp: bad checksum"),
            Self::CannotVerifyChecksums => write!(f, "nvcomp: cannot verify checksums"),
            Self::OutputBufferTooSmall => write!(f, "nvcomp: output buffer too small"),
            Self::WrongHeaderLength => write!(f, "nvcomp: wrong header length"),
            Self::Alignment => write!(f, "nvcomp: alignment error"),
            Self::ChunkSizeTooLarge => write!(f, "nvcomp: chunk size too large"),
            Self::CannotCompress => write!(f, "nvcomp: cannot compress"),
            Self::WrongInputLength => write!(f, "nvcomp: wrong input length"),
            Self::BatchSizeTooLarge => write!(f, "nvcomp: batch size too large"),
            Self::CudaError => write!(f, "nvcomp: CUDA error"),
            Self::InternalError => write!(f, "nvcomp: internal error"),
            Self::Unknown(code) => write!(f, "nvcomp: unknown error (status code {})", code),
        }
    }
}

impl std::error::Error for NvcompError {}

/// Checks an nvcomp status code and converts it to a Result.
pub(crate) fn check_status(status: sys::nvcompStatus_t) -> Result<(), NvcompError> {
    match status {
        sys::nvcompStatus_t_nvcompSuccess => Ok(()),
        sys::nvcompStatus_t_nvcompErrorInvalidValue => Err(NvcompError::InvalidValue),
        sys::nvcompStatus_t_nvcompErrorNotSupported => Err(NvcompError::NotSupported),
        sys::nvcompStatus_t_nvcompErrorCannotDecompress => Err(NvcompError::CannotDecompress),
        sys::nvcompStatus_t_nvcompErrorBadChecksum => Err(NvcompError::BadChecksum),
        sys::nvcompStatus_t_nvcompErrorCannotVerifyChecksums => {
            Err(NvcompError::CannotVerifyChecksums)
        }
        sys::nvcompStatus_t_nvcompErrorOutputBufferTooSmall => {
            Err(NvcompError::OutputBufferTooSmall)
        }
        sys::nvcompStatus_t_nvcompErrorWrongHeaderLength => Err(NvcompError::WrongHeaderLength),
        sys::nvcompStatus_t_nvcompErrorAlignment => Err(NvcompError::Alignment),
        sys::nvcompStatus_t_nvcompErrorChunkSizeTooLarge => Err(NvcompError::ChunkSizeTooLarge),
        sys::nvcompStatus_t_nvcompErrorCannotCompress => Err(NvcompError::CannotCompress),
        sys::nvcompStatus_t_nvcompErrorWrongInputLength => Err(NvcompError::WrongInputLength),
        sys::nvcompStatus_t_nvcompErrorBatchSizeTooLarge => Err(NvcompError::BatchSizeTooLarge),
        sys::nvcompStatus_t_nvcompErrorCudaError => Err(NvcompError::CudaError),
        sys::nvcompStatus_t_nvcompErrorInternal => Err(NvcompError::InternalError),
        code => Err(NvcompError::Unknown(code)),
    }
}
