// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CubError {
    /// Failed to load the CUB library at runtime.
    LibraryLoadError(String),
    /// CUDA returned an error code.
    CudaError {
        /// The CUDA error code.
        code: i32,
        /// Context describing what operation failed.
        context: String,
    },
}

impl std::fmt::Display for CubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LibraryLoadError(msg) => write!(f, "cub: failed to load library: {}", msg),
            Self::CudaError { code, context } => {
                write!(f, "cub: {} failed with CUDA error code {}", context, code)
            }
        }
    }
}

impl std::error::Error for CubError {}

/// Checks a CUDA error code and converts it to a Result.
pub(crate) fn check_cuda_error(err: i32, context: &str) -> Result<(), CubError> {
    if err == 0 {
        Ok(())
    } else {
        Err(CubError::CudaError {
            code: err,
            context: context.to_string(),
        })
    }
}
