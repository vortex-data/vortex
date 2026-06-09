// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary compression schemes.

#[cfg(feature = "zstd")]
mod zstd;
#[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
mod zstd_buffers;

// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::BinaryConstantScheme;
pub use vortex_compressor::builtins::BinaryDictScheme;
#[cfg(feature = "zstd")]
pub use zstd::ZstdScheme;
#[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
pub use zstd_buffers::ZstdBuffersScheme;
