// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! String compression schemes.

mod fsst;
mod sparse;

#[cfg(feature = "zstd")]
mod zstd;
#[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
mod zstd_buffers;

#[cfg(feature = "unstable_encodings")]
mod onpair;

pub use fsst::FSSTScheme;
pub use sparse::NullDominatedSparseScheme;

#[cfg(feature = "zstd")]
pub use zstd::ZstdScheme;
#[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
pub use zstd_buffers::ZstdBuffersScheme;

#[cfg(feature = "unstable_encodings")]
pub use onpair::OnPairScheme;

// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::StringConstantScheme;
pub use vortex_compressor::builtins::StringDictScheme;
pub use vortex_compressor::stats::StringStats;

#[cfg(test)]
mod scheme_selection_tests;
#[cfg(test)]
mod tests;
