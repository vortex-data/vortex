// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! String compression schemes.

mod fsst;
mod sparse;

#[cfg(feature = "zstd")]
mod zstd;
#[cfg(feature = "zstd")]
mod zstd_buffers;

mod onpair;

pub use fsst::FSSTScheme;
pub use onpair::OnPairScheme;
pub use sparse::NullDominatedSparseScheme;
// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::StringDictScheme;
pub use vortex_compressor::stats::StringStats;
#[cfg(feature = "zstd")]
pub use zstd::ZstdScheme;
#[cfg(feature = "zstd")]
pub use zstd_buffers::ZstdBuffersScheme;

#[cfg(test)]
mod scheme_selection_tests;
#[cfg(test)]
mod tests;
