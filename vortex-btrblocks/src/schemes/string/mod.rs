// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! String compression schemes.

// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::StringDictScheme;
pub use vortex_compressor::stats::StringStats;
pub use vortex_fsst::schemes::fsst::FSSTScheme;
pub use vortex_onpair::schemes::onpair::OnPairScheme;
pub use vortex_sparse::schemes::string::NullDominatedSparseScheme;
#[cfg(feature = "zstd")]
pub use vortex_zstd::schemes::string::ZstdScheme;
#[cfg(feature = "zstd")]
pub use vortex_zstd::schemes::string_buffers::ZstdBuffersScheme;

#[cfg(test)]
mod scheme_selection_tests;
#[cfg(test)]
mod tests;
