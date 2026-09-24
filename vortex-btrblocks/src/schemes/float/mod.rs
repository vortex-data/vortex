// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float compression schemes.

pub use vortex_alp::schemes::alp::ALPScheme;
pub use vortex_alp::schemes::alprd::ALPRDScheme;
// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::FloatDictScheme;
pub use vortex_compressor::stats::FloatStats;
pub use vortex_fastlanes::schemes::float_rle::FloatRLEScheme;
#[cfg(feature = "pco")]
pub use vortex_pco::schemes::float::PcoScheme;
pub use vortex_sparse::schemes::float::NullDominatedSparseScheme;

#[cfg(test)]
mod scheme_selection_tests;
#[cfg(test)]
mod tests;
