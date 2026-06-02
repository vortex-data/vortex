// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float compression schemes.

mod alp;
mod alprd;
mod rle;
mod sparse;

#[cfg(feature = "pco")]
mod pco;

pub use alp::ALPScheme;
pub use alprd::ALPRDScheme;
#[cfg(feature = "pco")]
pub use pco::PcoScheme;
pub use rle::FloatRLEScheme;
pub use sparse::NullDominatedSparseScheme;
// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::FloatConstantScheme;
pub use vortex_compressor::builtins::FloatDictScheme;
pub use vortex_compressor::stats::FloatStats;

#[cfg(test)]
mod scheme_selection_tests;
#[cfg(test)]
mod tests;
