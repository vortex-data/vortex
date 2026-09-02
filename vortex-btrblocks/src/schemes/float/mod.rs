// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float compression schemes.

mod alp;
mod alprd;
mod float_quant;
mod ordered_block_residual;
mod rle;
mod sparse;

#[cfg(feature = "pco")]
mod pco;

pub use alp::ALPScheme;
pub use alprd::ALPRDScheme;
pub use float_quant::FloatQuantScheme;
pub use ordered_block_residual::OrderedBlockResidualScheme;
#[cfg(feature = "pco")]
pub use pco::PcoScheme;
pub use rle::FloatRLEScheme;
pub use sparse::NullDominatedSparseScheme;
// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::FloatDictScheme;
pub use vortex_compressor::stats::FloatStats;

#[cfg(test)]
mod scheme_selection_tests;
#[cfg(test)]
mod tests;
