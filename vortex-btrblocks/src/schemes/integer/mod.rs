// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer compression schemes.

// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::IntDictScheme;
pub use vortex_compressor::stats::IntegerStats;
pub use vortex_fastlanes::schemes::bitpacking::BitPackingScheme;
pub use vortex_fastlanes::schemes::delta::DeltaScheme;
pub use vortex_fastlanes::schemes::for_::FoRScheme;
pub use vortex_fastlanes::schemes::integer_rle::IntRLEScheme;
#[cfg(feature = "pco")]
pub use vortex_pco::schemes::integer::PcoScheme;
pub use vortex_runend::schemes::runend::RunEndScheme;
pub use vortex_sequence::schemes::sequence::SequenceScheme;
pub use vortex_sparse::schemes::integer::SparseScheme;
pub use vortex_zigzag::schemes::zigzag::ZigZagScheme;

#[cfg(test)]
mod scheme_selection_tests;
#[cfg(test)]
mod tests;
