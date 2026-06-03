// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer compression schemes.

mod bitpacking;
mod for_;
mod rle;
mod runend;
mod sequence;
mod sparse;
mod zigzag;

#[cfg(feature = "pco")]
mod pco;

pub use bitpacking::BitPackingScheme;
pub use for_::FoRScheme;
#[cfg(feature = "pco")]
pub use pco::PcoScheme;
pub use rle::IntRLEScheme;
pub(crate) use rle::rle_compress;
#[cfg(feature = "unstable_encodings")]
pub(crate) use rle::try_compress_delta;
pub use runend::RunEndScheme;
pub use sequence::SequenceScheme;
pub use sparse::SparseScheme;
// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::IntConstantScheme;
pub use vortex_compressor::builtins::IntDictScheme;
pub use vortex_compressor::stats::IntegerStats;
pub use zigzag::ZigZagScheme;

/// Threshold for the average run length in an array before we consider run-length encoding.
pub(crate) const RUN_LENGTH_THRESHOLD: u32 = 4;

#[cfg(test)]
mod scheme_selection_tests;
#[cfg(test)]
mod tests;
