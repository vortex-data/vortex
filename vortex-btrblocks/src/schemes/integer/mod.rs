// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer compression schemes.

mod affine;
mod bitpacking;
mod chunk_delta;
mod delta;
mod for_;
mod rle;
mod runend;
mod sequence;
mod sparse;
mod var_bitpacking;
mod zigzag;

#[cfg(feature = "pco")]
mod pco;

pub use affine::AffineScheme;
pub use affine::ModeChoice;
pub use affine::choose_mode;
pub use bitpacking::BitPackingScheme;
pub use chunk_delta::ChunkDeltaScheme;
pub use delta::DeltaScheme;
pub(crate) use for_::FOR_V1;
pub use for_::FoRScheme;
#[cfg(feature = "pco")]
pub use pco::PcoScheme;
pub use rle::IntRLEScheme;
pub(crate) use rle::rle_compress;
pub use runend::RunEndScheme;
pub use sequence::SequenceScheme;
pub use sparse::SparseScheme;
pub use var_bitpacking::VarBitPackingScheme;
// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::IntDictScheme;
pub use vortex_compressor::stats::IntegerStats;
pub use zigzag::ZigZagScheme;

/// Threshold for the average run length in an array before we consider run-length encoding.
pub(crate) const RUN_LENGTH_THRESHOLD: u32 = 4;
