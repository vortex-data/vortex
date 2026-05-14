// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex arrays implementing pco's bin-partition layer for `i64` inputs
//! plus the per-bin variable-width bitpack used to store the resulting
//! offsets. This is the first layer in the layered pco stack whose primary
//! purpose is to **shrink bytes**.
//!
//! Two sibling arrays live in this crate:
//!
//! - [`VarWidthBitPackedArray`]: a primitive-integer array whose per-element
//!   bit width is variable, driven by a parallel stream of bin assignments
//!   `bin_idx[i]` and a per-bin width table. Conceptually a generalisation
//!   of fixed-width bitpacking to "width is a function of the bin index".
//! - [`BinPartitionArray`]: decomposes an `i64` input into `(bin_idx,
//!   offset)` such that `value[i] = bins[bin_idx[i]].lower + offset[i]`,
//!   where the per-bin offset width is `ceil(log2(span + 1))`.
//!
//! The two arrays share the same `bin_idx` child: `BinPartitionArray` owns
//! both children and the inner `VarWidthBitPackedArray` references
//! `bin_idx` so that random-access decode can look up the right width per
//! element.
//!
//! See [`BinPartitionArray`] and [`VarWidthBitPackedArray`] for the public
//! types.

pub use bin_partition::*;
pub use var_width::*;

mod bin_partition;
mod var_width;

/// Prost-encoded metadata for a [`VarWidthBitPackedArray`].
///
/// `widths` stores the per-bin bit width (always `<= 64`); `n_elements` is
/// the logical length of the array. The widths are stored as `u32` for
/// prost convenience even though every value fits in a `u8`.
#[derive(Clone, prost::Message)]
pub struct VarWidthBitPackedMetadata {
    /// Per-bin bit widths (one entry per bin). Each value must be `<= 64`.
    #[prost(uint32, repeated, tag = "1")]
    pub widths: Vec<u32>,
    /// Logical number of elements in the array.
    #[prost(uint64, tag = "2")]
    pub n_elements: u64,
}

/// One bin in a [`BinPartitionArray`].
///
/// `lower` is the bin's inclusive lower bound; `offset_bits` is the number
/// of bits the [`VarWidthBitPackedArray`] uses to store offsets in this
/// bin. The bin's exclusive upper bound is `lower + (1 << offset_bits)`.
#[derive(Clone, prost::Message)]
pub struct BinInfo {
    /// Inclusive lower bound of the bin.
    #[prost(int64, tag = "1")]
    pub lower: i64,
    /// Number of bits used to encode offsets within this bin.
    #[prost(uint32, tag = "2")]
    pub offset_bits: u32,
}

/// Prost-encoded metadata for a [`BinPartitionArray`].
///
/// The number of bins (`bins.len()`) must be in `1..=256` so that the
/// `bin_idx` child fits in `u8`.
#[derive(Clone, prost::Message)]
pub struct BinPartitionMetadata {
    /// The per-bin lower bound and offset bit-width.
    #[prost(message, repeated, tag = "1")]
    pub bins: Vec<BinInfo>,
}
