// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex array implementing pco's `Dict` mode: each input value is
//! represented as an index into a small dictionary of unique values, so
//! `out[i] = dict[indices[i]]`. The dictionary lives in a single buffer of
//! raw native bytes; the indices live in a `Primitive<L_idx>` child whose
//! width (`u8`/`u16`/`u32`) is chosen automatically based on dict cardinality.
//!
//! Only integer primitives (`u8`/`u16`/`u32`/`u64`/`i8`/`i16`/`i32`/`i64`)
//! are supported in this phase; float dicts are deferred because of
//! bit-equality issues around NaN.
//!
//! See [`PcoDictArray`] for the public type.

pub use array::*;

mod array;

/// Prost-encoded metadata for a [`PcoDictArray`].
///
/// `dict_len` is the number of distinct entries in the dictionary;
/// `idx_width` is the byte-width of the `indices` child (`1`, `2`, or `4`)
/// and is stored explicitly so a decoder can dispatch without having to
/// peek at the indices child's `PType`.
#[derive(Clone, prost::Message)]
pub struct PcoDictMetadata {
    /// Number of distinct entries stored in the dictionary buffer.
    #[prost(uint32, tag = "1")]
    pub dict_len: u32,
    /// Byte-width of each entry in the `indices` child (`1`, `2`, or `4`).
    #[prost(uint32, tag = "2")]
    pub idx_width: u32,
}
