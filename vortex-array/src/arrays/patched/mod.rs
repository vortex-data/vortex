// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An array that partially "patches" another array with new values.
//!
//! # Background
//!
//! This is meant to be the foundation of a fully data-parallel patching strategy, based on the
//! work published in ["G-ALP" from Hepkema et al.](https://ir.cwi.nl/pub/35205/35205.pdf)
//!
//! Patching is common when an encoding almost completely covers an array save a few exceptions.
//! In that case, rather than avoid the encoding entirely, it's preferable to
//!
//! * Replace unencodable values with fillers (zeros, frequent values, nulls, etc.)
//! * Wrap the array with a `PatchedArray` signaling that when the original array is executed,
//!   some of the decoded values must be overwritten.
//!
//! In Vortex, the FastLanes bit-packing encoding is often the terminal node in an encoding tree,
//! and FastLanes has an intrinsic chunking of 1024 elements. Thus, 1024 elements is pervasively
//! a useful unit of chunking throughout Vortex, and so we use 1024 as a chunk size here
//! as well.
//!
//! # Details
//!
//! To patch an array, we first divide it into a set of chunks of length 1024, and then within
//! each chunk, we assign each position to a lane: position `i` belongs to lane `i % n_lanes`.
//! The number of lanes is chosen per array by [`patch_lanes_for`]: sparse patches use a single
//! lane per chunk, and dense patches up to the number of FastLanes lanes for the width of the
//! type, so that every lane groups whole FastLanes lanes.
//!
//! Thus, rather than sorting patch indices and values by their global offset, they are sorted
//! primarily by their chunk, and then subsequently by their lanes.
//!
//! The Patched array layout has 4 children
//!
//! * `inner`: the inner array is the one containing encoded values, including the filler values
//!   that need to be patched over at execution time
//! * `lane_offsets`: this is an indexing buffer that allows you to see into ranges of the other
//!   two children
//! * `indices`: An array of `u16` chunk indices, indicating where within the chunk should the value
//!   be overwritten by the patch value
//! * `values`: The child array containing the patch values, which should be inserted over
//!   the values of the `inner` at the locations provided by `indices`
//!
//! `indices` and `values` are aligned and accessed together.
//!
//! The array and its patch values may have any dtype, and the patch values may be null. A patched
//! array over a constant `inner` is the equivalent of a sparse array ([`Patched::sparse`]), and
//! one whose patch values are a null constant is the equivalent of a masked array with sparse
//! nulls ([`Patched::with_nulls`]).
//!
//! ```text
//! 
//!                  chunk 0      chunk 0      chunk 0     chunk 0       chunk 0     chunk 0
//!                  lane  0      lane 1       lane  2     lane 3        lane  4     lane  5
//!              ┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
//! lane_offsets │     0      │     0      │     2      │     2      │     3      │     5      │  ...
//!              └─────┬──────┴─────┬──────┴─────┬──────┴──────┬─────┴──────┬─────┴──────┬─────┘
//!                    │            │            │             │            │            │
//!                    │            │            │             │            │            │
//!              ┌─────┴────────────┘            └──────┬──────┘     ┌──────┘            └─────┐
//!              │                                      │            │                         │
//!              │                                      │            │                         │
//!              │                                      │            │                         │
//!              ▼────────────┬────────────┬────────────▼────────────▼────────────┬────────────▼
//!    indices   │            │            │            │            │            │            │
//!              │            │            │            │            │            │            │
//!              ├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤
//!    values    │            │            │            │            │            │            │
//!              │            │            │            │            │            │            │
//!              └────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
//! ```
//!
//! It turns out that this layout is optimal for executing patching on GPUs, because the
//! `lane_offsets` allows each thread in a warp to seek to its patches in constant time.

mod array;
mod compute;
mod execute;
mod layout;
#[cfg(test)]
mod tests;
mod vtable;

pub use array::*;
pub(crate) use compute::aggregate::patched_aggregate_kernels;
pub use execute::patch_validity;
pub use execute::scatter;
pub use layout::*;
pub use vtable::*;

use crate::dtype::DType;

/// The number of values in each patched chunk. This is the FastLanes block size.
pub const PATCH_CHUNK_SIZE: usize = 1024;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
    compute::rules::register_parent_rules(session);
}

/// The default number of lanes used to lay out the patches of an array with the given [`DType`].
///
/// For primitive types this is the number of FastLanes lanes for a value of that width (1024
/// divided by the bit width): 128 lanes for 8-bit values, 64 for 16-bit, 32 for 32-bit and 16
/// for 64-bit values. Patches therefore line up with the lanes in which FastLanes decodes values,
/// which lets a decoder apply the patches of a lane as soon as that lane has been unpacked, and
/// lets each GPU thread seek directly to the patches for its lane.
///
/// Other types have no lane structure, so their patches are grouped only by chunk.
pub fn default_patch_lanes(dtype: &DType) -> usize {
    match dtype {
        DType::Primitive(ptype, _) => PATCH_CHUNK_SIZE / ptype.bit_width(),
        _ => 1,
    }
}

/// The number of patches each lane should hold before a chunk is split into more lanes.
const PATCHES_PER_LANE: usize = 32;

/// The number of lanes to lay out `n_patches` patches of an array of `len` values of `dtype`.
///
/// Lanes let data-parallel kernels give each lane its own patches, but cost an offset per lane per
/// chunk, and executing patches in lane order rather than position order costs some locality.
/// This picks the smallest power of two of lanes that holds at most about 32 patches each on
/// average, capped at the [`default_patch_lanes`] of the type: sparse patches use a single lane per
/// chunk, and only dense patches are split across lanes, up to the FastLanes lane count. Any power
/// of two below the FastLanes lane count groups whole FastLanes lanes, since position `i` belongs
/// to lane `i % n_lanes`.
pub fn patch_lanes_for(dtype: &DType, len: usize, n_patches: usize) -> usize {
    let n_chunks = len.div_ceil(PATCH_CHUNK_SIZE).max(1);
    n_patches
        .div_ceil(n_chunks * PATCHES_PER_LANE)
        .next_power_of_two()
        .min(default_patch_lanes(dtype))
}
