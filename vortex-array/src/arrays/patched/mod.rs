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
//! each chunk, we assign each position to a lane. The number of lanes depends on the width of
//! the underlying type.
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

use std::env;
use std::sync::LazyLock;

pub use array::*;
pub(crate) use compute::aggregate::patched_aggregate_kernels;
pub use execute::patch_validity;
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

/// Flag indicating if experimental patched array support is enabled.
///
/// This is set using the environment variable `VORTEX_EXPERIMENTAL_PATCHED_ARRAY`.
///
/// When this is true, any arrays with interior `Patches` will be read as a `Patched`
/// array, and eliminate the interior patches.
///
/// The builtin compressor will also generate Patched arrays.
pub fn use_experimental_patches() -> bool {
    static USE_EXPERIMENTAL_PATCHES: LazyLock<bool> =
        LazyLock::new(|| env::var("VORTEX_EXPERIMENTAL_PATCHED_ARRAY").is_ok_and(|v| v == "1"));
    *USE_EXPERIMENTAL_PATCHES
}
