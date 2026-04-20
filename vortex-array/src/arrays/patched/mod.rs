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
mod vtable;

use std::env;
use std::sync::LazyLock;

pub use array::*;
use vortex_buffer::ByteBuffer;
pub use vtable::*;

/// Patches that have been transposed into GPU format.
struct TransposedPatches {
    n_lanes: usize,
    lane_offsets: ByteBuffer,
    indices: ByteBuffer,
    values: ByteBuffer,
}

/// Number of lanes used at patch time for a value of type `V`.
///
/// This is *NOT* equal to the number of FastLanes lanes for the type `V`, rather this is going to
/// correspond to how many "lanes" we will end up copying data on.
///
/// When applied on the CPU, this configuration doesn't really matter. On the GPU, it is based
/// on the number of patches involved here.
const fn patch_lanes<V: Sized>() -> usize {
    // For types 32-bits or smaller, we use a 32 lane configuration, and for 64-bit we use 16 lanes.
    // This matches up with the number of lanes we use to execute copying results from bit-unpacking
    // from shared to global memory.
    if size_of::<V>() < 8 { 32 } else { 16 }
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
