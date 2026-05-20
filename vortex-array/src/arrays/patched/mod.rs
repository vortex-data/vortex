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
//! Patch indices and values are kept in their natural sorted (untransposed) layout, exactly like
//! the [`Patches`](crate::patches::Patches) helper. To allow constant-time seeking to the patches
//! belonging to a given chunk, we additionally store a `chunk_offsets` array holding one offset
//! per 1024-element chunk.
//!
//! The Patched array layout has 4 children
//!
//! * `inner`: the inner array is the one containing encoded values, including the filler values
//!   that need to be patched over at execution time
//! * `patch_indices`: a sorted array of unsigned global indices indicating which positions of
//!   `inner` should be overwritten by the patch value
//! * `patch_values`: the child array containing the patch values, which should be inserted over
//!   the values of the `inner` at the locations provided by `patch_indices`
//! * `chunk_offsets`: an indexing buffer with one entry per 1024-element chunk, so that the
//!   patches for chunk `c` are `patch_indices[chunk_offsets[c]..chunk_offsets[c + 1]]`
//!
//! `patch_indices` and `patch_values` are aligned and accessed together.
//!
//! The number of lanes that *would* be used if these patches were transposed into the
//! data-parallel GPU layout is retained as `n_lanes` metadata, but no transpose is performed: the
//! patches are stored untransposed.

mod array;
mod compute;
mod vtable;

use std::env;
use std::sync::LazyLock;

pub use array::*;
pub use vtable::*;

/// Number of lanes that would be used at patch time for a value of type `V` if the patches were
/// transposed into the data-parallel GPU layout.
///
/// This is *NOT* equal to the number of FastLanes lanes for the type `V`, rather this is going to
/// correspond to how many "lanes" we would end up copying data on.
///
/// The patches themselves are stored untransposed; this value is retained only as metadata.
pub(crate) const fn patch_lanes<V: Sized>() -> usize {
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
