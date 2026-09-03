// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An array that partially "patches" another array with new values.
//!
//! # Background
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
//! # Layout
//!
//! Patches are addressed by **chunk-local indices**. Logical row `i` lives at grid position
//! `offset + i`, in chunk `(offset + i) / 1024` at local index `(offset + i) % 1024`, so the
//! index child stays two bytes per patch at any array length.
//!
//! The Patched array layout has 4 children
//!
//! * `inner`: the inner array containing encoded values, including the filler values that need to
//!   be patched over at execution time
//! * `patch_indices`: `u16` positions within a chunk, strictly increasing within each chunk
//! * `patch_values`: the values that overwrite the `inner` at the locations given by
//!   `patch_indices`
//! * `chunk_offsets`: `u32` prefix patch counts, one per chunk plus a terminator, so chunk `c`
//!   owns `patch_indices[chunk_offsets[c]..chunk_offsets[c + 1]]`
//!
//! `patch_indices` and `patch_values` are aligned and accessed together.
//!
//! ```text
//!               chunk 0        chunk 1        chunk 2
//!              ┌──────────────┬──────────────┬──────────────┬─────┐
//! chunk_offsets│      0       │      1       │      3       │  5  │
//!              └──────┬───────┴──────┬───────┴──────┬───────┴─────┘
//!                     │              │              │
//!                     ▼              ▼              ▼
//!              ┌──────┬──────┬──────┬──────┬──────┐
//! patch_indices│  5   │  6   │  7   │ 152  │ 951  │   (row 1030 is local 6 of chunk 1)
//!              ├──────┼──────┼──────┼──────┼──────┤
//! patch_values │ 100  │ 200  │ 300  │ 400  │ 500  │
//!              └──────┴──────┴──────┴──────┴──────┘
//! ```
//!
//! Point lookups select the chunk in constant time and binary search at most 1024 indices.
//!
//! # Slicing
//!
//! Slicing keeps the patches on their original grid. The `inner` is sliced exactly and
//! `chunk_offsets` is sliced to the covered chunks, while `patch_indices` and `patch_values` are
//! shared unchanged. Patches of the first and last chunk whose row falls outside the slice stay
//! in the children and are skipped on read, so slicing never reads or copies patch data. The
//! `offset` records where the first row sits inside its chunk.
//!
//! # Wire format
//!
//! The chunk-local layout is serialized under `vortex.patched_v2`. The retired lane-transposed
//! layout of RFC 0027 is still readable under `vortex.patched` and is re-sorted into chunk order
//! on load; see [`PatchedPlugin`].

mod array;
mod compute;
mod layout;
mod plugin;
#[cfg(test)]
mod tests;
mod vtable;

use std::env;
use std::sync::LazyLock;

pub use array::*;
pub use layout::PatchedView;
pub use plugin::PatchedPlugin;
pub use plugin::patched_v2_id;
pub use vtable::*;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
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
