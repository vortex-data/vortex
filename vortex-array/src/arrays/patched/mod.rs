// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod compute;
mod vtable;

pub use array::*;
use vortex_buffer::ByteBuffer;
pub use vtable::*;

/// Patches that have been transposed into GPU format.
struct TransposedPatches {
    n_chunks: usize,
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

pub struct PatchAccessor<'a> {
    n_lanes: usize,
    lane_offsets: &'a [u32],
    indices: &'a [u16],
}

pub struct PatchOffset {
    /// Global offset into the list of patches. These are some of the
    pub index: usize,
    /// This is the value stored in the `indices` buffer, which encodes the offset of the `index`-th
    /// patch
    pub chunk_offset: u16,
}

impl<'a> PatchAccessor<'a> {
    /// Get an iterator over indices and values offsets.
    ///
    /// The first component is the index into the `indices` and `values`, and the second component
    /// is the set of values instead here...I think?
    pub fn offsets_iter(
        &self,
        chunk: usize,
        lane: usize,
    ) -> impl Iterator<Item = (usize, u16)> + '_ {
        let start = self.lane_offsets[chunk * self.n_lanes + lane] as usize;
        let stop = self.lane_offsets[chunk * self.n_lanes + lane + 1] as usize;

        std::iter::zip(start..stop, self.indices[start..stop].iter().copied())
    }
}

pub struct LanePatches<'a, V> {
    pub indices: &'a [u16],
    pub values: &'a [V],
}

impl<'a, V: Copy> LanePatches<'a, V> {
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (u16, V)> {
        self.indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }
}
