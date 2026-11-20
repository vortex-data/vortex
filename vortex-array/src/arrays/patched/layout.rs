// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The chunk/lane layout of the patches in a [`Patched`](crate::arrays::Patched) array.

use std::ops::Range;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::arrays::patched::PATCH_CHUNK_SIZE;

/// A borrowed view of the executed `lane_offsets` and `indices` of a patched array.
///
/// The patches are grouped first by their 1024-element chunk and then by their lane within the
/// chunk. Position `i` of a chunk belongs to lane `i % n_lanes`. The patches of lane `l` in chunk
/// `c` are found at `lane_offsets[c * n_lanes + l]..lane_offsets[c * n_lanes + l + 1]` of the
/// `indices` and `values`, and `indices` holds each patch's position within its chunk.
///
/// `offset` is the number of leading positions of the first chunk that are outside the array,
/// and `len` the number of positions inside it. Patches of the first and last chunks may fall
/// outside the array after slicing, and are skipped.
#[derive(Debug, Clone, Copy)]
pub struct PatchLayout<'a> {
    offset: usize,
    len: usize,
    n_lanes: usize,
    lane_offsets: &'a [u32],
    indices: &'a [u16],
}

impl<'a> PatchLayout<'a> {
    /// Creates a layout, checking that `lane_offsets` covers every chunk of the array.
    pub fn try_new(
        offset: usize,
        len: usize,
        n_lanes: usize,
        lane_offsets: &'a [u32],
        indices: &'a [u16],
    ) -> VortexResult<Self> {
        let n_chunks = (offset + len).div_ceil(PATCH_CHUNK_SIZE);
        vortex_ensure!(
            lane_offsets.len() == n_chunks * n_lanes + 1,
            "expected {} lane offsets for {n_chunks} chunks of {n_lanes} lanes, found {}",
            n_chunks * n_lanes + 1,
            lane_offsets.len()
        );
        vortex_ensure!(
            n_lanes.is_power_of_two() && n_lanes <= PATCH_CHUNK_SIZE,
            "patch lanes must be a power of two of at most {PATCH_CHUNK_SIZE}, got {n_lanes}"
        );
        vortex_ensure!(
            lane_offsets.last().is_none_or(|&end| end as usize <= indices.len()),
            "lane offsets exceed the number of patches"
        );
        // Positions within a chunk are what keep writes inside the array. A bitwise OR of all
        // indices vectorizes, which keeps this check cheap enough to run on every execution.
        vortex_ensure!(
            indices.iter().fold(0u16, |acc, &i| acc | i) < PATCH_CHUNK_SIZE as u16,
            "patch indices must be less than {PATCH_CHUNK_SIZE}"
        );
        Ok(Self {
            offset,
            len,
            n_lanes,
            lane_offsets,
            indices,
        })
    }

    /// Fully validates the layout, which may come from an untrusted source: lane offsets must be
    /// non-decreasing, and every patch must belong to the lane it is stored in.
    ///
    /// [`PatchLayout::try_new`] only performs the checks needed for safe execution on the CPU.
    /// This check also covers the lane assignment that data-parallel kernels rely on, and is run
    /// when an array is deserialized.
    pub fn validate(&self) -> VortexResult<()> {
        let lane_mask = self.n_lanes - 1;
        for (slot, window) in self.lane_offsets.windows(2).enumerate() {
            let (start, end) = (window[0] as usize, window[1] as usize);
            vortex_ensure!(start <= end, "lane offsets must be non-decreasing");
            let lane = slot & lane_mask;
            for &index in &self.indices[start..end] {
                vortex_ensure!(
                    (index as usize) & lane_mask == lane,
                    "patch index {index} does not belong to lane {lane}"
                );
            }
        }
        Ok(())
    }

    /// The number of chunks spanned by the array, including the partial first and last chunks.
    #[inline]
    pub fn n_chunks(&self) -> usize {
        (self.offset + self.len).div_ceil(PATCH_CHUNK_SIZE)
    }

    /// The offset of the array into its first chunk.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The length of the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of lanes in each chunk.
    #[inline]
    pub fn n_lanes(&self) -> usize {
        self.n_lanes
    }

    /// The position within its chunk of every patch.
    #[inline]
    pub fn indices(&self) -> &'a [u16] {
        self.indices
    }

    /// The range of patches between two lane offsets.
    ///
    /// Offsets are clamped to the patches, so that a corrupt layout can produce wrong values but
    /// never index outside the patches.
    #[inline]
    fn range(&self, start: usize, end: usize) -> Range<usize> {
        let n = self.indices.len();
        let start = (self.lane_offsets[start] as usize).min(n);
        let end = (self.lane_offsets[end] as usize).min(n);
        start..end
    }

    /// The range of patches, across all lanes, in `chunk`.
    #[inline]
    pub fn chunk_patches(&self, chunk: usize) -> Range<usize> {
        self.range(chunk * self.n_lanes, (chunk + 1) * self.n_lanes)
    }

    /// The range of patches in `lane` of `chunk`.
    #[inline]
    pub fn lane_patches(&self, chunk: usize, lane: usize) -> Range<usize> {
        let idx = chunk * self.n_lanes + lane;
        self.range(idx, idx + 1)
    }

    /// The range of patches for all chunks, including any outside of the array's bounds.
    #[inline]
    pub fn all_patches(&self) -> Range<usize> {
        self.range(0, self.lane_offsets.len() - 1)
    }

    /// Calls `f(chunk_position, patch)` for every patch of `chunk`, where `chunk_position` is the
    /// patch's position within the 1024-element chunk. Patches outside the array are included.
    #[inline]
    pub fn for_each_in_chunk(&self, chunk: usize, mut f: impl FnMut(usize, usize)) {
        for patch in self.chunk_patches(chunk) {
            f(self.indices[patch] as usize, patch);
        }
    }

    /// Calls `f(position, patch)` for every patch inside the array, where `position` is the
    /// patch's index into the array and `patch` its index into the patch indices and values.
    ///
    /// Patches are visited in chunk order, but not necessarily in position order within a chunk.
    #[inline]
    pub fn for_each(&self, mut f: impl FnMut(usize, usize)) {
        let end = self.offset + self.len;
        for chunk in 0..self.n_chunks() {
            let chunk_start = chunk * PATCH_CHUNK_SIZE;
            let range = self.chunk_patches(chunk);
            if chunk_start >= self.offset && chunk_start + PATCH_CHUNK_SIZE <= end {
                // Interior chunks are entirely in bounds.
                let base = chunk_start - self.offset;
                for patch in range {
                    f(base + self.indices[patch] as usize, patch);
                }
            } else {
                for patch in range {
                    let index = chunk_start + self.indices[patch] as usize;
                    if index >= self.offset && index < end {
                        f(index - self.offset, patch);
                    }
                }
            }
        }
    }

    /// Returns `true` if any patch of the first or last chunk is outside the array.
    pub fn has_out_of_bounds_patches(&self) -> bool {
        let n_chunks = self.n_chunks();
        if n_chunks == 0 {
            return !self.all_patches().is_empty();
        }
        let end = self.offset + self.len;
        let out_of_bounds = |chunk: usize| {
            let chunk_start = chunk * PATCH_CHUNK_SIZE;
            self.chunk_patches(chunk).any(|patch| {
                let index = chunk_start + self.indices[patch] as usize;
                index < self.offset || index >= end
            })
        };
        out_of_bounds(0) || out_of_bounds(n_chunks - 1)
    }

    /// The number of patches inside the array.
    pub fn count(&self) -> usize {
        if !self.has_out_of_bounds_patches() {
            return self.all_patches().len();
        }
        let mut count = 0;
        self.for_each(|_, _| count += 1);
        count
    }
}

/// Patches transposed from position order into chunk/lane order.
pub(crate) struct TransposedPatches {
    /// The `n_chunks * n_lanes + 1` offsets of each lane's patches.
    pub lane_offsets: Buffer<u32>,
    /// The position of each patch within its chunk.
    pub indices: Buffer<u16>,
    /// For each transposed patch, the index of the input patch it came from, or `None` when the
    /// input order is already the transposed order.
    pub permutation: Option<Buffer<u32>>,
}

/// Transposes patches at the ascending positions `indices` (less `offset`) of an array of length
/// `len` into chunk/lane order.
#[expect(clippy::cast_possible_truncation, reason = "positions within a chunk fit u16")]
pub(crate) fn transpose<I: AsPrimitive<usize>>(
    indices: &[I],
    offset: usize,
    len: usize,
    n_lanes: usize,
) -> VortexResult<TransposedPatches> {
    let n_chunks = len.div_ceil(PATCH_CHUNK_SIZE);
    vortex_ensure!(
        indices.len() <= u32::MAX as usize,
        "Patched arrays support at most u32::MAX patches"
    );
    vortex_ensure!(
        n_lanes.is_power_of_two() && n_lanes <= PATCH_CHUNK_SIZE,
        "patch lanes must be a power of two of at most {PATCH_CHUNK_SIZE}, got {n_lanes}"
    );

    let lane_mask = n_lanes - 1;
    let slot = |index: usize| (index / PATCH_CHUNK_SIZE) * n_lanes + (index & lane_mask);

    // Count the patches in each chunk/lane, checking that positions ascend and are in bounds.
    let mut lane_offsets: BufferMut<u32> = BufferMut::zeroed(n_chunks * n_lanes + 1);
    let mut previous: Option<usize> = None;
    for index in indices {
        let index = index.as_();
        vortex_ensure!(
            index >= offset && index - offset < len,
            "patch index {index} out of bounds for array of length {len} at offset {offset}"
        );
        let index = index - offset;
        vortex_ensure!(
            previous.is_none_or(|p| p < index),
            "patch indices must be strictly ascending"
        );
        previous = Some(index);
        lane_offsets[slot(index) + 1] += 1;
    }

    // With one lane per chunk, position order is already chunk/lane order.
    let sorted = n_lanes == 1;
    let mut out_indices = BufferMut::<u16>::with_capacity(indices.len());
    if sorted {
        out_indices.extend(
            indices
                .iter()
                .map(|i| ((i.as_() - offset) % PATCH_CHUNK_SIZE) as u16),
        );
    }

    for i in 1..lane_offsets.len() {
        lane_offsets[i] += lane_offsets[i - 1];
    }

    if sorted {
        return Ok(TransposedPatches {
            lane_offsets: lane_offsets.freeze(),
            indices: out_indices.freeze(),
            permutation: None,
        });
    }

    // Scatter each patch to its slot. `cursor` tracks the next free position of each lane.
    let mut permutation = BufferMut::<u32>::with_capacity(indices.len());
    let mut identity = true;
    {
        let out_indices = out_indices.spare_capacity_mut();
        let out_permutation = permutation.spare_capacity_mut();
        let mut cursor = lane_offsets[..n_chunks * n_lanes].to_vec();
        for (input, index) in indices.iter().enumerate() {
            let index = index.as_() - offset;
            let position = &mut cursor[slot(index)];
            let out = *position as usize;
            identity &= out == input;
            out_indices[out].write((index % PATCH_CHUNK_SIZE) as u16);
            out_permutation[out].write(input as u32);
            *position += 1;
        }
    }
    // SAFETY: every input patch wrote one distinct output slot, covering all of them.
    unsafe {
        out_indices.set_len(indices.len());
        permutation.set_len(indices.len());
    }

    Ok(TransposedPatches {
        lane_offsets: lane_offsets.freeze(),
        indices: out_indices.freeze(),
        permutation: (!identity).then(|| permutation.freeze()),
    })
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn transpose_orders_by_chunk_then_lane() -> VortexResult<()> {
        let indices = [1u32, 2, 33, 1024 + 1, 1024 + 32];
        let t = transpose(&indices, 0, 2048, 32)?;
        // chunk 0: lane 1 holds positions 1 and 33, lane 2 holds 2. chunk 1: lane 0 holds 1056,
        // lane 1 holds 1025.
        assert_eq!(t.indices.as_slice(), &[1, 33, 2, 32, 1]);
        assert_eq!(t.permutation.as_deref(), Some([0u32, 2, 1, 4, 3].as_slice()));
        assert_eq!(t.lane_offsets[0..4], [0, 0, 2, 3]);
        assert_eq!(t.lane_offsets[32..35], [3, 4, 5]);
        assert_eq!(*t.lane_offsets.last().unwrap(), 5);

        let layout = PatchLayout::try_new(0, 2048, 32, &t.lane_offsets, &t.indices)?;
        let mut seen = Vec::new();
        layout.for_each(|pos, patch| seen.push((pos, patch)));
        seen.sort();
        assert_eq!(seen, vec![(1, 0), (2, 2), (33, 1), (1025, 4), (1056, 3)]);
        Ok(())
    }

    #[test]
    fn transpose_identity_has_no_permutation() -> VortexResult<()> {
        let t = transpose(&[0u8, 1, 2], 0, 10, 32)?;
        assert!(t.permutation.is_none());
        let t = transpose(&[10u64, 2000], 10, 3000, 1)?;
        assert!(t.permutation.is_none());
        assert_eq!(t.indices.as_slice(), &[0, 1990 % 1024]);
        Ok(())
    }

    #[test]
    fn transpose_rejects_unsorted_and_out_of_bounds() {
        assert!(transpose(&[2u32, 1], 0, 10, 1).is_err());
        assert!(transpose(&[10u32], 0, 10, 1).is_err());
        assert!(transpose(&[1u32], 0, 10, 3).is_err());
    }

    #[test]
    fn layout_rejects_malformed_children() {
        let check = |n_lanes, offsets: &[u32], indices: &[u16]| {
            PatchLayout::try_new(0, 10, n_lanes, offsets, indices).and_then(|l| l.validate())
        };
        // Decreasing lane offsets.
        assert!(check(2, &[1, 0, 1], &[0]).is_err());
        // An index outside its chunk.
        assert!(PatchLayout::try_new(0, 10, 1, &[0, 1], &[1024]).is_err());
        // An index in the wrong lane.
        assert!(check(2, &[0, 1, 1], &[1]).is_err());
        // Offsets past the patches.
        assert!(PatchLayout::try_new(0, 10, 1, &[0, 2], &[1]).is_err());
        assert!(check(2, &[0, 1, 2], &[2, 3]).is_ok());
    }

    #[test]
    fn layout_skips_out_of_bounds_patches() -> VortexResult<()> {
        let t = transpose(&[1u32, 5, 1030, 2047], 0, 2048, 1)?;
        // A view of positions 3..1031.
        let layout = PatchLayout::try_new(3, 1028, 1, &t.lane_offsets, &t.indices)?;
        let mut seen = Vec::new();
        layout.for_each(|pos, patch| seen.push((pos, patch)));
        assert_eq!(seen, vec![(2, 1), (1027, 2)]);
        assert!(layout.has_out_of_bounds_patches());
        assert_eq!(layout.count(), 2);
        Ok(())
    }
}
