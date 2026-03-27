// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::Canonical;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::TransposedPatches;
use crate::arrays::patched::patch_lanes;
use crate::buffer::BufferHandle;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;
use crate::stats::ArrayStats;
use crate::validity::Validity;

/// An array that partially "patches" another array with new values.
#[derive(Debug, Clone)]
pub struct PatchedArray {
    /// The inner array that is being patched. This is the zeroth child.
    pub(super) inner: ArrayRef,

    /// Number of 1024-element chunks. Pre-computed for convenience.
    pub(super) n_chunks: usize,

    /// Number of lanes the patch indices and values have been split into. Each of the `n_chunks`
    /// of 1024 values is split into `n_lanes` lanes horizontally, each lane having 1024 / n_lanes
    /// values that might be patched.
    pub(super) n_lanes: usize,

    /// Offset into the first chunk
    pub(super) offset: usize,
    /// Total length.
    pub(super) len: usize,

    /// lane offsets. The PType of these MUST be u32
    pub(super) lane_offsets: BufferHandle,
    /// indices within a 1024-element chunk. The PType of these MUST be u16
    pub(super) indices: ArrayRef,
    /// patch values corresponding to the indices. The ptype is specified by `values_ptype`.
    pub(super) values: ArrayRef,

    pub(super) stats_set: ArrayStats,
}

impl PatchedArray {
    /// Create a new `PatchedArray` from a child array and a set of [`Patches`].
    ///
    /// # Errors
    ///
    /// The `inner` array must be primitive type, and it must have the same DType as the patches.
    pub fn from_array_and_patches(
        inner: ArrayRef,
        patches: &Patches,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            inner.dtype().eq_with_nullability_superset(patches.dtype()),
            "array DType must match patches DType"
        );

        vortex_ensure!(
            inner.dtype().is_primitive(),
            "Creating PatchedArray from Patches only supported for primitive arrays"
        );

        let values_ptype = patches.dtype().as_ptype();

        let TransposedPatches {
            n_chunks,
            n_lanes,
            lane_offsets,
            indices,
            values,
        } = transpose_patches(patches, ctx)?;

        let indices = PrimitiveArray::from_buffer_handle(
            BufferHandle::new_host(indices),
            PType::U16,
            Validity::NonNullable,
        )
        .into_array();
        let values = PrimitiveArray::from_buffer_handle(
            BufferHandle::new_host(values),
            values_ptype,
            Validity::NonNullable,
        )
        .into_array();

        let len = inner.len();

        Ok(Self {
            inner,
            n_chunks,
            n_lanes,
            offset: 0,
            len,
            lane_offsets: BufferHandle::new_host(lane_offsets),
            indices,
            values,
            stats_set: ArrayStats::default(),
        })
    }
}

impl PatchedArray {
    /// Get a range of indices that can be used to access the `indices` and `values` children
    /// to retrieve all patches for a specified lane.
    ///
    /// # Panics
    ///
    /// Note that this function will panic if the caller requests out of bounds chunk/lane ordinals.
    pub(crate) fn lane_range(&self, chunk: usize, lane: usize) -> Range<usize> {
        assert!(chunk < self.n_chunks);
        assert!(lane < self.n_lanes);

        let lane_offsets = self.lane_offsets.as_host().reinterpret::<u32>();

        let start = lane_offsets[chunk * self.n_lanes + lane] as usize;
        let stop = lane_offsets[chunk * self.n_lanes + lane + 1] as usize;

        start..stop
    }

    /// Slice the array to just the patches and inner values that are within the chunk range.
    pub(crate) fn slice_chunks(&self, chunks: Range<usize>) -> VortexResult<Self> {
        let lane_offsets_start = chunks.start * self.n_lanes;
        let lane_offsets_stop = chunks.end * self.n_lanes + 1;

        let sliced_lane_offsets = self
            .lane_offsets
            .slice_typed::<u32>(lane_offsets_start..lane_offsets_stop);
        let indices = self.indices.clone();
        let values = self.values.clone();

        let begin = (chunks.start * 1024).max(self.offset);
        let end = (chunks.end * 1024).min(self.len);

        let offset = begin % 1024;

        let inner = self.inner.slice(begin..end)?;

        let len = end - begin;
        let n_chunks = (end - begin).div_ceil(1024);

        Ok(PatchedArray {
            inner,
            n_chunks,
            n_lanes: self.n_lanes,
            offset,
            len,
            indices,
            values,
            lane_offsets: sliced_lane_offsets,
            stats_set: ArrayStats::default(),
        })
    }
}

/// Transpose a set of patches from the default sorted layout into the data parallel layout.
#[allow(clippy::cognitive_complexity)]
fn transpose_patches(patches: &Patches, ctx: &mut ExecutionCtx) -> VortexResult<TransposedPatches> {
    let array_len = patches.array_len();
    let offset = patches.offset();

    let indices = patches
        .indices()
        .clone()
        .execute::<Canonical>(ctx)?
        .into_primitive();

    let values = patches
        .values()
        .clone()
        .execute::<Canonical>(ctx)?
        .into_primitive();

    let indices_ptype = indices.ptype();
    let values_ptype = values.ptype();

    let indices = indices.buffer_handle().clone().unwrap_host();
    let values = values.buffer_handle().clone().unwrap_host();

    match_each_unsigned_integer_ptype!(indices_ptype, |I| {
        match_each_native_ptype!(values_ptype, |V| {
            let indices: Buffer<I> = Buffer::from_byte_buffer(indices);
            let values: Buffer<V> = Buffer::from_byte_buffer(values);

            Ok(transpose(
                indices.as_slice(),
                values.as_slice(),
                offset,
                array_len,
            ))
        })
    })
}

#[allow(clippy::cast_possible_truncation)]
fn transpose<I: IntegerPType, V: NativePType>(
    indices_in: &[I],
    values_in: &[V],
    offset: usize,
    array_len: usize,
) -> TransposedPatches {
    // Total number of slots is number of chunks times number of lanes.
    let n_chunks = array_len.div_ceil(1024);
    assert!(
        n_chunks <= u32::MAX as usize,
        "Cannot transpose patches for array with >= 4 trillion elements"
    );

    let n_lanes = patch_lanes::<V>();

    // We know upfront how many indices and values we'll have.
    let mut indices_buffer = BufferMut::with_capacity(indices_in.len());
    let mut values_buffer = BufferMut::with_capacity(values_in.len());

    // number of patches in each chunk.
    let mut lane_offsets: BufferMut<u32> = BufferMut::zeroed(n_chunks * n_lanes + 1);

    // Scan the index/values once to get chunk/lane counts
    for index in indices_in {
        let index = index.as_() - offset;
        let chunk = index / 1024;
        let lane = index % n_lanes;

        lane_offsets[chunk * n_lanes + lane + 1] += 1;
    }

    // Prefix-sum sizes -> offsets
    for index in 1..lane_offsets.len() {
        lane_offsets[index] += lane_offsets[index - 1];
    }

    // Loop over patches, writing them to final positions
    let indices_out = indices_buffer.spare_capacity_mut();
    let values_out = values_buffer.spare_capacity_mut();
    for (index, &value) in std::iter::zip(indices_in, values_in) {
        let index = index.as_() - offset;
        let chunk = index / 1024;
        let lane = index % n_lanes;

        let position = &mut lane_offsets[chunk * n_lanes + lane];
        indices_out[*position as usize].write((index % 1024) as u16);
        values_out[*position as usize].write(value);
        *position += 1;
    }

    // SAFETY: we know there are exactly indices_in.len() indices/values, and we just
    //  set them to the appropriate values in the loop above.
    unsafe {
        indices_buffer.set_len(indices_in.len());
        values_buffer.set_len(values_in.len());
    }

    // Now, pass over all the indices and values again and subtract out the position increments.
    for index in indices_in {
        let index = index.as_() - offset;
        let chunk = index / 1024;
        let lane = index % n_lanes;

        lane_offsets[chunk * n_lanes + lane] -= 1;
    }

    TransposedPatches {
        n_chunks,
        n_lanes,
        lane_offsets: lane_offsets.freeze().into_byte_buffer(),
        indices: indices_buffer.freeze().into_byte_buffer(),
        values: values_buffer.freeze().into_byte_buffer(),
    }
}
