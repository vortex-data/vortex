// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::Canonical;
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
///
/// # Background
///
/// This is meant to be the foundation of a fully data-parallel patching strategy, based on the
/// work published in ["G-ALP" from Hepkema et al.](https://ir.cwi.nl/pub/35205/35205.pdf)
///
/// Patching is common when an encoding almost completely covers an array save a few exceptions.
/// In that case, rather than avoid the encoding entirely, it's preferable to
///
/// * Replace unencodable values with fillers (zeros, frequent values, nulls, etc.)
/// * Wrap the array with a `PatchedArray` signaling that when the original array is executed,
///   some of the decoded values must be overwritten.
///
/// In Vortex, the FastLanes bit-packing encoding is often the terminal node in an encoding tree,
/// and FastLanes has an intrinsic chunking of 1024 elements. Thus, 1024 elements is pervasively
/// a useful unit of chunking throughout Vortex, and so we use 1024 as a chunk size here
/// as well.
///
/// # Details
///
/// To patch an array, we first divide it into a set of chunks of length 1024, and then within
/// each chunk, we assign each position to a lane. The number of lanes depends on the width of
/// the underlying type.
///
/// Thus, rather than sorting patch indices and values by their global offset, they are sorted
/// primarily by their chunk, and then subsequently by their lanes.
///
/// The Patched array layout has 4 children
///
/// * `inner`: the inner array is the one containing encoded values, including the filler values
///   that need to be patched over at execution time
/// * `lane_offsets`: this is an indexing buffer that allows you to see into ranges of the other
///   two children
/// * `indices`: An array of `u16` chunk indices, indicating where within the chunk should the value
///   be overwritten by the patch value
/// * `values`: The child array containing the patch values, which should be inserted over
///   the values of the `inner` at the locations provided by `indices`
///
/// `indices` and `values` are aligned and accessed together.
///
/// ```text
///
///                  chunk 0      chunk 0      chunk 0     chunk 0       chunk 0     chunk 0
///                  lane  0      lane 1       lane  2     lane 3        lane  4     lane  5
///              ┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
/// lane_offsets │     0      │     0      │     2      │     2      │     3      │     5      │  ...
///              └─────┬──────┴─────┬──────┴─────┬──────┴──────┬─────┴──────┬─────┴──────┬─────┘
///                    │            │            │             │            │            │
///                    │            │            │             │            │            │
///              ┌─────┴────────────┘            └──────┬──────┘     ┌──────┘            └─────┐
///              │                                      │            │                         │
///              │                                      │            │                         │
///              │                                      │            │                         │
///              ▼────────────┬────────────┬────────────▼────────────▼────────────┬────────────▼
///    indices   │            │            │            │            │            │            │
///              │            │            │            │            │            │            │
///              ├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤
///    values    │            │            │            │            │            │            │
///              │            │            │            │            │            │            │
///              └────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
/// ```
///
/// It turns out that this layout is optimal for executing patching on GPUs, because the
/// `lane_offsets` allows each thread in a warp to seek to its patches in constant time.
/// The inner array containing the base unpatched values.
pub(super) const INNER_SLOT: usize = 0;
/// The lane offsets array for locating patches within lanes.
pub(super) const LANE_OFFSETS_SLOT: usize = 1;
/// The indices of patched (exception) values.
pub(super) const INDICES_SLOT: usize = 2;
/// The patched (exception) values at the corresponding indices.
pub(super) const VALUES_SLOT: usize = 3;
pub(super) const NUM_SLOTS: usize = 4;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] =
    ["inner", "lane_offsets", "patch_indices", "patch_values"];

#[derive(Debug, Clone)]
pub struct PatchedArray {
    /// Child arrays stored as slots:
    /// 0: inner - the inner array being patched
    /// 1: lane_offsets - u32 array for indexing into indices/values
    /// 2: indices - u16 array of chunk indices
    /// 3: values - array of patch values
    pub(super) slots: Vec<Option<ArrayRef>>,

    /// Number of lanes the patch indices and values have been split into. Each of the `n_chunks`
    /// of 1024 values is split into `n_lanes` lanes horizontally, each lane having 1024 / n_lanes
    /// values that might be patched.
    pub(super) n_lanes: usize,

    /// The offset into that first chunk that is considered in bounds.
    ///
    /// The patch indices of the first chunk less than `offset` should be skipped, and the offset
    /// should be subtracted out of the remaining offsets to get their final position in the
    /// executed array.
    pub(super) offset: usize,
    /// Length of the array
    pub(super) len: usize,

    pub(super) stats_set: ArrayStats,
}

impl PatchedArray {
    /// Create a new `PatchedArray` from a child array and a set of [`Patches`].
    ///
    /// # Errors
    ///
    /// The `inner` array must be primitive type, and it must have the same `DType` as the patches.
    ///
    /// The patches cannot contain nulls themselves. Any nulls must be stored in the `inner` array's
    /// validity.
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

        vortex_ensure!(
            patches.num_patches() <= u32::MAX as usize,
            "PatchedArray does not support > u32::MAX patch values"
        );

        vortex_ensure!(
            patches.values().all_valid()?,
            "PatchedArray cannot be built from Patches with nulls"
        );

        let values_ptype = patches.dtype().as_ptype();

        let TransposedPatches {
            n_lanes,
            lane_offsets,
            indices,
            values,
        } = transpose_patches(patches, ctx)?;

        let lane_offsets = PrimitiveArray::from_buffer_handle(
            BufferHandle::new_host(lane_offsets),
            PType::U32,
            Validity::NonNullable,
        )
        .into_array();
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
            slots: vec![Some(inner), Some(lane_offsets), Some(indices), Some(values)],
            n_lanes,
            offset: 0,
            len,
            stats_set: ArrayStats::default(),
        })
    }
}

impl PatchedArray {
    /// Returns a reference to the base array being patched.
    #[inline]
    pub fn base_array(&self) -> &ArrayRef {
        self.slots[INNER_SLOT]
            .as_ref()
            .vortex_expect("PatchedArray inner slot")
    }

    /// Returns a reference to the lane offsets array (u32).
    #[inline]
    pub fn lane_offsets(&self) -> &ArrayRef {
        self.slots[LANE_OFFSETS_SLOT]
            .as_ref()
            .vortex_expect("PatchedArray lane_offsets slot")
    }

    /// Returns a reference to the indices array (u16).
    #[inline]
    pub fn patch_indices(&self) -> &ArrayRef {
        self.slots[INDICES_SLOT]
            .as_ref()
            .vortex_expect("PatchedArray indices slot")
    }

    /// Returns a reference to the patch values array.
    #[inline]
    pub fn patch_values(&self) -> &ArrayRef {
        self.slots[VALUES_SLOT]
            .as_ref()
            .vortex_expect("PatchedArray values slot")
    }
}

impl PatchedArray {
    /// Get a range of indices that can be used to access the `indices` and `values` children
    /// to retrieve all patches for a specified lane.
    ///
    /// # Panics
    ///
    /// Note that this function will panic if the caller requests out of bounds chunk/lane ordinals.
    pub(crate) fn lane_range(&self, chunk: usize, lane: usize) -> VortexResult<Range<usize>> {
        assert!(chunk * 1024 <= self.len + self.offset);
        assert!(lane < self.n_lanes);

        let start = self.lane_offsets().scalar_at(chunk * self.n_lanes + lane)?;
        let stop = self
            .lane_offsets()
            .scalar_at(chunk * self.n_lanes + lane + 1)?;

        let start = start
            .as_primitive()
            .as_::<usize>()
            .ok_or_else(|| vortex_err!("could not cast lane_offset to usize"))?;

        let stop = stop
            .as_primitive()
            .as_::<usize>()
            .ok_or_else(|| vortex_err!("could not cast lane_offset to usize"))?;

        Ok(start..stop)
    }

    /// Slice the array to just the patches and inner values that are within the chunk range.
    pub(crate) fn slice_chunks(&self, chunks: Range<usize>) -> VortexResult<Self> {
        let lane_offsets_start = chunks.start * self.n_lanes;
        let lane_offsets_stop = chunks.end * self.n_lanes + 1;

        let sliced_lane_offsets = self
            .lane_offsets()
            .slice(lane_offsets_start..lane_offsets_stop)?;
        let indices = self.patch_indices().clone();
        let values = self.patch_values().clone();

        // Find the new start/end for slicing the inner array.
        // The inner array has already been sliced to start at position `offset` in absolute terms,
        // so we need to convert chunk boundaries to inner-relative coordinates.
        let begin = (chunks.start * 1024).saturating_sub(self.offset);
        let end = (chunks.end * 1024)
            .saturating_sub(self.offset)
            .min(self.len);

        let offset = if chunks.start == 0 { self.offset } else { 0 };

        let inner = self.base_array().slice(begin..end)?;

        let len = end - begin;

        Ok(PatchedArray {
            slots: vec![
                Some(inner),
                Some(sliced_lane_offsets),
                Some(indices),
                Some(values),
            ],
            n_lanes: self.n_lanes,
            offset,
            len,
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
        n_lanes,
        lane_offsets: lane_offsets.freeze().into_byte_buffer(),
        indices: indices_buffer.freeze().into_byte_buffer(),
        values: values_buffer.freeze().into_byte_buffer(),
    }
}
