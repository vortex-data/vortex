// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Range;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::Patched;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::TransposedPatches;
use crate::arrays::patched::patch_lanes;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;
use crate::validity::Validity;

#[derive(Debug, Clone)]
pub struct PatchedData {
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
}

#[array_slots(Patched)]
pub struct PatchedSlots {
    /// The inner array containing the base unpatched values.
    pub inner: ArrayRef,
    /// The lane offsets array for locating patches within lanes.
    pub lane_offsets: ArrayRef,
    /// The indices of patched (exception) values.
    pub patch_indices: ArrayRef,
    /// The patched (exception) values at the corresponding indices.
    pub patch_values: ArrayRef,
}

impl Display for PatchedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "n_lanes: {}, offset: {}", self.n_lanes, self.offset)
    }
}

impl PatchedData {
    pub(crate) fn validate(
        &self,
        dtype: &DType,
        len: usize,
        slots: &PatchedSlotsView,
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.inner.dtype() == dtype,
            "PatchedArray base dtype {} does not match outer dtype {}",
            slots.inner.dtype(),
            dtype
        );
        vortex_ensure!(
            slots.inner.len() == len,
            "PatchedArray base len {} does not match outer len {}",
            slots.inner.len(),
            len
        );
        vortex_ensure!(
            slots.patch_indices.len() == slots.patch_values.len(),
            "PatchedArray patch indices len {} does not match patch values len {}",
            slots.patch_indices.len(),
            slots.patch_values.len()
        );
        Ok(())
    }
}

pub trait PatchedArrayExt: PatchedArraySlotsExt {
    #[inline]
    fn n_lanes(&self) -> usize {
        self.n_lanes
    }

    #[inline]
    fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    fn lane_range(&self, chunk: usize, lane: usize) -> VortexResult<Range<usize>> {
        assert!(chunk * 1024 <= self.as_ref().len() + self.offset());
        assert!(lane < self.n_lanes());

        let start = self
            .lane_offsets()
            .scalar_at(chunk * self.n_lanes() + lane)?;
        let stop = self
            .lane_offsets()
            .scalar_at(chunk * self.n_lanes() + lane + 1)?;

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

    fn slice_chunks(&self, chunks: Range<usize>) -> VortexResult<Array<Patched>> {
        let lane_offsets_start = chunks.start * self.n_lanes();
        let lane_offsets_stop = chunks.end * self.n_lanes() + 1;

        let sliced_lane_offsets = self
            .lane_offsets()
            .slice(lane_offsets_start..lane_offsets_stop)?;
        let indices = self.patch_indices().clone();
        let values = self.patch_values().clone();

        let begin = (chunks.start * 1024).saturating_sub(self.offset());
        let end = (chunks.end * 1024)
            .saturating_sub(self.offset())
            .min(self.as_ref().len());

        let offset = if chunks.start == 0 { self.offset() } else { 0 };
        let inner = self.inner().slice(begin..end)?;
        let len = inner.len();
        let dtype = self.as_ref().dtype().clone();
        let slots = PatchedSlots {
            inner,
            lane_offsets: sliced_lane_offsets,
            patch_indices: indices,
            patch_values: values,
        }
        .into_slots();

        Ok(unsafe { Patched::new_unchecked(dtype, len, slots, self.n_lanes(), offset) })
    }
}

impl<T: TypedArrayRef<Patched>> PatchedArrayExt for T {}

impl Patched {
    pub fn from_array_and_patches(
        inner: ArrayRef,
        patches: &Patches,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Array<Patched>> {
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

        let dtype = inner.dtype().clone();
        let len = inner.len();
        let slots = PatchedSlots {
            inner,
            lane_offsets,
            patch_indices: indices,
            patch_values: values,
        }
        .into_slots();
        Ok(unsafe { Self::new_unchecked(dtype, len, slots, n_lanes, 0) })
    }

    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        len: usize,
        slots: Vec<Option<ArrayRef>>,
        n_lanes: usize,
        offset: usize,
    ) -> Array<Patched> {
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Patched, dtype, len, PatchedData { n_lanes, offset })
                    .with_slots(slots),
            )
        }
    }
}

/// Transpose a set of patches from the default sorted layout into the data parallel layout.
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

#[expect(clippy::cast_possible_truncation)]
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

    // Number of patches in each chunk/lane.
    let mut lane_offsets: BufferMut<u32> = BufferMut::zeroed(n_chunks * n_lanes + 1);

    // Scan the index/value pairs once to get chunk/lane counts.
    for index in indices_in {
        let index = index.as_() - offset;
        let chunk = index / 1024;
        let lane = index % n_lanes;

        lane_offsets[chunk * n_lanes + lane + 1] += 1;
    }

    for index in 1..lane_offsets.len() {
        lane_offsets[index] += lane_offsets[index - 1];
    }

    // Loop over patches, writing them to final positions.
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

    unsafe {
        indices_buffer.set_len(indices_in.len());
        values_buffer.set_len(values_in.len());
    }

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

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::PatchedSlots;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::array_slots;
    use crate::arrays::Null;
    use crate::arrays::PrimitiveArray;
    use crate::validity::Validity;

    #[array_slots(Null)]
    struct OptionalPatchedSlots {
        required: ArrayRef,
        maybe: Option<ArrayRef>,
    }

    #[test]
    fn generated_slots_round_trip() {
        let required = PrimitiveArray::new(buffer![1u8, 2, 3], Validity::NonNullable).into_array();
        let optional = PrimitiveArray::new(buffer![4u8, 5, 6], Validity::NonNullable).into_array();

        let slot_vec = vec![Some(required.clone()), Some(optional.clone())];
        let view = OptionalPatchedSlotsView::from_slots(&slot_vec);
        assert_eq!(view.required.len(), 3);
        assert_eq!(view.maybe.expect("optional slot").len(), 3);

        let cloned = OptionalPatchedSlots::from_slots(slot_vec);
        assert_eq!(cloned.required.len(), required.len());
        assert_eq!(cloned.maybe.expect("optional clone").len(), optional.len());

        let rebuilt = PatchedSlots::from_slots(vec![
            Some(required.clone()),
            Some(optional.clone()),
            Some(required.clone()),
            Some(optional.clone()),
        ]);
        assert_eq!(rebuilt.inner.len(), required.len());
        assert_eq!(rebuilt.patch_values.len(), optional.len());
    }
}
