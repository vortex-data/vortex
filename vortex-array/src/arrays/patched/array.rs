// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Range;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::ArraySlots;
use crate::CanonicalValidity;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::ConstantArray;
use crate::arrays::Masked;
use crate::arrays::Patched;
use crate::arrays::PrimitiveArray;
use crate::arrays::masked::MaskedArrayExt;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::arrays::patched::PATCH_CHUNK_SIZE;
use crate::arrays::patched::PatchLayout;
use crate::arrays::patched::layout::TransposedPatches;
use crate::arrays::patched::layout::transpose;
use crate::arrays::patched::patch_lanes_for;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_integer_ptype;
use crate::patches::Patches;
use crate::scalar::Scalar;
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
    #[slot(0)]
    pub inner: ArrayRef,
    /// The lane offsets array for locating patches within lanes.
    #[slot(1)]
    pub lane_offsets: ArrayRef,
    /// The indices of patched (exception) values.
    #[slot(2)]
    pub patch_indices: ArrayRef,
    /// The patched (exception) values at the corresponding indices.
    #[slot(3)]
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
        vortex_ensure_eq!(
            slots.inner.dtype(),
            dtype,
            "PatchedArray inner dtype does not match outer dtype",
        );
        vortex_ensure_eq!(
            slots.patch_values.dtype(),
            dtype,
            "PatchedArray patch values dtype does not match outer dtype",
        );
        vortex_ensure_eq!(
            slots.inner.len(),
            len,
            "PatchedArray inner len does not match outer len",
        );
        vortex_ensure_eq!(
            slots.patch_indices.len(),
            slots.patch_values.len(),
            "PatchedArray patch indices len does not match patch values len",
        );
        vortex_ensure_eq!(
            slots.patch_indices.dtype(),
            &DType::Primitive(PType::U16, Nullability::NonNullable),
            "PatchedArray patch indices must be non-nullable u16",
        );
        vortex_ensure_eq!(
            slots.lane_offsets.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable),
            "PatchedArray lane offsets must be non-nullable u32",
        );
        vortex_ensure!(
            self.n_lanes.is_power_of_two() && self.n_lanes <= PATCH_CHUNK_SIZE,
            "PatchedArray lanes must be a power of two of at most {PATCH_CHUNK_SIZE}, got {}",
            self.n_lanes
        );
        vortex_ensure!(
            self.offset < PATCH_CHUNK_SIZE,
            "PatchedArray offset {} must be less than {PATCH_CHUNK_SIZE}",
            self.offset
        );
        let n_chunks = (self.offset + len).div_ceil(PATCH_CHUNK_SIZE);
        vortex_ensure_eq!(
            slots.lane_offsets.len(),
            n_chunks * self.n_lanes + 1,
            "PatchedArray lane offsets must have one entry per chunk lane, plus one",
        );
        Ok(())
    }
}

pub trait PatchedArrayExt: TypedArrayRef<Patched> {
    #[inline]
    fn n_lanes(&self) -> usize {
        self.n_lanes
    }

    #[inline]
    fn offset(&self) -> usize {
        self.offset
    }

    /// The number of chunks spanned by the array, including partial first and last chunks.
    #[inline]
    fn n_chunks(&self) -> usize {
        (self.offset() + self.len()).div_ceil(PATCH_CHUNK_SIZE)
    }

    #[inline]
    fn lane_range(
        &self,
        chunk: usize,
        lane: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Range<usize>> {
        assert!(chunk * PATCH_CHUNK_SIZE <= self.len() + self.offset());
        assert!(lane < self.n_lanes());

        let start = self
            .lane_offsets()
            .execute_scalar(chunk * self.n_lanes() + lane, ctx)?;
        let stop = self
            .lane_offsets()
            .execute_scalar(chunk * self.n_lanes() + lane + 1, ctx)?;

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

    /// Execute the lane offsets and patch indices, returning them for use with [`PatchLayout`].
    fn execute_layout(&self, ctx: &mut ExecutionCtx) -> VortexResult<ExecutedLayout> {
        Ok(ExecutedLayout {
            offset: self.offset(),
            len: self.len(),
            n_lanes: self.n_lanes(),
            lane_offsets: self.lane_offsets().clone().execute::<PrimitiveArray>(ctx)?,
            indices: self
                .patch_indices()
                .clone()
                .execute::<PrimitiveArray>(ctx)?,
        })
    }
}

impl<T: TypedArrayRef<Patched>> PatchedArrayExt for T {}

/// The executed lane offsets and patch indices of a patched array.
pub struct ExecutedLayout {
    offset: usize,
    len: usize,
    n_lanes: usize,
    lane_offsets: PrimitiveArray,
    indices: PrimitiveArray,
}

impl ExecutedLayout {
    /// A borrowed [`PatchLayout`] over the executed buffers.
    pub fn layout(&self) -> VortexResult<PatchLayout<'_>> {
        PatchLayout::try_new(
            self.offset,
            self.len,
            self.n_lanes,
            self.lane_offsets.as_slice::<u32>(),
            self.indices.as_slice::<u16>(),
        )
    }
}

impl Patched {
    /// Creates a patched array that replaces the values of `inner` at the ascending positions
    /// `indices` with the corresponding `values`.
    ///
    /// The patches are laid out with [`patch_lanes_for`] lanes.
    ///
    /// `values` may have any nullability; it is cast to the nullability of `inner`, which must be
    /// nullable if `values` contains nulls.
    pub fn try_new(
        inner: ArrayRef,
        indices: ArrayRef,
        values: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Array<Patched>> {
        let n_lanes = patch_lanes_for(inner.dtype(), inner.len(), indices.len());
        Self::try_new_with_lanes(inner, indices, values, n_lanes, ctx)
    }

    /// Creates a patched array like [`Patched::try_new`], laid out with `n_lanes` lanes.
    pub fn try_new_with_lanes(
        inner: ArrayRef,
        indices: ArrayRef,
        values: ArrayRef,
        n_lanes: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Array<Patched>> {
        Self::try_new_offset(inner, indices, values, 0, n_lanes, ctx)
    }

    /// Creates a patched array whose `indices` are all offset by `indices_offset`.
    fn try_new_offset(
        inner: ArrayRef,
        indices: ArrayRef,
        values: ArrayRef,
        indices_offset: usize,
        n_lanes: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Array<Patched>> {
        vortex_ensure!(
            inner.dtype().eq_ignore_nullability(values.dtype()),
            "patch values dtype {} does not match array dtype {}",
            values.dtype(),
            inner.dtype()
        );
        vortex_ensure_eq!(
            indices.len(),
            values.len(),
            "patch indices and values must have the same length"
        );
        vortex_ensure!(
            indices.dtype().is_int(),
            "patch indices must be integers, found {}",
            indices.dtype()
        );

        let nullability = inner.dtype().nullability() | values.dtype().nullability();
        let dtype = inner.dtype().with_nullability(nullability);
        let len = inner.len();
        let inner = if inner.dtype() == &dtype {
            inner
        } else {
            inner.cast(dtype.clone())?
        };
        let values = if values.dtype() == &dtype {
            values
        } else {
            values.cast(dtype.clone())?
        };

        let indices = indices.execute::<PrimitiveArray>(ctx)?;
        vortex_ensure!(
            indices.all_valid(ctx)?,
            "patch indices must not contain nulls"
        );
        let TransposedPatches {
            lane_offsets,
            indices,
            permutation,
        } = match_each_integer_ptype!(indices.ptype(), |I| {
            transpose(indices.as_slice::<I>(), indices_offset, len, n_lanes)?
        });

        let values = match permutation {
            None => values,
            Some(permutation) => {
                let permuted = values
                    .take(PrimitiveArray::new(permutation, Validity::NonNullable).into_array())?;
                // Reorder canonical values now, rather than on every execution. Compressed values
                // stay lazy so that the reordering does not decompress them.
                if values.is::<AnyCanonical>() {
                    // Also materialize the validity, which would otherwise be reordered lazily.
                    permuted.execute::<CanonicalValidity>(ctx)?.0.into_array()
                } else {
                    permuted
                }
            }
        };

        let slots = PatchedSlots {
            inner,
            lane_offsets: PrimitiveArray::new(lane_offsets, Validity::NonNullable).into_array(),
            patch_indices: PrimitiveArray::new(indices, Validity::NonNullable).into_array(),
            patch_values: values,
        }
        .into_slots();
        // SAFETY: `transpose` validated the indices and produced a layout covering every chunk,
        // and the dtypes of the children were checked above.
        Ok(unsafe { Self::new_unchecked(dtype, len, slots, n_lanes, 0) })
    }

    /// Creates a patched array from an array and its [`Patches`].
    pub fn from_array_and_patches(
        inner: ArrayRef,
        patches: &Patches,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Array<Patched>> {
        let n_lanes = patch_lanes_for(patches.dtype(), patches.array_len(), patches.num_patches());
        Self::from_array_and_patches_with_lanes(inner, patches, n_lanes, ctx)
    }

    /// Creates a patched array from an array and its [`Patches`], laid out with `n_lanes` lanes.
    ///
    /// With a single lane the patches keep their order, so the patch values are used as they
    /// are, without being reordered.
    pub fn from_array_and_patches_with_lanes(
        inner: ArrayRef,
        patches: &Patches,
        n_lanes: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Array<Patched>> {
        vortex_ensure_eq!(
            inner.len(),
            patches.array_len(),
            "array length must match patches length"
        );
        Self::try_new_offset(
            inner,
            patches.indices().clone(),
            patches.values().clone(),
            patches.offset(),
            n_lanes,
            ctx,
        )
    }

    /// Creates an array of `len` copies of `fill_value`, except at the ascending positions
    /// `indices` which take the corresponding `values`.
    ///
    /// This is the patched equivalent of a sparse array.
    pub fn sparse(
        indices: ArrayRef,
        values: ArrayRef,
        len: usize,
        fill_value: Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Array<Patched>> {
        let n_lanes = patch_lanes_for(fill_value.dtype(), len, indices.len());
        Self::sparse_with_lanes(indices, values, len, fill_value, n_lanes, ctx)
    }

    /// Creates a sparse patched array like [`Patched::sparse`], laid out with `n_lanes` lanes.
    pub fn sparse_with_lanes(
        indices: ArrayRef,
        values: ArrayRef,
        len: usize,
        fill_value: Scalar,
        n_lanes: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Array<Patched>> {
        let nullable = fill_value.dtype().is_nullable() || values.dtype().is_nullable();
        let fill_value = fill_value.cast(&fill_value.dtype().with_nullability(nullable.into()))?;
        let inner = ConstantArray::new(fill_value, len).into_array();
        Self::try_new_with_lanes(inner, indices, values, n_lanes, ctx)
    }

    /// Creates a nullable array with the values of `inner` and nulls at the ascending positions
    /// `null_positions`.
    ///
    /// This is the patched equivalent of a masked array whose nulls are sparse: the patch values
    /// are a single null constant, so only the positions of the nulls are stored.
    pub fn with_nulls(
        inner: ArrayRef,
        null_positions: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Array<Patched>> {
        let dtype = inner.dtype().as_nullable();
        let values = ConstantArray::new(Scalar::null(dtype), null_positions.len()).into_array();
        Self::try_new(inner, null_positions, values, ctx)
    }

    /// Converts a [`Masked`] array into the equivalent patched array of nulls over its child.
    pub fn from_masked(
        masked: ArrayView<'_, Masked>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Array<Patched>> {
        let mask = masked.masked_validity().execute_mask(masked.len(), ctx)?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "patch positions fit in u32"
        )]
        let null_positions: Buffer<u32> = match mask {
            Mask::AllTrue(_) => Buffer::empty(),
            Mask::AllFalse(len) => (0..len as u32).collect(),
            Mask::Values(values) => values
                .bit_buffer()
                .iter()
                .enumerate()
                .filter(|(_, valid)| !valid)
                .map(|(pos, _)| pos as u32)
                .collect(),
        };
        Self::with_nulls(
            masked.child().clone(),
            PrimitiveArray::new(null_positions, Validity::NonNullable).into_array(),
            ctx,
        )
    }

    /// Creates a patched array from its parts without validation.
    ///
    /// # Safety
    ///
    /// The slots must satisfy the invariants checked by [`PatchedData::validate`], and the patch
    /// indices of every lane must be in bounds of their chunk.
    pub unsafe fn new_unchecked(
        dtype: DType,
        len: usize,
        slots: ArraySlots,
        n_lanes: usize,
        offset: usize,
    ) -> Array<Patched> {
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(
                Patched,
                dtype,
                len,
                PatchedData { n_lanes, offset },
                slots,
            ))
        }
    }

    /// Creates a patched array that shares the lane offsets and patch indices of `array`, with
    /// new `inner` and `values` children.
    ///
    /// # Safety
    ///
    /// `inner` must have the length of `array`, `values` the length of its patch values, and both
    /// must have the dtype `dtype`.
    pub(crate) unsafe fn with_inner_and_values<A: TypedArrayRef<Patched>>(
        array: &A,
        dtype: DType,
        inner: ArrayRef,
        values: ArrayRef,
    ) -> Array<Patched> {
        let slots = PatchedSlots {
            inner,
            lane_offsets: array.lane_offsets().clone(),
            patch_indices: array.patch_indices().clone(),
            patch_values: values,
        }
        .into_slots();
        unsafe { Self::new_unchecked(dtype, array.len(), slots, array.n_lanes(), array.offset()) }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use super::PatchedSlots;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::array_slots;
    use crate::arrays::Chunked;
    use crate::arrays::Null;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::Union;
    use crate::validity::Validity;

    #[array_slots(Null)]
    struct OptionalPatchedSlots {
        #[slot(0)]
        required: ArrayRef,
        #[slot(1)]
        maybe: Option<ArrayRef>,
    }

    #[array_slots(Chunked)]
    struct VariadicSlots {
        #[slot(0)]
        offsets: ArrayRef,
        #[slot(1)]
        maybe_validity: Option<ArrayRef>,
        #[slot(2..)]
        chunks: Vec<ArrayRef>,
    }

    /// The same layout as [`VariadicSlots`], but with every field declaration moved. The
    /// `#[slot(..)]` annotations must keep the storage layout identical.
    #[array_slots(Union)]
    struct ShuffledVariadicSlots {
        #[slot(2..)]
        chunks: Vec<ArrayRef>,
        #[slot(1)]
        maybe_validity: Option<ArrayRef>,
        #[slot(0)]
        offsets: ArrayRef,
    }

    #[test]
    fn generated_slots_round_trip() {
        let required = PrimitiveArray::new(buffer![1u8, 2, 3], Validity::NonNullable).into_array();
        let optional = PrimitiveArray::new(buffer![4u8, 5, 6], Validity::NonNullable).into_array();

        let slot_vec = vec![Some(required.clone()), Some(optional.clone())];
        let view = OptionalPatchedSlotsView::from_slots(&slot_vec);
        assert!(ArrayRef::ptr_eq(view.required, &required));
        assert!(ArrayRef::ptr_eq(
            view.maybe.expect("optional slot"),
            &optional
        ));
        assert_eq!(view.required.len(), 3);
        assert_eq!(view.maybe.expect("optional slot").len(), 3);

        let cloned = OptionalPatchedSlots::from_slots(slot_vec.into());
        assert!(ArrayRef::ptr_eq(&cloned.required, &required));
        assert!(ArrayRef::ptr_eq(
            cloned.maybe.as_ref().expect("optional clone"),
            &optional
        ));
        assert_eq!(cloned.required.len(), required.len());
        assert_eq!(cloned.maybe.expect("optional clone").len(), optional.len());

        let rebuilt = PatchedSlots::from_slots(
            vec![
                Some(required.clone()),
                Some(optional.clone()),
                Some(required.clone()),
                Some(optional.clone()),
            ]
            .into(),
        );
        assert!(ArrayRef::ptr_eq(&rebuilt.inner, &required));
        assert!(ArrayRef::ptr_eq(&rebuilt.lane_offsets, &optional));
        assert!(ArrayRef::ptr_eq(&rebuilt.patch_indices, &required));
        assert!(ArrayRef::ptr_eq(&rebuilt.patch_values, &optional));
        assert_eq!(rebuilt.inner.len(), required.len());
        assert_eq!(rebuilt.patch_values.len(), optional.len());
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn variadic_slots_round_trip() {
        let offsets = PrimitiveArray::new(buffer![0u64, 3, 5], Validity::NonNullable).into_array();
        let chunk0 = PrimitiveArray::new(buffer![1u8, 2, 3], Validity::NonNullable).into_array();
        let chunk1 = PrimitiveArray::new(buffer![4u8, 5], Validity::NonNullable).into_array();

        assert_eq!(VariadicSlots::OFFSETS, 0);
        assert_eq!(VariadicSlots::MAYBE_VALIDITY, 1);
        assert_eq!(VariadicSlots::CHUNKS_OFFSET, 2);
        assert_eq!(VariadicSlots::FIXED_COUNT, 2);
        assert_eq!(VariadicSlots::slot_name(0), "offsets");
        assert_eq!(VariadicSlots::slot_name(3), "chunks[1]");

        let slot_vec = vec![
            Some(offsets.clone()),
            None,
            Some(chunk0.clone()),
            Some(chunk1.clone()),
        ];

        let view = VariadicSlotsView::from_slots(&slot_vec);
        assert!(ArrayRef::ptr_eq(view.offsets, &offsets));
        assert!(ArrayRef::ptr_eq(&view.chunks[0], &chunk0));
        assert!(ArrayRef::ptr_eq(
            view.chunks.get(1).expect("second chunk"),
            &chunk1
        ));
        assert_eq!(view.offsets.len(), 3);
        assert!(view.maybe_validity.is_none());
        assert_eq!(view.chunks.len(), 2);
        assert_eq!(view.chunks[0].len(), 3);
        assert_eq!(view.chunks.get(1).map(|c| c.len()), Some(2));
        assert!(view.chunks.get(2).is_none());
        assert_eq!(
            view.chunks.iter().map(|c| c.len()).collect::<Vec<_>>(),
            vec![3, 2]
        );

        let owned = view.to_owned();
        assert_eq!(owned.chunks.len(), 2);
        assert!(ArrayRef::ptr_eq(&owned.offsets, &offsets));
        assert!(owned.maybe_validity.is_none());
        assert!(ArrayRef::ptr_eq(&owned.chunks[0], &chunk0));
        assert!(ArrayRef::ptr_eq(&owned.chunks[1], &chunk1));

        let owned = VariadicSlots::from_slots(slot_vec.into());
        assert_eq!(owned.offsets.len(), offsets.len());
        assert!(owned.maybe_validity.is_none());
        assert_eq!(owned.chunks.len(), 2);

        assert!(ArrayRef::ptr_eq(&owned.offsets, &offsets));
        assert!(ArrayRef::ptr_eq(&owned.chunks[0], &chunk0));
        assert!(ArrayRef::ptr_eq(&owned.chunks[1], &chunk1));
        let slots = owned.into_slots();
        assert_eq!(slots.len(), 4);
        for (index, expected) in [(0, &offsets), (2, &chunk0), (3, &chunk1)] {
            assert!(ArrayRef::ptr_eq(
                slots[index].as_ref().expect("present slot"),
                expected
            ));
        }
        assert!(slots[1].is_none());
        assert_eq!(
            slots[VariadicSlots::CHUNKS_OFFSET]
                .as_ref()
                .map(|c| c.len()),
            Some(3)
        );
    }

    #[test]
    fn variadic_slots_empty_tail() {
        let offsets = PrimitiveArray::new(buffer![0u64], Validity::NonNullable).into_array();
        let slot_vec = vec![Some(offsets), None];

        let view = VariadicSlotsView::from_slots(&slot_vec);
        assert!(view.chunks.is_empty());

        let owned = VariadicSlots::from_slots(slot_vec.into());
        assert!(owned.chunks.is_empty());
        assert_eq!(owned.into_slots().len(), 2);
    }

    #[test]
    fn variadic_slots_spilled_tail_keeps_order() {
        let offsets = PrimitiveArray::new(buffer![0u64], Validity::NonNullable).into_array();
        let chunks: Vec<ArrayRef> = (0..5u8)
            .map(|len| {
                PrimitiveArray::new((0..len).collect::<Buffer<u8>>(), Validity::NonNullable)
                    .into_array()
            })
            .collect();

        let slots = VariadicSlots {
            offsets: offsets.clone(),
            maybe_validity: None,
            chunks,
        }
        .into_slots();

        assert!(slots.spilled());
        assert_eq!(slots.len(), VariadicSlots::FIXED_COUNT + 5);
        assert_eq!(
            slots[VariadicSlots::OFFSETS].as_ref().map(|s| s.len()),
            Some(offsets.len())
        );
        assert!(slots[VariadicSlots::MAYBE_VALIDITY].is_none());
        assert_eq!(
            slots[VariadicSlots::CHUNKS_OFFSET..]
                .iter()
                .map(|s| s.as_ref().map(|c| c.len()))
                .collect::<Vec<_>>(),
            (0..5).map(Some).collect::<Vec<_>>()
        );
    }

    #[test]
    fn slot_indices_follow_annotations_not_declaration_order() {
        assert_eq!(
            ShuffledVariadicSlots::OFFSETS,
            VariadicSlots::OFFSETS,
            "field declaration order must not move a slot"
        );
        assert_eq!(
            ShuffledVariadicSlots::MAYBE_VALIDITY,
            VariadicSlots::MAYBE_VALIDITY
        );
        assert_eq!(
            ShuffledVariadicSlots::CHUNKS_OFFSET,
            VariadicSlots::CHUNKS_OFFSET
        );
        assert_eq!(
            ShuffledVariadicSlots::FIXED_COUNT,
            VariadicSlots::FIXED_COUNT
        );
        assert_eq!(ShuffledVariadicSlots::slot_name(0), "offsets");
        assert_eq!(ShuffledVariadicSlots::slot_name(1), "maybe_validity");
        assert_eq!(ShuffledVariadicSlots::slot_name(3), "chunks[1]");
    }

    #[test]
    fn shuffled_declaration_order_round_trips_through_storage() {
        let offsets = PrimitiveArray::new(buffer![0u64, 3], Validity::NonNullable).into_array();
        let validity = PrimitiveArray::new(buffer![1u8, 1], Validity::NonNullable).into_array();
        let chunk = PrimitiveArray::new(buffer![1u8, 2, 3], Validity::NonNullable).into_array();

        let slot_vec = vec![
            Some(offsets.clone()),
            Some(validity.clone()),
            Some(chunk.clone()),
        ];

        let view = ShuffledVariadicSlotsView::from_slots(&slot_vec);
        assert!(ArrayRef::ptr_eq(view.offsets, &offsets));
        assert!(ArrayRef::ptr_eq(
            view.maybe_validity.expect("validity"),
            &validity
        ));
        assert!(ArrayRef::ptr_eq(&view.chunks[0], &chunk));
        assert_eq!(view.offsets.len(), offsets.len());
        assert_eq!(view.maybe_validity.map(|v| v.len()), Some(validity.len()));
        assert_eq!(view.chunks.len(), 1);
        assert_eq!(view.chunks[0].len(), chunk.len());

        // `into_slots` must emit annotation order, not the shuffled declaration order.
        let round_tripped = ShuffledVariadicSlots::from_slots(slot_vec.into()).into_slots();
        assert_eq!(round_tripped.len(), 3);
        for (index, expected) in [&offsets, &validity, &chunk].into_iter().enumerate() {
            assert!(ArrayRef::ptr_eq(
                round_tripped[index].as_ref().expect("present slot"),
                expected
            ));
        }
        assert_eq!(
            round_tripped[ShuffledVariadicSlots::OFFSETS]
                .as_ref()
                .map(|s| s.len()),
            Some(offsets.len())
        );
        assert_eq!(
            round_tripped[ShuffledVariadicSlots::MAYBE_VALIDITY]
                .as_ref()
                .map(|s| s.len()),
            Some(validity.len())
        );
        assert_eq!(
            round_tripped[ShuffledVariadicSlots::CHUNKS_OFFSET]
                .as_ref()
                .map(|s| s.len()),
            Some(chunk.len())
        );
    }
}
