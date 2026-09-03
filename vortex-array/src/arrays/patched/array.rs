// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Range;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::Patched;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::layout::ChunkLocal;
use crate::arrays::patched::layout::PatchedView;
use crate::arrays::patched::layout::chunk_local_from_global;
use crate::arrays::patched::layout::n_chunks;
use crate::arrays::patched::layout::validate_layout;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::PType;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::patches::Patches;
use crate::validity::Validity;

#[derive(Debug, Clone)]
pub struct PatchedData {
    /// Grid position of logical row zero, always below [`PATCH_CHUNK_SIZE`].
    ///
    /// Slicing keeps patches on their original 1024-row grid, so a slice that starts mid-chunk
    /// records where its first row sits inside that chunk.
    pub(super) offset: usize,
}

#[array_slots(Patched)]
pub struct PatchedSlots {
    /// The inner array containing the base values, with placeholders at patched rows.
    #[slot(0)]
    pub inner: ArrayRef,
    /// Chunk-local `u16` positions of the patched rows, sorted within each chunk.
    #[slot(1)]
    pub patch_indices: ArrayRef,
    /// The values that overwrite the inner array at the patched rows.
    #[slot(2)]
    pub patch_values: ArrayRef,
    /// `u32` prefix patch counts, one per chunk plus a terminator.
    #[slot(3)]
    pub chunk_offsets: ArrayRef,
}

impl Display for PatchedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "offset: {}", self.offset)
    }
}

impl PatchedData {
    /// Check the shape of the children. The layout invariants over the patch positions are only
    /// checked by [`Patched::try_new`], which sees canonical children.
    pub(crate) fn validate(
        &self,
        dtype: &DType,
        len: usize,
        slots: &PatchedSlotsView,
    ) -> VortexResult<()> {
        vortex_ensure!(
            self.offset < PATCH_CHUNK_SIZE,
            "Patched offset {} must be within the first chunk",
            self.offset
        );
        vortex_ensure!(
            slots.inner.dtype() == dtype,
            "Patched inner dtype {} does not match outer dtype {}",
            slots.inner.dtype(),
            dtype
        );
        vortex_ensure!(
            slots.inner.len() == len,
            "Patched inner len {} does not match outer len {}",
            slots.inner.len(),
            len
        );
        vortex_ensure!(
            slots.patch_values.dtype().eq_ignore_nullability(dtype),
            "Patched values dtype {} does not match outer dtype {}",
            slots.patch_values.dtype(),
            dtype
        );
        vortex_ensure!(
            slots.patch_indices.len() == slots.patch_values.len(),
            "Patched indices len {} does not match values len {}",
            slots.patch_indices.len(),
            slots.patch_values.len()
        );
        vortex_ensure!(
            slots.patch_indices.dtype() == &DType::Primitive(PType::U16, NonNullable),
            "Patched indices must be non-nullable u16, got {}",
            slots.patch_indices.dtype()
        );
        vortex_ensure!(
            slots.chunk_offsets.dtype() == &DType::Primitive(PType::U32, NonNullable),
            "Patched chunk offsets must be non-nullable u32, got {}",
            slots.chunk_offsets.dtype()
        );
        let expected = n_chunks(self.offset, len) + 1;
        vortex_ensure!(
            slots.chunk_offsets.len() == expected,
            "Patched expects {} chunk offsets, got {}",
            expected,
            slots.chunk_offsets.len()
        );
        Ok(())
    }
}

pub trait PatchedArrayExt: PatchedArraySlotsExt {
    /// Grid position of logical row zero.
    #[inline]
    fn offset(&self) -> usize {
        self.offset
    }

    /// Number of 1024-row chunks the array spans, counting from the grid origin.
    #[inline]
    fn n_chunks(&self) -> usize {
        n_chunks(self.offset(), self.as_ref().len())
    }

    /// Slice to `range` without reading the patch children.
    ///
    /// The inner array is sliced exactly and `chunk_offsets` is sliced to the covered chunks,
    /// while `patch_indices` and `patch_values` are shared. Patches of the first and last chunk
    /// that fall outside the range stay in the children and are skipped on read. Returns the bare
    /// inner slice when no patch survives.
    fn slice_range(&self, range: Range<usize>) -> VortexResult<ArrayRef> {
        let inner = self.inner().slice(range.clone())?;
        if range.is_empty() {
            return Ok(inner);
        }

        let grid_start = self.offset() + range.start;
        let grid_end = self.offset() + range.end;
        let chunk_start = grid_start / PATCH_CHUNK_SIZE;
        let chunk_end = grid_end.div_ceil(PATCH_CHUNK_SIZE);
        let new_offset = grid_start % PATCH_CHUNK_SIZE;
        let chunk_offsets = self.chunk_offsets().slice(chunk_start..chunk_end + 1)?;

        if let (Some(indices), Some(offsets)) = (
            self.patch_indices().as_opt::<Primitive>(),
            chunk_offsets.as_opt::<Primitive>(),
        ) {
            let view = PatchedView::new(
                new_offset,
                range.len(),
                indices.as_slice::<u16>(),
                offsets.as_slice::<u32>(),
            );
            if view.live().is_empty() {
                return Ok(inner);
            }
        }

        let slots = PatchedSlots {
            inner,
            patch_indices: self.patch_indices().clone(),
            patch_values: self.patch_values().clone(),
            chunk_offsets,
        }
        .into_slots();
        Ok(unsafe {
            Patched::new_unchecked(
                self.as_ref().dtype().clone(),
                range.len(),
                slots,
                new_offset,
            )
        }
        .into_array())
    }

    /// Drop dead patches and rebase `chunk_offsets` so the array carries only its own patches.
    ///
    /// Only canonical position children can be compacted; other arrays are returned unchanged.
    /// Returns the bare inner array when no live patch remains.
    fn compact(&self) -> VortexResult<ArrayRef> {
        let (Some(indices), Some(offsets)) = (
            self.patch_indices().as_opt::<Primitive>(),
            self.chunk_offsets().as_opt::<Primitive>(),
        ) else {
            return Ok(self.as_ref().clone());
        };
        let offsets = offsets.as_slice::<u32>();
        let view = PatchedView::new(
            self.offset(),
            self.as_ref().len(),
            indices.as_slice::<u16>(),
            offsets,
        );
        let live = view.live();
        if live.is_empty() {
            return Ok(self.inner().clone());
        }
        if live == (0..view.indices().len()) {
            return Ok(self.as_ref().clone());
        }

        let rebased = offsets
            .iter()
            .map(|&ordinal| {
                u32::try_from((ordinal as usize).clamp(live.start, live.end) - live.start)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let slots = PatchedSlots {
            inner: self.inner().clone(),
            patch_indices: self.patch_indices().slice(live.clone())?,
            patch_values: self.patch_values().slice(live)?,
            chunk_offsets: PrimitiveArray::new(Buffer::from(rebased), Validity::NonNullable)
                .into_array(),
        }
        .into_slots();
        Ok(unsafe {
            Patched::new_unchecked(
                self.as_ref().dtype().clone(),
                self.as_ref().len(),
                slots,
                self.offset(),
            )
        }
        .into_array())
    }
}

impl<T: TypedArrayRef<Patched>> PatchedArrayExt for T {}

impl Patched {
    /// Wrap `inner` with patches addressed by chunk-local indices.
    ///
    /// When the position children are canonical, every layout invariant is checked: sorted
    /// in-chunk indices, non-decreasing chunk offsets with a terminator, and an `offset` inside
    /// the first chunk.
    pub fn try_new(
        inner: ArrayRef,
        patch_indices: ArrayRef,
        patch_values: ArrayRef,
        chunk_offsets: ArrayRef,
        offset: usize,
    ) -> VortexResult<Array<Patched>> {
        if let (Some(indices), Some(offsets)) = (
            patch_indices.as_opt::<Primitive>(),
            chunk_offsets.as_opt::<Primitive>(),
        ) && indices.dtype() == &DType::Primitive(PType::U16, NonNullable)
            && offsets.dtype() == &DType::Primitive(PType::U32, NonNullable)
        {
            validate_layout(
                offset,
                inner.len(),
                indices.as_slice::<u16>(),
                offsets.as_slice::<u32>(),
            )?;
        }

        let dtype = inner.dtype().clone();
        let len = inner.len();
        let slots = PatchedSlots {
            inner,
            patch_indices,
            patch_values,
            chunk_offsets,
        }
        .into_slots();
        Array::try_from_parts(
            ArrayParts::new(Patched, dtype, len, PatchedData { offset }).with_slots(slots),
        )
    }

    /// Wrap `inner` with a global-index [`Patches`] set, converting it to chunk-local form.
    ///
    /// The patches keep their grid alignment: logical row zero lands at
    /// `patches.offset() % PATCH_CHUNK_SIZE`, so the patch chunks line up with the chunks of an
    /// inner encoding that was sliced by the same offset.
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
            inner.len() == patches.array_len(),
            "Patches cover {} rows but the inner array has {}",
            patches.array_len(),
            inner.len()
        );
        vortex_ensure!(
            patches.num_patches() <= u32::MAX as usize,
            "PatchedArray does not support > u32::MAX patch values"
        );
        vortex_ensure!(
            patches.values().all_valid(ctx)?,
            "PatchedArray cannot be built from Patches with nulls"
        );

        let global = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let ChunkLocal {
            offset,
            indices,
            chunk_offsets,
        } = match_each_unsigned_integer_ptype!(global.ptype(), |I| {
            chunk_local_from_global(
                global.as_slice::<I>(),
                patches.offset(),
                patches.array_len(),
            )?
        });

        // Patch values take the outer nullability but never carry nulls.
        let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        let validity = if inner.dtype().is_nullable() {
            Validity::AllValid
        } else {
            Validity::NonNullable
        };
        let values = PrimitiveArray::from_buffer_handle(
            values.buffer_handle().clone(),
            values.ptype(),
            validity,
        );

        let dtype = inner.dtype().clone();
        let len = inner.len();
        let slots = PatchedSlots {
            inner,
            patch_indices: PrimitiveArray::new(Buffer::from(indices), Validity::NonNullable)
                .into_array(),
            patch_values: values.into_array(),
            chunk_offsets: PrimitiveArray::new(Buffer::from(chunk_offsets), Validity::NonNullable)
                .into_array(),
        }
        .into_slots();
        Ok(unsafe { Self::new_unchecked(dtype, len, slots, offset) })
    }

    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        len: usize,
        slots: ArraySlots,
        offset: usize,
    ) -> Array<Patched> {
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Patched, dtype, len, PatchedData { offset }).with_slots(slots),
            )
        }
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(view.required.len(), 3);
        assert_eq!(view.maybe.expect("optional slot").len(), 3);

        let cloned = OptionalPatchedSlots::from_slots(slot_vec.into());
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
        assert_eq!(rebuilt.inner.len(), required.len());
        assert_eq!(rebuilt.patch_values.len(), required.len());
        assert_eq!(rebuilt.chunk_offsets.len(), optional.len());
    }

    #[test]
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

        let slot_vec = vec![Some(offsets.clone()), None, Some(chunk0), Some(chunk1)];

        let view = VariadicSlotsView::from_slots(&slot_vec);
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

        let owned = VariadicSlots::from_slots(slot_vec.into());
        assert_eq!(owned.offsets.len(), offsets.len());
        assert!(owned.maybe_validity.is_none());
        assert_eq!(owned.chunks.len(), 2);

        let slots = owned.into_slots();
        assert_eq!(slots.len(), 4);
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
        assert_eq!(view.offsets.len(), offsets.len());
        assert_eq!(view.maybe_validity.map(|v| v.len()), Some(validity.len()));
        assert_eq!(view.chunks.len(), 1);
        assert_eq!(view.chunks[0].len(), chunk.len());

        // `into_slots` must emit annotation order, not the shuffled declaration order.
        let round_tripped = ShuffledVariadicSlots::from_slots(slot_vec.into()).into_slots();
        assert_eq!(round_tripped.len(), 3);
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
