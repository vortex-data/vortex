// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use super::indices::local_position;
use super::indices::visit_chunks;
use crate::Array;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::patches::Patches;
use crate::search_sorted::decode_sorted_positions;
use crate::validity::Validity;

/// Sorted patches over a base array of the same logical type.
#[derive(Clone, Debug)]
pub struct Patched;

/// An array using the sorted patch encoding.
pub type PatchedArray = Array<Patched>;

/// Strategy families can grow independently, e.g. numeric addition or integer bit composition.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum PatchStrategy {
    /// Operations on integer values.
    Integer(IntegerPatchStrategy),
    /// Operations on floating-point values.
    Float(FloatPatchStrategy),
}

/// Operations available for integer patches.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, prost::Enumeration)]
#[repr(i32)]
#[non_exhaustive]
pub enum IntegerPatchStrategy {
    /// Replace both value and validity.
    Replace = 0,
}

/// Operations available for floating-point patches.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, prost::Enumeration)]
#[repr(i32)]
#[non_exhaustive]
pub enum FloatPatchStrategy {
    /// Replace both value and validity.
    Replace = 0,
}

impl fmt::Display for PatchStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer(IntegerPatchStrategy::Replace)
            | Self::Float(FloatPatchStrategy::Replace) => f.write_str("replace"),
        }
    }
}

impl PatchStrategy {
    pub(super) fn validate(self, dtype: &DType) -> VortexResult<()> {
        let supported = match self {
            Self::Integer(IntegerPatchStrategy::Replace) => dtype.is_int(),
            Self::Float(FloatPatchStrategy::Replace) => dtype.is_float(),
        };
        vortex_ensure!(supported, "Patch strategy {self} does not support {dtype}");
        Ok(())
    }

    fn replace(dtype: &DType) -> VortexResult<Self> {
        if dtype.is_int() {
            Ok(Self::Integer(IntegerPatchStrategy::Replace))
        } else if dtype.is_float() {
            Ok(Self::Float(FloatPatchStrategy::Replace))
        } else {
            vortex_bail!("Unsupported patched dtype {dtype}")
        }
    }
}

/// Scalar metadata accompanying the four child arrays.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PatchedData {
    pub(super) offset: usize,
    pub(super) strategy: PatchStrategy,
}

impl fmt::Display for PatchedData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, offset={}", self.strategy, self.offset)
    }
}

impl ArrayHash for PatchedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _mode: EqMode) {
        self.hash(state);
    }
}

impl ArrayEq for PatchedData {
    fn array_eq(&self, other: &Self, _mode: EqMode) -> bool {
        self == other
    }
}

#[array_slots(Patched)]
pub struct PatchedSlots {
    /// Base array, also the spine of a fused decode.
    #[slot(0)]
    pub inner: ArrayRef,
    /// Chunk-relative u16 positions, strictly increasing within each 1,024-row chunk.
    #[slot(1)]
    pub patch_indices: ArrayRef,
    /// Replacement values and their validity.
    #[slot(2)]
    pub patch_values: ArrayRef,
    /// Starting patch position for each physical patch chunk.
    #[slot(3)]
    pub patch_chunk_offsets: ArrayRef,
}

/// Access the patch description on an owned or borrowed array.
pub trait PatchedArrayExt: TypedArrayRef<Patched> {
    /// Reconstruct absolute positions for kernels that consume shared `Patches`.
    fn absolute_patches(&self, ctx: &mut ExecutionCtx) -> VortexResult<Patches> {
        let slots = self.as_ref().as_::<Patched>();
        let mut rows = Vec::with_capacity(slots.patch_indices().len());
        visit_chunks::<false>(
            slots.patch_indices(),
            slots.patch_chunk_offsets(),
            self.offset,
            self.len(),
            ctx,
            |base, _, local| rows.extend(local.iter().map(|&i| base as u64 + u64::from(i))),
        )?;
        // SAFETY: Patched construction validates sorted positions, value count, and chunk starts.
        // Chunk starts are normalized to this view, including a partial first chunk.
        Ok(unsafe {
            Patches::new_unchecked(
                self.len(),
                self.offset,
                Buffer::from(rows).into_array(),
                slots.patch_values().clone(),
                Some(slots.patch_chunk_offsets().clone()),
                Some(0),
            )
        })
    }

    /// How patch values combine with the base.
    fn strategy(&self) -> PatchStrategy {
        self.strategy
    }
}

impl<T: TypedArrayRef<Patched>> PatchedArrayExt for T {}

impl Patched {
    /// Replace the selected base rows, including their validity.
    pub fn from_array_and_patches(
        inner: ArrayRef,
        patches: &Patches,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PatchedArray> {
        let strategy = PatchStrategy::replace(inner.dtype())?;
        Self::with_strategy(inner, patches, strategy, ctx)
    }

    /// Construct a patched array with an explicit dtype-specific strategy.
    pub fn with_strategy(
        inner: ArrayRef,
        patches: &Patches,
        strategy: PatchStrategy,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PatchedArray> {
        let dtype = inner.dtype().clone();
        strategy.validate(&dtype)?;
        vortex_ensure_eq!(inner.len(), patches.array_len());
        vortex_ensure!(dtype.eq_with_nullability_superset(patches.dtype()));
        let rows = decode_sorted_positions(patches.indices(), ctx)?;
        let offset = patches.offset();
        let end = offset
            .checked_add(inner.len())
            .ok_or_else(|| vortex_err!("Patch range overflow"))?;
        vortex_ensure!(rows.is_sorted_by(|a, b| a < b));
        vortex_ensure!(
            rows.first().is_some_and(|&r| r >= offset as u64)
                && rows.last().is_some_and(|&r| r < end as u64)
        );
        let mut position = 0;
        let offsets = (offset / PATCH_CHUNK_SIZE..end.div_ceil(PATCH_CHUNK_SIZE))
            .map(|chunk| {
                let start = position;
                while position < rows.len()
                    && rows[position] / PATCH_CHUNK_SIZE as u64 == chunk as u64
                {
                    position += 1;
                }
                start as u64
            })
            .collect::<Buffer<_>>();
        if let Some(supplied) = patches.chunk_offsets() {
            vortex_ensure!(supplied.dtype().is_unsigned_int() && !supplied.dtype().is_nullable());
            let supplied = decode_sorted_positions(supplied, ctx)?;
            vortex_ensure_eq!(supplied.len(), offsets.len());
            let skipped = patches.offset_within_chunk().unwrap_or(0) as u64;
            for (&actual, &relative) in supplied.iter().zip(offsets.iter()).skip(1) {
                vortex_ensure!(
                    actual
                        .checked_sub(supplied[0])
                        .and_then(|n| n.checked_sub(skipped))
                        == Some(relative),
                    "Chunk offset disagrees with patch indices"
                );
            }
        }
        let indices = rows
            .iter()
            .map(|&r| local_position(r))
            .collect::<Buffer<_>>();
        let offsets = PrimitiveArray::new(offsets, Validity::NonNullable)
            .narrow(ctx)?
            .into_array();
        Self::from_relative(
            inner,
            indices.into_array(),
            patches.values().cast(dtype)?,
            offsets,
            offset,
            strategy,
        )
    }

    pub(super) fn from_relative(
        inner: ArrayRef,
        indices: ArrayRef,
        values: ArrayRef,
        offsets: ArrayRef,
        offset: usize,
        strategy: PatchStrategy,
    ) -> VortexResult<PatchedArray> {
        let dtype = inner.dtype().clone();
        let len = inner.len();
        let data = PatchedData { offset, strategy };
        let slots = PatchedSlots {
            inner,
            patch_indices: indices,
            patch_values: values,
            patch_chunk_offsets: offsets,
        };
        Array::try_from_parts(ArrayParts::new(Self, dtype, len, data, slots.into_slots()))
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
        assert!(ArrayRef::ptr_eq(&rebuilt.patch_indices, &optional));
        assert!(ArrayRef::ptr_eq(&rebuilt.patch_values, &required));
        assert!(ArrayRef::ptr_eq(&rebuilt.patch_chunk_offsets, &optional));
        assert_eq!(rebuilt.inner.len(), required.len());
        assert_eq!(rebuilt.patch_values.len(), required.len());
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
