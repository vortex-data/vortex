// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use num_traits::AsPrimitive;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
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
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;
use crate::validity::Validity;

#[derive(Debug, Clone)]
pub struct PatchedData {
    /// The absolute offset of the first in-view element, accounting for any slicing.
    ///
    /// Patch indices are stored as global positions, so the final position of a patch within the
    /// executed array is `index - offset`.
    pub(super) offset: usize,

    /// Number of patches sliced off the start of the first in-view chunk.
    ///
    /// `chunk_offsets` are sliced at chunk granularity while the patches themselves are sliced at
    /// element granularity, so this records how many leading patches of the first chunk fall
    /// outside the view.
    pub(super) offset_within_chunk: usize,
}

#[array_slots(Patched)]
pub struct PatchedSlots {
    /// The inner array containing the base unpatched values.
    pub inner: ArrayRef,
    /// The sorted global indices of patched (exception) values.
    pub patch_indices: ArrayRef,
    /// The patched (exception) values at the corresponding indices.
    pub patch_values: ArrayRef,
    /// One offset per 1024-element chunk into `patch_indices`/`patch_values`.
    pub chunk_offsets: ArrayRef,
}

impl Display for PatchedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "offset: {}, offset_within_chunk: {}",
            self.offset, self.offset_within_chunk
        )
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
        vortex_ensure!(
            slots.patch_indices.dtype().is_unsigned_int(),
            "PatchedArray patch indices must be unsigned integers, got {}",
            slots.patch_indices.dtype()
        );
        Ok(())
    }
}

pub trait PatchedArrayExt: PatchedArraySlotsExt {
    /// The absolute offset of the first in-view element.
    #[inline]
    fn offset(&self) -> usize {
        self.offset
    }

    /// Number of patches sliced off the start of the first in-view chunk.
    #[inline]
    fn offset_within_chunk(&self) -> usize {
        self.offset_within_chunk
    }

    /// Reconstruct the untransposed [`Patches`] backing this array.
    fn patches(&self) -> Patches {
        // SAFETY: a `Patched` array is only ever constructed from valid, sorted patches with
        // matching index/value lengths and chunk offsets.
        unsafe {
            Patches::new_unchecked(
                self.as_ref().len(),
                self.offset(),
                self.patch_indices().clone(),
                self.patch_values().clone(),
                Some(self.chunk_offsets().clone()),
                Some(self.offset_within_chunk()),
            )
        }
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
            patches.values().all_valid(ctx)?,
            "PatchedArray cannot be built from Patches with nulls"
        );

        // Ensure the patches carry a chunk offset for every 1024-element chunk, computing them
        // when the source patches don't already provide them.
        let patches = match patches.chunk_offsets() {
            Some(_) => patches.clone(),
            None => {
                let chunk_offsets = compute_chunk_offsets(patches, ctx)?;
                // SAFETY: we only attach freshly computed chunk offsets to existing valid patches.
                unsafe {
                    Patches::new_unchecked(
                        patches.array_len(),
                        patches.offset(),
                        patches.indices().clone(),
                        patches.values().clone(),
                        Some(chunk_offsets),
                        Some(0),
                    )
                }
            }
        };

        Ok(Self::wrap(inner, &patches))
    }

    /// Wrap an `inner` array and untransposed `patches` (which must carry chunk offsets) into a
    /// [`Patched`] array.
    pub(super) fn wrap(inner: ArrayRef, patches: &Patches) -> Array<Patched> {
        let chunk_offsets = patches
            .chunk_offsets()
            .clone()
            .vortex_expect("Patched requires chunk offsets");
        let dtype = inner.dtype().clone();
        let len = inner.len();
        let slots = PatchedSlots {
            inner,
            patch_indices: patches.indices().clone(),
            patch_values: patches.values().clone(),
            chunk_offsets,
        }
        .into_slots();
        unsafe {
            Self::new_unchecked(
                dtype,
                len,
                slots,
                patches.offset(),
                patches.offset_within_chunk().unwrap_or(0),
            )
        }
    }

    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        len: usize,
        slots: ArraySlots,
        offset: usize,
        offset_within_chunk: usize,
    ) -> Array<Patched> {
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(
                    Patched,
                    dtype,
                    len,
                    PatchedData {
                        offset,
                        offset_within_chunk,
                    },
                )
                .with_slots(slots),
            )
        }
    }
}

/// Compute one `u64` chunk offset per 1024-element chunk for a set of sorted patches.
///
/// `chunk_offsets[c]` is the position in the patch arrays at which the patches for chunk `c`
/// begin, i.e. the number of patches whose global index is less than `c * 1024`.
fn compute_chunk_offsets(patches: &Patches, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let array_len = patches.array_len();
    let offset = patches.offset();
    let total_chunks = array_len.div_ceil(1024);

    let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let indices_ptype = indices.ptype();

    let chunk_offsets = match_each_unsigned_integer_ptype!(indices_ptype, |I| {
        let indices = indices.as_slice::<I>();
        let mut offsets: BufferMut<u64> = BufferMut::with_capacity(total_chunks);
        let mut pos = 0usize;
        for chunk in 0..total_chunks {
            let chunk_start = chunk * 1024;
            while pos < indices.len() && {
                let index: usize = indices[pos].as_();
                index - offset < chunk_start
            } {
                pos += 1;
            }
            offsets.push(pos as u64);
        }
        offsets.freeze()
    });

    Ok(PrimitiveArray::new(chunk_offsets, Validity::NonNullable).into_array())
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
        assert_eq!(rebuilt.patch_values.len(), optional.len());
    }
}
