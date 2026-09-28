// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::registry::CachedId;

use crate::RLEData;
use crate::rle::array::RLESlots;
use crate::rle::array::RLESlotsView;
use crate::rle::array::rle_decompress::rle_decompress;
use crate::rle::vtable::rules::RULES;

mod operations;
mod plugin;
mod rules;
mod validity;

/// A [`RLE`]-encoded Vortex array.
pub type RLEArray = Array<RLE>;

#[derive(Clone, prost::Message)]
pub struct RLEMetadata {
    #[prost(uint64, tag = "1")]
    pub values_len: u64,
    #[prost(uint64, tag = "2")]
    pub indices_len: u64,
    #[prost(enumeration = "PType", tag = "3")]
    pub indices_ptype: i32,
    #[prost(uint64, tag = "4")]
    pub values_idx_offsets_len: u64,
    #[prost(enumeration = "PType", tag = "5")]
    pub values_idx_offsets_ptype: i32,
    #[prost(uint64, tag = "6", default = "0")]
    pub offset: u64,
}

impl ArrayHash for RLEData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.offset.hash(state);
    }
}

impl ArrayEq for RLEData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.offset == other.offset
    }
}

impl VTable for RLE {
    type TypedArrayData = RLEData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("fastlanes.rle");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let rle_slots = RLESlotsView::from_slots(slots);
        validate_parts(
            rle_slots.values,
            rle_slots.indices,
            rle_slots.values_idx_offsets,
            data.offset,
            dtype,
            len,
        )
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("RLEArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        RLESlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            rle_decompress(&array, ctx)?.into_array(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct RLE;

impl RLE {
    pub fn try_new(
        values: ArrayRef,
        indices: ArrayRef,
        values_idx_offsets: ArrayRef,
        offset: usize,
        length: usize,
    ) -> VortexResult<RLEArray> {
        let dtype = DType::Primitive(values.dtype().as_ptype(), indices.dtype().nullability());
        let slots = RLESlots {
            values,
            indices,
            values_idx_offsets,
        }
        .into_slots();
        let data = RLEData::try_new(offset)?;
        Array::try_from_parts(ArrayParts::new(RLE, dtype, length, data).with_slots(slots))
    }

    /// Create a new RLE array without validation.
    ///
    /// # Safety
    /// See [`RLE::validate`] for preconditions.
    pub unsafe fn new_unchecked(
        values: ArrayRef,
        indices: ArrayRef,
        values_idx_offsets: ArrayRef,
        offset: usize,
        length: usize,
    ) -> RLEArray {
        let dtype = DType::Primitive(values.dtype().as_ptype(), indices.dtype().nullability());
        let slots = RLESlots {
            values,
            indices,
            values_idx_offsets,
        }
        .into_slots();
        let data = unsafe { RLEData::new_unchecked(offset) };
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(RLE, dtype, length, data).with_slots(slots))
        }
    }

    /// Encode a primitive array using FastLanes RLE.
    pub fn encode(
        array: ArrayView<'_, Primitive>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RLEArray> {
        RLEData::encode(array, ctx)
    }
}

fn validate_parts(
    values: &ArrayRef,
    indices: &ArrayRef,
    values_idx_offsets: &ArrayRef,
    offset: usize,
    dtype: &DType,
    length: usize,
) -> VortexResult<()> {
    vortex_ensure!(
        matches!(
            values.dtype(),
            DType::Primitive(_, Nullability::NonNullable)
        ),
        "RLE values must be a non-nullable primitive type, got {}",
        values.dtype()
    );

    vortex_ensure!(
        matches!(indices.dtype().as_ptype(), PType::U8 | PType::U16),
        "RLE indices must be u8 or u16, got {}",
        indices.dtype()
    );

    vortex_ensure!(
        values_idx_offsets.dtype().is_unsigned_int() && !values_idx_offsets.dtype().is_nullable(),
        "RLE value idx offsets must be non-nullable unsigned integer, got {}",
        values_idx_offsets.dtype()
    );

    vortex_ensure!(
        indices.len().is_multiple_of(crate::FL_CHUNK_SIZE),
        "RLE indices length must be a multiple of {}, got {}",
        crate::FL_CHUNK_SIZE,
        indices.len()
    );

    vortex_ensure!(
        offset + length <= indices.len(),
        "RLE offset + length, {offset} + {length}, must not exceed the indices length {}",
        indices.len()
    );

    vortex_ensure!(
        indices.len().div_ceil(crate::FL_CHUNK_SIZE) == values_idx_offsets.len(),
        "RLE must have one value idx offset per chunk, got {}",
        values_idx_offsets.len()
    );

    vortex_ensure!(
        indices.len() >= values.len(),
        "RLE must have at least as many indices as values, got {} indices and {} values",
        indices.len(),
        values.len()
    );

    let expected_dtype = DType::Primitive(values.dtype().as_ptype(), indices.dtype().nullability());
    vortex_ensure!(
        dtype == &expected_dtype,
        "RLE dtype mismatch: expected {expected_dtype}, got {dtype}"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::test_harness::check_metadata;

    use super::RLEMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_rle_metadata() {
        check_metadata(
            "rle.metadata",
            &RLEMetadata {
                values_len: u64::MAX,
                indices_len: u64::MAX,
                indices_ptype: i32::MAX,
                values_idx_offsets_len: u64::MAX,
                values_idx_offsets_ptype: i32::MAX,
                offset: u64::MAX,
            }
            .encode_to_vec(),
        );
    }
}
