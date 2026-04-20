// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::arrays::Primitive;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::VTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::RLEData;
use crate::rle::array::INDICES_SLOT;
use crate::rle::array::RLEArrayExt;
use crate::rle::array::VALUES_IDX_OFFSETS_SLOT;
use crate::rle::array::VALUES_SLOT;
use crate::rle::array::rle_decompress::rle_decompress;
use crate::rle::kernel::PARENT_KERNELS;
use crate::rle::vtable::rules::RULES;

mod operations;
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
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.offset.hash(state);
    }
}

impl ArrayEq for RLEData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.offset == other.offset
    }
}

impl VTable for RLE {
    type ArrayData = RLEData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("fastlanes.rle");
        *ID
    }

    fn validate(
        &self,
        data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        validate_parts(
            slots[VALUES_SLOT]
                .as_ref()
                .vortex_expect("RLEArray values slot must be populated"),
            slots[INDICES_SLOT]
                .as_ref()
                .vortex_expect("RLEArray indices slot must be populated"),
            slots[VALUES_IDX_OFFSETS_SLOT]
                .as_ref()
                .vortex_expect("RLEArray values_idx_offsets slot must be populated"),
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

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        crate::rle::array::SLOT_NAMES[idx].to_string()
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            RLEMetadata {
                values_len: array.values().len() as u64,
                indices_len: array.indices().len() as u64,
                indices_ptype: PType::try_from(array.indices().dtype())? as i32,
                values_idx_offsets_len: array.values_idx_offsets().len() as u64,
                values_idx_offsets_ptype: PType::try_from(array.values_idx_offsets().dtype())?
                    as i32,
                offset: array.offset() as u64,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.is_empty(),
            "RLEArray expects 0 buffers, got {}",
            buffers.len()
        );
        let metadata = RLEMetadata::decode(metadata)?;
        let values = children.get(
            0,
            &DType::Primitive(dtype.as_ptype(), Nullability::NonNullable),
            usize::try_from(metadata.values_len)?,
        )?;

        let indices = children.get(
            1,
            &DType::Primitive(metadata.indices_ptype(), dtype.nullability()),
            usize::try_from(metadata.indices_len)?,
        )?;

        let values_idx_offsets = children.get(
            2,
            &DType::Primitive(
                metadata.values_idx_offsets_ptype(),
                Nullability::NonNullable,
            ),
            usize::try_from(metadata.values_idx_offsets_len)?,
        )?;

        let slots = vec![Some(values), Some(indices), Some(values_idx_offsets)];
        let data = RLEData::try_new(metadata.offset as usize)?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
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
        let slots = vec![Some(values), Some(indices), Some(values_idx_offsets)];
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
        let slots = vec![Some(values), Some(indices), Some(values_idx_offsets)];
        let data = unsafe { RLEData::new_unchecked(offset) };
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(RLE, dtype, length, data).with_slots(slots))
        }
    }

    /// Encode a primitive array using FastLanes RLE.
    pub fn encode(array: ArrayView<'_, Primitive>) -> VortexResult<RLEArray> {
        RLEData::encode(array)
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
