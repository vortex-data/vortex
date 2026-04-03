// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

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
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::RLEData;
use crate::rle::array::rle_decompress::rle_decompress;
use crate::rle::kernel::PARENT_KERNELS;
use crate::rle::vtable::rules::RULES;

mod operations;
mod rules;
mod validity;

vtable!(RLE, RLE, RLEData);

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

impl VTable for RLE {
    type ArrayData = RLEData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(&self, data: &Self::ArrayData, dtype: &DType, len: usize) -> VortexResult<()> {
        data.validate_against_outer(dtype, len)
    }

    fn array_hash<H: std::hash::Hasher>(array: &RLEData, state: &mut H, precision: Precision) {
        array.values().array_hash(state, precision);
        array.indices().array_hash(state, precision);
        array.values_idx_offsets().array_hash(state, precision);
        array.offset().hash(state);
    }

    fn array_eq(array: &RLEData, other: &RLEData, precision: Precision) -> bool {
        array.values().array_eq(other.values(), precision)
            && array.indices().array_eq(other.indices(), precision)
            && array
                .values_idx_offsets()
                .array_eq(other.values_idx_offsets(), precision)
            && array.offset() == other.offset()
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

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        crate::rle::array::SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == crate::rle::array::NUM_SLOTS,
            "RLEArray expects {} slots, got {}",
            crate::rle::array::NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn serialize(array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
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
    ) -> VortexResult<RLEData> {
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

        RLEData::try_new(
            values,
            indices,
            values_idx_offsets,
            metadata.offset as usize,
            len,
        )
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
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.rle");

    pub fn try_new(
        values: ArrayRef,
        indices: ArrayRef,
        values_idx_offsets: ArrayRef,
        offset: usize,
        length: usize,
    ) -> VortexResult<RLEArray> {
        let dtype = DType::Primitive(values.dtype().as_ptype(), indices.dtype().nullability());
        let data = RLEData::try_new(values, indices, values_idx_offsets, offset, length)?;
        Ok(unsafe { Array::from_parts_unchecked(ArrayParts::new(RLE, dtype, length, data)) })
    }

    /// Create a new RLE array without validation.
    ///
    /// # Safety
    /// See [`RLEData::new_unchecked`] for preconditions.
    pub unsafe fn new_unchecked(
        values: ArrayRef,
        indices: ArrayRef,
        values_idx_offsets: ArrayRef,
        offset: usize,
        length: usize,
    ) -> RLEArray {
        let dtype = DType::Primitive(values.dtype().as_ptype(), indices.dtype().nullability());
        let data = unsafe { RLEData::new_unchecked(values, indices, values_idx_offsets, offset) };
        unsafe { Array::from_parts_unchecked(ArrayParts::new(RLE, dtype, length, data)) }
    }

    /// Encode a primitive array using FastLanes RLE.
    pub fn encode(array: &PrimitiveArray) -> VortexResult<RLEArray> {
        RLEData::encode(array)
    }
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
