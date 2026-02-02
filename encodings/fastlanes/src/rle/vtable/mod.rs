// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::ProstMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromChildSliceHelper;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::RLEArray;
use crate::rle::array::rle_decompress::rle_decompress;
use crate::rle::kernel::PARENT_KERNELS;

mod array;
mod operations;
mod rules;
mod validity;
mod visitor;

vtable!(RLE);

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

impl VTable for RLEVTable {
    type Array = RLEArray;

    type Metadata = ProstMetadata<RLEMetadata>;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChildSliceHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn slice(array: &Self::Array, range: std::ops::Range<usize>) -> VortexResult<Option<ArrayRef>> {
        use vortex_array::IntoArray;

        use crate::FL_CHUNK_SIZE;

        let offset_in_chunk = array.offset();
        let chunk_start_idx = (offset_in_chunk + range.start) / FL_CHUNK_SIZE;
        let chunk_end_idx = (offset_in_chunk + range.end).div_ceil(FL_CHUNK_SIZE);

        let values_start_idx = array.values_idx_offset(chunk_start_idx);
        let values_end_idx = if chunk_end_idx < array.values_idx_offsets().len() {
            array.values_idx_offset(chunk_end_idx)
        } else {
            array.values().len()
        };

        let sliced_values = array.values().slice(values_start_idx..values_end_idx)?;

        let sliced_values_idx_offsets = array
            .values_idx_offsets()
            .slice(chunk_start_idx..chunk_end_idx)?;

        let sliced_indices = array
            .indices()
            .slice(chunk_start_idx * FL_CHUNK_SIZE..chunk_end_idx * FL_CHUNK_SIZE)?;

        // SAFETY: Slicing preserves all invariants.
        Ok(Some(unsafe {
            RLEArray::new_unchecked(
                sliced_values,
                sliced_indices,
                sliced_values_idx_offsets,
                array.dtype().clone(),
                // Keep the offset relative to the first chunk.
                (array.offset() + range.start) % FL_CHUNK_SIZE,
                range.len(),
            )
            .into_array()
        }))
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        // RLEArray children order (from visit_children):
        // 1. values
        // 2. indices
        // 3. values_idx_offsets

        vortex_ensure!(
            children.len() == 3,
            "Expected 3 children for RLE encoding, got {}",
            children.len()
        );

        array.values = children[0].clone();
        array.indices = children[1].clone();
        array.values_idx_offsets = children[2].clone();

        Ok(())
    }

    fn metadata(array: &RLEArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(RLEMetadata {
            values_len: array.values().len() as u64,
            indices_len: array.indices().len() as u64,
            indices_ptype: PType::try_from(array.indices().dtype())? as i32,
            values_idx_offsets_len: array.values_idx_offsets().len() as u64,
            values_idx_offsets_ptype: PType::try_from(array.values_idx_offsets().dtype())? as i32,
            offset: array.offset() as u64,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.0.encode_to_vec()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(RLEMetadata::decode(buffer)?))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<RLEArray> {
        let metadata = &metadata.0;
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

        RLEArray::try_new(
            values,
            indices,
            values_idx_offsets,
            metadata.offset as usize,
            len,
        )
    }

    fn execute_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        Ok(Canonical::Primitive(rle_decompress(array, ctx)?))
    }

    fn reduce_parent(
        array: &RLEArray,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        rules::RULES.evaluate(array, parent, child_idx)
    }
}

#[derive(Debug)]
pub struct RLEVTable;

impl RLEVTable {
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.rle");
}

#[cfg(test)]
mod tests {
    use vortex_array::test_harness::check_metadata;

    use super::ProstMetadata;
    use super::RLEMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_rle_metadata() {
        check_metadata(
            "rle.metadata",
            ProstMetadata(RLEMetadata {
                values_len: u64::MAX,
                indices_len: u64::MAX,
                indices_ptype: i32::MAX,
                values_idx_offsets_len: u64::MAX,
                values_idx_offsets_ptype: i32::MAX,
                offset: u64::MAX,
            }),
        );
    }
}
