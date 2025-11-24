// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{
    ArrayId, ArrayVTable, ArrayVTableExt, NotSupported, VTable, ValidityVTableFromChildSliceHelper,
};
use vortex_array::{ProstMetadata, vtable};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::VortexResult;

use crate::RLEArray;

mod array;
mod canonical;
mod encode;
mod operations;
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
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChildSliceHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type OperatorVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("fastlanes.rle")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        RLEVTable.as_vtable()
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
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
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
}

#[derive(Clone, Debug)]
pub struct RLEVTable;

#[cfg(test)]
mod tests {
    use vortex_array::test_harness::check_metadata;

    use super::{ProstMetadata, RLEMetadata};

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
