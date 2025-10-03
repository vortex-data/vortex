// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{
    ArrayBufferVisitor, ArrayChildVisitor, Canonical, DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::VortexResult;

use super::RLEEncoding;
use crate::{RLEArray, RLEVTable};

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
}

impl SerdeVTable<RLEVTable> for RLEVTable {
    type Metadata = ProstMetadata<RLEMetadata>;

    fn metadata(array: &RLEArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(RLEMetadata {
            values_len: array.values().len() as u64,
            indices_len: array.indices().len() as u64,
            indices_ptype: PType::try_from(array.indices().dtype())? as i32,
            values_idx_offsets_len: array.values_idx_offsets().len() as u64,
            values_idx_offsets_ptype: PType::try_from(array.values_idx_offsets().dtype())? as i32,
        })))
    }

    fn build(
        _encoding: &RLEEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<RLEArray> {
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

        RLEArray::try_new(values, indices, values_idx_offsets, len)
    }
}

impl EncodeVTable<RLEVTable> for RLEVTable {
    fn encode(
        _encoding: &RLEEncoding,
        canonical: &Canonical,
        _like: Option<&RLEArray>,
    ) -> VortexResult<Option<RLEArray>> {
        let array = canonical.clone().into_primitive();
        Ok(Some(RLEArray::encode(&array)?))
    }
}

impl VisitorVTable<RLEVTable> for RLEVTable {
    fn visit_buffers(_array: &RLEArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // RLE stores all data in child arrays, no direct buffers
    }

    fn visit_children(array: &RLEArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("values", array.values());
        visitor.visit_child("indices", array.indices());
        visitor.visit_child("values_idx_offsets", array.values_idx_offsets());
        // Don't call visit_validity since the nullability is stored in the indices array.
    }
}

#[cfg(test)]
mod test {
    use vortex_array::test_harness::check_metadata;

    use super::*;

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
            }),
        );
    }
}
