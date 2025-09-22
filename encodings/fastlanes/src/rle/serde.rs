// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, ValidityHelper, VisitorVTable};
use vortex_array::{
    ArrayBufferVisitor, ArrayChildVisitor, Canonical, DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexResult, vortex_bail};

use super::RLEEncoding;
use crate::{RLEArray, RLEVTable};

#[derive(Clone, prost::Message)]
pub struct RLEMetadata {
    #[prost(uint64, tag = "1")]
    pub values_len: u64,
    #[prost(uint64, tag = "2")]
    pub indices_len: u64,
    #[prost(uint64, tag = "3")]
    pub value_chunk_offsets_len: u64,
}

impl SerdeVTable<RLEVTable> for RLEVTable {
    type Metadata = ProstMetadata<RLEMetadata>;

    fn metadata(array: &RLEArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(RLEMetadata {
            values_len: array.values().len() as u64,
            indices_len: array.indices().len() as u64,
            value_chunk_offsets_len: array.value_chunk_offsets().len() as u64,
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
        let validity = if children.len() == 3 {
            // `visit_validity` does not add a child array in case
            // the validity is either `NonNullable` or `AllValid`.
            Validity::from(dtype.nullability())
        } else if children.len() == 4 {
            let validity = children.get(3, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("RLEArray: expected 3 or 4 children, got {}", children.len());
        };

        let values = children.get(
            0,
            &DType::Primitive(dtype.as_ptype(), dtype.nullability()),
            usize::try_from(metadata.values_len)?,
        )?;

        let indices = children.get(
            1,
            &DType::Primitive(PType::U16, Nullability::NonNullable),
            usize::try_from(metadata.indices_len)?,
        )?;

        let value_chunk_offsets = children.get(
            2,
            &DType::Primitive(PType::U64, Nullability::NonNullable),
            usize::try_from(metadata.value_chunk_offsets_len)?,
        )?;

        RLEArray::try_new(values, indices, value_chunk_offsets, validity, len)
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
        visitor.visit_child("value_chunk_offsets", array.value_chunk_offsets());
        visitor.visit_validity(array.validity(), array.len());
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
                value_chunk_offsets_len: u64::MAX,
            }),
        );
    }
}
