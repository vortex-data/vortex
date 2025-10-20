// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VTable, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, Canonical, DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::builders::dict_encode;
use crate::{DictArray, DictEncoding, DictVTable};

#[derive(Clone, prost::Message)]
pub struct DictMetadata {
    #[prost(uint32, tag = "1")]
    pub values_len: u32,
    #[prost(enumeration = "PType", tag = "2")]
    pub codes_ptype: i32,
    // nullable codes are optional since they were added after stabilisation
    #[prost(optional, bool, tag = "3")]
    pub is_nullable_codes: Option<bool>,
}

impl SerdeVTable<DictVTable> for DictVTable {
    fn build(
        _encoding: &DictEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<<DictVTable as VTable>::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DictArray> {
        if children.len() != 2 {
            vortex_bail!(
                "Expected 2 children for dict encoding, found {}",
                children.len()
            )
        }
        let codes_nullable: Nullability = metadata
            .is_nullable_codes
            // The old behaviour of (without `is_nullable_codes` metadata) used the nullability
            // of the values (and whole array).
            .unwrap_or_else(|| dtype.is_nullable())
            .into();
        let codes_dtype = DType::Primitive(metadata.codes_ptype(), codes_nullable);
        let codes = children.get(0, &codes_dtype, len)?;
        let values = children.get(1, dtype, metadata.values_len as usize)?;

        DictArray::try_new(codes, values)
    }
}

impl EncodeVTable<DictVTable> for DictVTable {
    fn encode(
        _encoding: &DictEncoding,
        canonical: &Canonical,
        _like: Option<&DictArray>,
    ) -> VortexResult<Option<DictArray>> {
        Ok(Some(dict_encode(canonical.as_ref())?))
    }
}

impl VisitorVTable<DictVTable> for DictVTable {
    fn metadata(array: &DictArray) -> <DictVTable as VTable>::Metadata {
        ProstMetadata(DictMetadata {
            codes_ptype: array.codes().dtype().as_ptype() as i32,
            values_len: u32::try_from(array.values().len())
                .vortex_expect("Dictionary values size overflowed u32"),
            is_nullable_codes: Some(array.codes().dtype().is_nullable()),
        })
    }

    fn visit_buffers(_array: &DictArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &DictArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("codes", array.codes());
        visitor.visit_child("values", array.values());
    }
}

#[cfg(test)]
mod test {
    use vortex_array::ProstMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::serde::DictMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_dict_metadata() {
        check_metadata(
            "dict.metadata",
            ProstMetadata(DictMetadata {
                codes_ptype: PType::U64 as i32,
                values_len: u32::MAX,
                is_nullable_codes: None,
            }),
        );
    }
}
