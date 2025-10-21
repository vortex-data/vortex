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
    #[prost(uint64, tag = "6", default = "0")]
    pub offset: u64,
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
            offset: array.offset() as u64,
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

        RLEArray::try_new(
            values,
            indices,
            values_idx_offsets,
            metadata.offset as usize,
            len,
        )
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
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::serde::{ArrayParts, SerializeOptions};
    use vortex_array::test_harness::check_metadata;
    use vortex_array::{Array, ArrayContext, EncodingRef, ToCanonical};
    use vortex_buffer::ByteBufferMut;

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
                offset: u64::MAX,
            }),
        );
    }

    #[test]
    fn test_rle_serialization() {
        let primitive = PrimitiveArray::from_iter((0..2048).map(|i| (i / 100) as u32));
        let rle_array = RLEArray::encode(&primitive).unwrap();
        assert_eq!(rle_array.len(), 2048);

        let original_data = rle_array.to_primitive();
        let original_values = original_data.as_slice::<u32>();

        let ctx = ArrayContext::empty().with(EncodingRef::new_ref(RLEEncoding.as_ref()));
        let serialized = rle_array
            .to_array()
            .serialize(&ctx, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = ArrayParts::try_from(concat).unwrap();
        let decoded = parts
            .decode(
                &ctx,
                &DType::Primitive(PType::U32, Nullability::NonNullable),
                2048,
            )
            .unwrap();

        let decoded_data = decoded.to_primitive();
        let decoded_values = decoded_data.as_slice::<u32>();

        assert_eq!(original_values, decoded_values);
    }

    #[test]
    fn test_rle_serialization_slice() {
        let primitive = PrimitiveArray::from_iter((0..2048).map(|i| (i / 100) as u32));
        let rle_array = RLEArray::encode(&primitive).unwrap();
        let sliced = rle_array.slice(100..200);
        assert_eq!(sliced.len(), 100);

        let ctx = ArrayContext::empty().with(EncodingRef::new_ref(RLEEncoding.as_ref()));
        let serialized = sliced
            .serialize(&ctx, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = ArrayParts::try_from(concat).unwrap();
        let decoded = parts.decode(&ctx, sliced.dtype(), sliced.len()).unwrap();

        let original_data = sliced.to_primitive();
        let decoded_data = decoded.to_primitive();

        let original_values = original_data.as_slice::<u32>();
        let decoded_values = decoded_data.as_slice::<u32>();

        assert_eq!(original_values, decoded_values);
    }
}
