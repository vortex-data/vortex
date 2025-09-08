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
use vortex_error::{VortexResult, vortex_bail, vortex_ensure};

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

        vortex_ensure!(indices.dtype().nullability() == Nullability::NonNullable);
        vortex_ensure!(value_chunk_offsets.dtype().nullability() == Nullability::NonNullable);

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
        visitor.visit_child("value_chunks_offsets", array.value_chunk_offsets());
        visitor.visit_validity(array.validity(), array.len());
    }
}

#[cfg(test)]
mod test {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::serde::ArrayChildren;
    use vortex_array::test_harness::check_metadata;
    use vortex_array::vtable::SerdeVTable;
    use vortex_array::{IntoArray, ProstMetadata, ToCanonical};
    use vortex_error::vortex_err;

    use super::*;
    use crate::rle::RLEVTable;

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

    #[test]
    fn test_serde_build() {
        let original = PrimitiveArray::from_iter([1u32, 1, 1, 2, 2, 3, 3, 3, 3]).into_array();

        let encoded = RLEVTable::encode(&RLEEncoding, &original.to_canonical(), None)
            .unwrap()
            .unwrap();

        let metadata = RLEVTable::metadata(&encoded).unwrap().unwrap();
        let encoded_values = encoded.values().to_array();
        let encoded_indices = encoded.indices().to_array();
        let encoded_value_chunk_offsets = encoded.value_chunk_offsets().to_array();

        assert_eq!(encoded_values.len(), 3);
        // Indices get padded to 1024.
        assert_eq!(encoded_indices.len(), 1024);
        assert_eq!(encoded.len(), 9);

        // Verify metadata has correct lengths
        assert_eq!(metadata.0.values_len, 3);
        assert_eq!(metadata.0.indices_len, 1024);
        assert_eq!(
            metadata.0.value_chunk_offsets_len as usize,
            encoded_value_chunk_offsets.len()
        );

        struct MockArray(Vec<vortex_array::ArrayRef>);

        impl ArrayChildren for MockArray {
            fn len(&self) -> usize {
                self.0.len()
            }
            fn get(
                &self,
                index: usize,
                _dtype: &DType,
                _len: usize,
            ) -> VortexResult<vortex_array::ArrayRef> {
                self.0
                    .get(index)
                    .cloned()
                    .ok_or_else(|| vortex_err!("Index out of bounds"))
            }
        }

        let rebuilt = RLEVTable::build(
            &RLEEncoding,
            encoded.dtype(),
            encoded.len(),
            &metadata.0,
            &[],
            &MockArray(vec![
                encoded_values.clone(),
                encoded_indices.clone(),
                encoded_value_chunk_offsets.clone(),
            ]),
        )
        .unwrap();

        assert_eq!(rebuilt.len(), encoded.len());

        assert_eq!(
            rebuilt.indices().to_primitive().as_slice::<u16>(),
            encoded_indices.to_primitive().as_slice::<u16>()
        );

        assert_eq!(
            rebuilt.values().to_primitive().as_slice::<u32>(),
            encoded_values.to_primitive().as_slice::<u32>()
        );

        assert_eq!(
            rebuilt
                .value_chunk_offsets()
                .to_primitive()
                .as_slice::<u64>(),
            encoded_value_chunk_offsets.to_primitive().as_slice::<u64>()
        );

        assert_eq!(
            original.to_primitive().as_slice::<u32>(),
            rebuilt.to_primitive().as_slice::<u32>()
        );
    }
}
