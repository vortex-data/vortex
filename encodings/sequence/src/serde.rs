// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable};
use vortex_array::{Canonical, DeserializeMetadata, ProstMetadata};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_proto::scalar::ScalarValue;
use vortex_scalar::Scalar;

use crate::array::{SequenceArray, SequenceEncoding, SequenceVTable};

#[derive(Clone, prost::Message)]
pub struct SequenceMetadata {
    #[prost(message, tag = "1")]
    base: Option<ScalarValue>,
    #[prost(message, tag = "2")]
    multiplier: Option<ScalarValue>,
}

impl EncodeVTable<SequenceVTable> for SequenceVTable {
    fn encode(
        _encoding: &SequenceEncoding,
        _canonical: &Canonical,
        _like: Option<&SequenceArray>,
    ) -> VortexResult<Option<SequenceArray>> {
        // TODO(joe): hook up compressor
        Ok(None)
    }
}

impl SerdeVTable<SequenceVTable> for SequenceVTable {
    type Metadata = ProstMetadata<SequenceMetadata>;

    fn metadata(array: &SequenceArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(SequenceMetadata {
            base: Some((&array.base()).into()),
            multiplier: Some((&array.multiplier()).into()),
        })))
    }

    fn build(
        _encoding: &SequenceEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<SequenceArray> {
        let ptype = dtype.as_ptype();

        // We go via scalar to cast the scalar values into the correct PType
        let base = Scalar::new(
            DType::Primitive(ptype, NonNullable),
            metadata
                .base
                .as_ref()
                .ok_or_else(|| vortex_err!("base required"))?
                .try_into()?,
        )
        .as_primitive()
        .pvalue()
        .vortex_expect("non-nullable primitive");

        let multiplier = Scalar::new(
            DType::Primitive(ptype, NonNullable),
            metadata
                .multiplier
                .as_ref()
                .ok_or_else(|| vortex_err!("base required"))?
                .try_into()?,
        )
        .as_primitive()
        .pvalue()
        .vortex_expect("non-nullable primitive");

        Ok(SequenceArray::unchecked_new(
            base,
            multiplier,
            ptype,
            dtype.nullability(),
            len,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::ToCanonical;
    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_array::iter::ArrayIteratorExt;
    use vortex_dtype::Nullability;
    use vortex_expr::{get_item, root};
    use vortex_file::{VortexOpenOptions, VortexWriteOptions};
    use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;

    use crate::SequenceArray;

    #[tokio::test]
    async fn round_trip_seq() {
        let seq = SequenceArray::typed_new(2i8, 3, Nullability::NonNullable, 4).unwrap();
        let st = StructArray::from_fields(&[("a", seq.to_array())]).unwrap();

        let file = tokio::fs::File::create("/tmp/abc.vx").await.unwrap();
        VortexWriteOptions::default()
            .with_strategy(Arc::new(FlatLayoutStrategy::default()))
            .write(file, st.to_array_stream())
            .await
            .unwrap();

        let file = VortexOpenOptions::file().open("/tmp/abc.vx").await.unwrap();
        let array = file
            .scan()
            .unwrap()
            .with_projection(get_item("a", root()))
            .into_array_iter()
            .unwrap()
            .read_all()
            .unwrap();

        let canon = PrimitiveArray::from_iter((0..4).map(|i| 2i8 + i * 3));

        assert_eq!(
            array.to_primitive().as_slice::<i8>(),
            canon.as_slice::<i8>()
        )
    }
}
