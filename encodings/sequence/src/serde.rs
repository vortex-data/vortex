// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable};
use vortex_array::{Canonical, DeserializeMetadata, ProstMetadata};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
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
