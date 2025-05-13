use std::fmt::Formatter;

use vortex_array::serde::ArrayParts;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, Canonical,
    DeserializeMetadata, SerializeMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_scalar::{Scalar, ScalarValue};

use super::FoREncoding;
use crate::{FoRArray, FoRVTable};

impl SerdeVTable<FoRVTable> for FoRVTable {
    type Metadata = ScalarValueMetadata;

    fn metadata(array: &FoRArray) -> Option<Self::Metadata> {
        Some(ScalarValueMetadata(
            array.reference_scalar().value().clone(),
        ))
    }

    fn decode(
        _encoding: &FoREncoding,
        dtype: DType,
        len: usize,
        metadata: &ScalarValue,
        _buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<FoRArray> {
        if children.len() != 1 {
            vortex_bail!(
                "Expected 1 child for FoR encoding, found {}",
                children.len()
            )
        }

        let ptype = PType::try_from(&dtype)?;
        let encoded_dtype = DType::Primitive(ptype.to_unsigned(), dtype.nullability());
        let encoded = children[0].decode(ctx, encoded_dtype, len)?;

        let reference = Scalar::new(dtype, metadata.clone());

        FoRArray::try_new(encoded, reference)
    }
}

impl EncodeVTable<FoRVTable> for FoRVTable {
    fn encode(
        _encoding: &FoREncoding,
        canonical: &Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<FoRArray>> {
        let parray = canonical.clone().into_primitive()?;
        Ok(Some(FoRArray::encode(parray)?))
    }
}

impl VisitorVTable<FoRVTable> for FoRVTable {
    fn visit_buffers(_array: &FoRArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &FoRArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", array.encoded())
    }

    fn with_children(array: &FoRArray, children: &[ArrayRef]) -> VortexResult<FoRArray> {
        if children.len() != 1 {
            vortex_bail!(
                "Expected 1 child for FoR encoding, found {}",
                children.len()
            )
        }
        FoRArray::try_new(children[0].clone(), array.reference_scalar().clone())
    }
}

#[derive(Clone, Debug)]
pub struct ScalarValueMetadata(ScalarValue);

impl SerializeMetadata for ScalarValueMetadata {
    fn serialize(&self) -> Option<Vec<u8>> {
        Some(self.0.to_protobytes())
    }
}

impl DeserializeMetadata for ScalarValueMetadata {
    type Output = ScalarValue;

    fn deserialize(metadata: Option<&[u8]>) -> VortexResult<Self::Output> {
        ScalarValue::from_protobytes(
            metadata.ok_or_else(|| vortex_err!("Missing ScalarValue metadata"))?,
        )
    }

    fn format(metadata: Option<&[u8]>, f: &mut Formatter<'_>) -> std::fmt::Result {
        Self::deserialize(metadata)
            .map(|value| write!(f, "{}", value))
            .unwrap_or_else(|_| write!(f, "<unknown>"))
    }
}
